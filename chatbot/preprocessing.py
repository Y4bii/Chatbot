"""Aufbereitung der abgerufenen Seiten und PDF-Dokumente.

Aus jeder Seite wird nur der Inhaltsbereich übernommen. Überschriften, Listen und
Tabellen bleiben als Gliederung erhalten. Links auf Dokumente und auf Seiten außerhalb
der Wissensbasis behalten ihre Adresse; Links auf Seiten, die selbst Teil der
Wissensbasis sind, erscheinen nur mit ihrem Text. Textblöcke, die auf
vielen Seiten gleich sind, etwa Kontaktkästen, werden nur einmal behalten, damit sie die
Suche nicht verzerren, ihre Information aber nicht verloren geht. Aus PDF-Dokumenten
wird der enthaltene Text gelesen. Dokumente ohne lesbaren Text entfallen.

Das Ergebnis ist eine Liste von Dokumenten, die jeweils in Abschnitte entlang ihrer
Überschriften gegliedert sind.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import PreformattedString
from pypdf import PdfReader

from chatbot.config import CrawlerConfig, PreprocessingConfig

logger = logging.getLogger(__name__)
# pypdf meldet bei fehlerhaft aufgebauten PDFs viele unkritische Warnungen.
logging.getLogger("pypdf").setLevel(logging.ERROR)

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCK_TAGS = HEADING_TAGS | {
    "p", "div", "section", "article", "header", "footer", "aside", "main", "nav",
    "ul", "ol", "li", "table", "tr", "td", "th", "dl", "dt", "dd", "blockquote", "pre",
    "figure", "figcaption", "address", "details", "summary", "form", "hr",
}
_WHITESPACE = re.compile(r"\s+")
# Bauteil der Webseite für Regelstudienpläne und Modulkennung wie „GW3510-KP05“
_STUDY_PLAN_CLASS = "studienplan-container"
_MODULE_CODE = re.compile(r"[A-Z]{2,3}\d{4}(?:-KP\d+)?")
# Semesterkopf wie „4. Semester (30KP)“ oder „4. Semester (30 KP, 4 Prüfungen)“
_SEMESTER_HEAD = re.compile(r"(\d+)\.\s*Semester(?:\s*\((\d+)\s*KP)?")
# Linien und Füllzeichen in PDF-Dokumenten, etwa "_____" oder Punktreihen in Inhaltsverzeichnissen
_FILLER = re.compile(r"([._\-=·…])\1{3,}")
# Linktexte, die nichts über das Ziel aussagen und daher nicht als Titel taugen
_GENERIC_LINK_TEXTS = {"hier", "pdf", "download", "link", "mehr", "weiter", "weitere informationen", "datei"}
# Semesterangabe wie "WS 2019/2020", "SoSe 24" oder "Wintersemester 2016"
_SEMESTER = re.compile(r"\b(WS|WiSe|Wintersemester|SS|SoSe|Sommersemester)\s*(\d{2,4})\b", re.IGNORECASE)


def semester_key(text: str) -> tuple[int, int] | None:
    """Sortierschlüssel (Jahr, Semester) der ersten Semesterangabe im Text, Wintersemester nach Sommersemester."""
    match = _SEMESTER.search(text)
    if not match:
        return None
    year = int(match.group(2)[:4]) if len(match.group(2)) >= 4 else 2000 + int(match.group(2))
    winter = match.group(1).lower() in ("ws", "wise", "wintersemester")
    return year, int(winter)


@dataclass
class Section:
    """Ein Abschnitt eines Dokuments mit dem Pfad seiner übergeordneten Überschriften."""

    headings: list[str]
    text: str


@dataclass
class Document:
    """Eine aufbereitete Seite oder ein aufbereitetes PDF-Dokument."""

    url: str
    title: str
    breadcrumb: list[str]
    kind: str
    retrieved_at: str
    sections: list[Section] = field(default_factory=list)
    # Fassung einer Regelung: "current", "older" oder None, wenn nicht bekannt
    version: str | None = None
    # Bei PDF-Dokumenten: Anfang jeder Seite im Text als (Zeichenposition, Seitenzahl)
    page_starts: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class PrepareStats:
    """Zusammenfassung der Aufbereitung."""

    pages: int = 0
    documents: int = 0
    excluded: int = 0
    unreadable_documents: int = 0
    empty: int = 0
    duplicates: int = 0
    boilerplate_blocks: int = 0
    boilerplate_removed: int = 0
    versioned_by_year: int = 0
    overviews: int = 0


def _previous_is_link(node: Tag) -> bool:
    """Ist das vorhergehende Element ein Link, ohne sichtbaren Text dazwischen?

    HTML-Kommentare und Leerraum zwischen den Links werden übergangen.
    """
    for sibling in node.previous_siblings:
        if isinstance(sibling, PreformattedString):
            continue
        if isinstance(sibling, NavigableString):
            if sibling.strip():
                return False
            continue
        return isinstance(sibling, Tag) and sibling.name == "a"
    return False


def normalize_url(url: str) -> str:
    """Vereinheitlicht eine Adresse wie beim Abruf: ohne Sprungmarke, immer ``https``."""
    return url.split("#", 1)[0].replace("http://", "https://", 1)


def clean_text(text: str) -> str:
    """Vereinheitlicht Leerraum innerhalb von Zeilen und entfernt leere Zeilen."""
    lines = (" ".join(line.split()) for line in text.replace("\xa0", " ").splitlines())
    return "\n".join(line for line in lines if line)


def clean_pdf_text(text: str) -> str:
    """Wie :func:`clean_text`, entfernt zusätzlich Linien und Füllzeichen aus PDF-Dokumenten."""
    return clean_text(_FILLER.sub(" ", text))


class HtmlConverter:
    """Wandelt den Inhaltsbereich einer Seite in eine Folge von Überschriften und Textblöcken um."""

    def __init__(self, base_url: str, known_pages: frozenset[str] = frozenset()) -> None:
        """
        Args:
            base_url: Adresse, auf die sich relative Links beziehen.
            known_pages: Vereinheitlichte Adressen der Seiten in der Wissensbasis. Links auf
                diese Seiten werden ohne Adresse wiedergegeben.
        """
        self.base_url = base_url
        self.known_pages = known_pages
        self.blocks: list[tuple[int, str]] = []
        # Beschriftung je Linkziel aus letzter Überschrift und Linktext, z. B. als Titel verlinkter PDFs
        self.link_labels: dict[str, str] = {}
        # Fassung je Linkziel ("current" oder "older"), sofern es in einer Tabelle von Fassungen steht
        self.link_versions: dict[str, str] = {}
        self._inline: list[str] = []
        self._last_heading = ""
        self._last_level = 1
        # Geltungsbereich und Fassung der Tabellenzeile, die gerade verarbeitet wird
        self._row_context: tuple[str, str] | None = None

    def convert(self, node: Tag) -> list[tuple[int, str]]:
        """Gibt die Blöcke als ``(Ebene, Text)`` zurück, Ebene 1 bis 6 für Überschriften, 0 für Text."""
        self._walk(node)
        self._flush()
        return self.blocks

    def _flush(self) -> None:
        """Schließt den aktuellen Absatz ab."""
        text = clean_text("".join(self._inline))
        self._inline = []
        if text:
            self.blocks.append((0, text))

    def _walk(self, node: Tag) -> None:
        for child in node.children:
            if isinstance(child, PreformattedString):
                continue
            if isinstance(child, NavigableString):
                # Wie im Browser fällt Leerraum im HTML-Quelltext zu einem Leerzeichen zusammen.
                self._inline.append(_WHITESPACE.sub(" ", str(child)))
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in HEADING_TAGS:
                self._flush()
                text = " ".join(child.get_text(" ").split())
                if text:
                    self.blocks.append((int(name[1]), text))
                    self._last_heading = text
                    self._last_level = int(name[1])
            elif name == "div" and _STUDY_PLAN_CLASS in (child.get("class") or []):
                self._flush()
                self._study_plan(child)
            elif name == "table":
                self._flush()
                self._table(child)
            elif name in ("ul", "ol"):
                self._flush()
                lines: list[str] = []
                self._list(child, 0, lines)
                if lines:
                    self.blocks.append((0, "\n".join(lines)))
            elif name == "br":
                self._inline.append("\n")
            elif name == "a":
                # Folgen Links ohne Text dazwischen aufeinander, etwa die Module eines
                # Regelstudienplans, steht jeder auf einer eigenen Zeile.
                if _previous_is_link(child):
                    self._inline.append("\n")
                self._inline.append(self._link(child))
            elif name in BLOCK_TAGS:
                self._flush()
                self._walk(child)
                self._flush()
            else:
                self._walk(child)

    def _study_plan(self, container: Tag) -> None:
        """Gibt einen Regelstudienplan als Unterüberschrift je Semester mit dessen Modulen wieder.

        Die Webseite zeigt die Zahl der Prüfungen je Semester in einer eigenen Zeile unter allen
        Semestern; sie wird dem jeweiligen Semester zugeordnet. Module, die über mehrere Semester
        laufen, stehen für die mobile Ansicht mehrfach im Quelltext und erscheinen je Semester nur
        einmal. Legende und Farbkategorien des Plans entfallen.
        """
        semesters = container.select("div.semester")
        exams = [" ".join(exam.get_text(" ").split()) for exam in container.select("div.exam-counts div.exam")]
        plan_heading, plan_level = self._last_heading, self._last_level
        level = min(plan_level + 1, 6)
        for index, semester in enumerate(semesters):
            head = semester.select_one("div.semester-count")
            title = " ".join(head.get_text(" ").split()) if head else f"{index + 1}. Semester"
            # Die Prüfungszahlen gelten nur, wenn es für jedes Semester genau eine gibt.
            exam = exams[index] if len(exams) == len(semesters) else ""
            if exam:
                title = f"{title[:-1]}, {exam})" if title.endswith(")") else f"{title} ({exam})"
            modules: dict[str, str] = {}
            for anchor in semester.select("a.module"):
                text = " ".join(self._link(anchor).replace("\xad", "").split())
                if not text:
                    continue
                code = _MODULE_CODE.match(text)
                modules.setdefault(code.group(0) if code else text, text)
            self.blocks.append((level, title))
            self._last_heading = title
            # Ein einleitender Satz nennt Semester, Umfang und Prüfungen ausgeschrieben. Nur mit
            # Zahlen in der Überschrift verwechselte die Suche Semester, etwa „4. Semester“ mit
            # „1. Semester (30KP, 4 Prüfungen)“.
            number = _SEMESTER_HEAD.match(title)
            if number:
                credits = f"{number.group(2)} KP" if number.group(2) else ""
                details = " und ".join(part for part in (credits, exam) if part)
                summary = f"Im {number.group(1)}. Semester sind {details} vorgesehen." if details else ""
                lead = f"Module im {number.group(1)}. Semester:"
                self.blocks.append((0, f"{summary}\n{lead}".strip()))
            if modules:
                self.blocks.append((0, "\n".join(modules.values())))
        # Inhalte nach dem Plan, etwa der Link auf das PDF, gehören wieder zur Plan-Überschrift.
        if plan_heading:
            self.blocks.append((plan_level, plan_heading))
            self._last_heading = plan_heading

    def _render_inline(self, node: Tag, skip: tuple[str, ...] = ()) -> str:
        """Gibt den Text eines Elements einschließlich Links zurück, ohne die Elemente in ``skip``."""
        parts = []
        for child in node.children:
            if isinstance(child, PreformattedString):
                continue
            if isinstance(child, NavigableString):
                parts.append(_WHITESPACE.sub(" ", str(child)))
            elif isinstance(child, Tag) and child.name not in skip:
                if child.name == "br":
                    parts.append("\n")
                elif child.name == "a":
                    parts.append(self._link(child))
                else:
                    inner = self._render_inline(child, skip)
                    parts.append(f" {inner} " if child.name in BLOCK_TAGS else inner)
        return "".join(parts)

    def _link(self, anchor: Tag) -> str:
        """Gibt einen Link als ``Text (Adresse)`` wieder, damit auf das Ziel verwiesen werden kann.

        Ist das Ziel selbst eine Seite der Wissensbasis, bleibt nur der Text. Die Seite kann
        dann selbst als Quelle erscheinen, und die Adressen blähen Chunks mit vielen Links,
        etwa Regelstudienpläne, nicht auf.
        """
        text = " ".join(anchor.get_text(" ").split())
        href = (anchor.get("href") or "").strip()
        if not text or not href or href.startswith(("#", "javascript:")):
            return text
        url = urljoin(self.base_url, href)
        if text.lower() not in _GENERIC_LINK_TEXTS and len(text) >= 4:
            label = f"{self._last_heading}: {text}" if self._last_heading and self._last_heading != text else text
            if self._row_context:
                label = f"{label} ({self._row_context[0]})"
            self.link_labels.setdefault(url, label)
        if self._row_context:
            self.link_versions.setdefault(url, self._row_context[1])
        if url == text or normalize_url(url) in self.known_pages:
            return text
        return f"{text} ({url})"

    def _list(self, node: Tag, depth: int, lines: list[str]) -> None:
        """Gibt eine (verschachtelte) Liste als eingerückte Aufzählung wieder."""
        for item in node.find_all("li", recursive=False):
            text = clean_text(self._render_inline(item, skip=("ul", "ol"))).replace("\n", " ")
            if text:
                lines.append("  " * depth + "- " + text)
            for sublist in item.find_all(["ul", "ol"]):
                if sublist.find_parent("li") is item:
                    self._list(sublist, depth + 1, lines)

    def _table(self, table: Tag) -> None:
        """Gibt eine Tabelle zeilenweise wieder.

        Hat die Tabelle eine Kopfzeile, wird jede Zelle mit ihrer Spaltenüberschrift
        versehen. So bleibt eine Zeile auch dann verständlich, wenn sie ohne die
        Kopfzeile in einem Chunk landet.

        Steht in der ersten Spalte eine Semesterangabe, etwa „Studienbeginn ab: WS 2019/2020“
        bei den Studiengangsordnungen, sind die Zeilen Fassungen derselben Regelung.
        Verlinkte Dokumente erhalten dann ihren Geltungsbereich und den Hinweis, ob es die
        aktuelle (jüngste) oder eine ältere Fassung ist. Ältere Fassungen gelten weiterhin
        für Studierende mit früherem Studienbeginn.
        """
        rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
        if not rows:
            return
        first_cells = rows[0].find_all(["th", "td"], recursive=False)
        header = None
        if first_cells and all(cell.name == "th" for cell in first_cells):
            header = [" ".join(cell.get_text(" ").split()) for cell in first_cells]
            rows = rows[1:]

        # Geltungsbereich je Zeile aus der ersten Spalte, z. B. "Studienbeginn ab WS 2019/2020"
        contexts = []
        for row in rows:
            cells = row.find_all(["th", "td"], recursive=False)
            first = " ".join(cells[0].get_text(" ").split()) if cells else ""
            contexts.append(f"{header[0].rstrip(':')} {first}".strip() if header and header[0] else first)
        keys = [semester_key(context) for context in contexts]
        newest = max((key for key in keys if key), default=None)

        lines = []
        for row, context, key in zip(rows, contexts, keys):
            if key is not None:
                version = "current" if key == newest else "older"
                status = "aktuelle Fassung" if version == "current" else "ältere Fassung"
                self._row_context = (f"{status}, {context}", version)
            cells = row.find_all(["th", "td"], recursive=False)
            texts = [clean_text(self._render_inline(cell)).replace("\n", " ") for cell in cells]
            self._row_context = None
            if header and len(header) == len(texts):
                pairs = (f"{name}: {value}" if name else value for name, value in zip(header, texts) if value)
                line = "; ".join(pairs)
            else:
                line = " | ".join(text for text in texts if text)
            if line:
                lines.append(line)
        if lines:
            self.blocks.append((0, "\n".join(lines)))


def build_sections(blocks: list[tuple[int, str]], title: str) -> list[Section]:
    """Gliedert Blöcke entlang der Überschriften in Abschnitte.

    Args:
        blocks: Blöcke als ``(Ebene, Text)``.
        title: Titel des Dokuments. Überschriften, die bereits im Titel enthalten sind, entfallen im Pfad.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    texts: list[str] = []

    def close() -> None:
        if texts:
            headings = [text for _, text in stack if text not in title]
            sections.append(Section(headings=headings, text="\n\n".join(texts)))

    for level, text in blocks:
        if level == 0:
            texts.append(text)
            continue
        close()
        texts = []
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
    close()
    return sections


class Preprocessor:
    """Bereitet einen Abruf auf."""

    def __init__(self, config: PreprocessingConfig, crawler_config: CrawlerConfig, university_name: str) -> None:
        """
        Args:
            config: Einstellungen der Aufbereitung.
            crawler_config: Einstellungen des Abrufs, aus denen der Inhaltsbereich stammt.
            university_name: Name der Hochschule, der aus Seitentiteln entfernt wird.
        """
        self.config = config
        self.crawler_config = crawler_config
        self._title_suffix = re.compile(rf"\s*[:|–-]\s*{re.escape(university_name)}\s*$")
        # Beschriftungen und Fassungen verlinkter Ziele aus allen bisher verarbeiteten Seiten
        self._link_labels: dict[str, str] = {}
        self._link_versions: dict[str, str] = {}
        self.stats = PrepareStats()

    def run(self, raw_dir: Path) -> list[Document]:
        """Bereitet alle Seiten und PDF-Dokumente eines Abrufs auf.

        Args:
            raw_dir: Ordner des Abrufs mit ``manifest.jsonl``.

        Returns:
            Die aufbereiteten Dokumente in der Reihenfolge des Abrufs.
        """
        records = [
            json.loads(line)
            for line in (raw_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Titel und Pfad bereits verarbeiteter Seiten, damit Seiten ohne eigene Breadcrumb
        # und PDF-Dokumente den Pfad der Seite übernehmen können, die auf sie verweist.
        context: dict[str, tuple[str, list[str]]] = {}
        parsed: list[tuple[Document, list[tuple[int, str]]]] = []
        # Seiten, die in die Wissensbasis eingehen, unter ihrer ursprünglichen und ihrer
        # endgültigen Adresse (nach Weiterleitung)
        self._known_pages = frozenset(
            normalize_url(url)
            for record in records
            if record["kind"] == "page" and record.get("file") and not self._is_excluded(record)
            for url in (record["url"], record["final_url"])
        )

        for record in records:
            if self._is_excluded(record):
                self.stats.excluded += 1
                continue
            if record["kind"] == "page":
                result = self._parse_page(raw_dir / record["file"], record, context)
            elif record["kind"] == "document":
                result = self._parse_pdf(raw_dir / record["file"], record, context)
            else:
                continue
            if result is not None:
                parsed.append(result)
                document = result[0]
                context[record["final_url"]] = (document.title, document.breadcrumb)

        self._remove_boilerplate(parsed)
        return self._finish(parsed)

    def _is_excluded(self, record: dict) -> bool:
        """Ist ein Eintrag ausdrücklich ausgeschlossen, liegt eine Seite in einem ausgeschlossenen
        Pfad oder wurde ein Dokument nur über eine solche gefunden?

        So lassen sich Pfade nachträglich ausschließen, ohne den gespeicherten Abruf zu verändern.
        """
        if record["url"] in self.config.excluded_documents or record["final_url"] in self.config.excluded_documents:
            return True
        prefixes = self.crawler_config.excluded_path_prefixes
        url = record["final_url"] if record["kind"] == "page" else (record.get("referrer") or "")
        return any(urlsplit(url).path.startswith(prefix) for prefix in prefixes)

    # Seiten

    def _parse_page(
        self, path: Path, record: dict, context: dict[str, tuple[str, list[str]]]
    ) -> tuple[Document, list[tuple[int, str]]] | None:
        """Liest eine Seite und zerlegt ihren Inhaltsbereich in Blöcke."""
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        title = self._page_title(soup)
        breadcrumb = self._breadcrumb(soup) or self._inherited_path(record, context)

        area = next(
            (found for selector in self.crawler_config.content_selectors if (found := soup.select_one(selector))),
            soup.body or soup,
        )
        if title in self.config.generic_titles:
            # Nichtssagende Titel wie "Details" durch die ersten Überschriften des Inhalts ersetzen
            headings = [" ".join(h.get_text(" ").split()) for h in area.find_all(["h1", "h2"])[:2]]
            specific = " – ".join(heading for heading in headings if heading)
            if specific:
                breadcrumb = [specific if entry == title else entry for entry in breadcrumb]
                title = specific
        for selector in self.config.removed_selectors:
            for element in area.select(selector):
                element.decompose()

        base_url = record["final_url"]
        if soup.base is not None and soup.base.get("href"):
            base_url = urljoin(base_url, soup.base["href"])
        converter = HtmlConverter(base_url, self._known_pages)
        blocks = converter.convert(area)
        # Adressen wie beim Abruf vereinheitlichen, damit sie den PDF-Einträgen im Verzeichnis entsprechen
        for url, label in converter.link_labels.items():
            self._link_labels.setdefault(normalize_url(url), label)
        for url, version in converter.link_versions.items():
            self._link_versions.setdefault(normalize_url(url), version)
        self.stats.pages += 1
        document = Document(
            url=record["final_url"],
            title=title,
            breadcrumb=breadcrumb,
            kind="page",
            retrieved_at=record["retrieved_at"],
        )
        return document, blocks

    def _page_title(self, soup: BeautifulSoup) -> str:
        """Seitentitel ohne den angehängten Namen der Hochschule, ersatzweise die erste Überschrift."""
        if soup.title and soup.title.get_text(strip=True):
            return self._title_suffix.sub("", soup.title.get_text(" ", strip=True))
        heading = soup.find("h1")
        return " ".join(heading.get_text(" ").split()) if heading else ""

    def _breadcrumb(self, soup: BeautifulSoup) -> list[str]:
        """Liest die Einträge der Breadcrumb ohne den Eintrag der Startseite."""
        for selector in self.config.breadcrumb_selectors:
            path = [" ".join(item.get_text(" ").split()) for item in soup.select(selector)]
            path = [entry for entry in path if entry and entry != self.config.breadcrumb_root]
            if path:
                return path
        return []

    @staticmethod
    def _inherited_path(record: dict, context: dict[str, tuple[str, list[str]]]) -> list[str]:
        """Pfad der verweisenden Seite einschließlich ihres Titels, ersatzweise aus der Adresse abgeleitet."""
        referrer = record.get("referrer")
        if referrer in context:
            title, path = context[referrer]
            return [*path, title] if title and (not path or path[-1] != title) else list(path)
        segments = [unquote(segment) for segment in urlsplit(record["final_url"]).path.split("/") if segment]
        return [PurePosixPath(segment).stem.replace("-", " ").capitalize() for segment in segments[:-1]]

    # PDF-Dokumente

    def _parse_pdf(
        self, path: Path, record: dict, context: dict[str, tuple[str, list[str]]]
    ) -> tuple[Document, list[tuple[int, str]]] | None:
        """Liest den Text eines PDF-Dokuments. Dokumente ohne lesbaren Text entfallen."""
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                reader.decrypt("")
            page_texts = [clean_pdf_text(page.extract_text() or "") for page in reader.pages]
            metadata_title = (reader.metadata.title if reader.metadata else None) or ""
        except Exception as error:  # pypdf wirft bei beschädigten Dateien unterschiedliche Fehler
            logger.warning("PDF nicht lesbar: %s (%s)", record["final_url"], error)
            self.stats.unreadable_documents += 1
            return None

        # Seiten zusammenfügen und festhalten, wo jede Seite beginnt, damit Quellen auf die
        # passende Seite verweisen können. Leere Seiten entfallen, ihre Nummer wird übersprungen.
        parts: list[str] = []
        page_starts: list[tuple[int, int]] = []
        position = 0
        for number, page_text in enumerate(page_texts, start=1):
            if not page_text:
                continue
            if parts:
                position += 2  # Länge des Trennzeichens "\n\n"
            page_starts.append((position, number))
            parts.append(page_text)
            position += len(page_text)
        text = "\n\n".join(parts)
        if len(text) < self.config.min_pdf_chars_per_page * max(len(page_texts), 1):
            self.stats.unreadable_documents += 1
            return None

        self.stats.documents += 1
        document = Document(
            url=record["final_url"],
            title=self._pdf_title(record, metadata_title, context),
            breadcrumb=self._inherited_path(record, context),
            kind="document",
            retrieved_at=record["retrieved_at"],
            version=self._link_versions.get(record["url"]) or self._link_versions.get(record["final_url"]),
            page_starts=page_starts,
        )
        # PDF-Dokumente haben meist keine maschinenlesbare Gliederung und bilden einen
        # einzigen Abschnitt, der beim Chunking in Teile fester Länge zerlegt wird.
        return document, [(0, text)]

    def _pdf_title(self, record: dict, metadata_title: str, context: dict[str, tuple[str, list[str]]]) -> str:
        """Titel eines PDF-Dokuments.

        Vorrang hat die Beschriftung des Links auf der verweisenden Seite, da sie etwa
        Fassungen einer Ordnung unterscheidet („Medieninformatik - Bachelor:
        Studiengangsordnung/WS20“). Wiederholt der Linktext nur die Überschrift, etwa
        „Studienverlaufsplan: Studienverlaufsplan (PDF)“, tritt der Titel der verweisenden
        Seite an die Stelle der Überschrift, damit der Studiengang erkennbar ist. Danach
        folgen der Titel aus den Metadaten, sofern aussagekräftig, und zuletzt der Dateiname.
        """
        for url in (record["url"], record["final_url"]):
            if url in self._link_labels:
                label = self._link_labels[url]
                heading, _, text = label.partition(": ")
                referrer_title = context.get(record.get("referrer"), ("", []))[0]
                if text and referrer_title and heading.lower() in text.lower():
                    return f"{referrer_title}: {text}"
                return label
        title = " ".join(metadata_title.split())
        url = record["final_url"]
        generic = title.lower().startswith(("microsoft word", "untitled", "unbenannt", "dokument"))
        if len(title) >= 4 and not generic and not title.lower().endswith((".doc", ".docx", ".pdf")):
            return title
        stem = PurePosixPath(unquote(urlsplit(url).path)).stem
        return " ".join(stem.replace("_", " ").replace("-", " ").split())

    # Abschluss

    def _remove_boilerplate(self, parsed: list[tuple[Document, list[tuple[int, str]]]]) -> None:
        """Behält Textblöcke, die auf vielen Seiten gleich sind, nur auf der ersten Seite.

        Mit ``boilerplate_min_pages = 0`` ist die Regel abgeschaltet.
        """
        if self.config.boilerplate_min_pages <= 0:
            return
        counts: dict[str, int] = {}
        for document, blocks in parsed:
            if document.kind != "page":
                continue
            for text in {text for level, text in blocks if level == 0}:
                counts[text] = counts.get(text, 0) + 1

        boilerplate = {
            text
            for text, count in counts.items()
            if count >= self.config.boilerplate_min_pages and len(text) >= self.config.boilerplate_min_chars
        }
        self.stats.boilerplate_blocks = len(boilerplate)
        kept: set[str] = set()
        for document, blocks in parsed:
            if document.kind != "page":
                continue
            filtered = []
            for level, text in blocks:
                if level == 0 and text in boilerplate:
                    if text in kept:
                        self.stats.boilerplate_removed += 1
                        continue
                    kept.add(text)
                filtered.append((level, text))
            blocks[:] = filtered

    def _mark_versions_by_year(self, documents: list[Document]) -> None:
        """Kennzeichnet PDF-Dokumente, die sich nur in der Jahreszahl im Titel unterscheiden, als Fassungen.

        Betrifft etwa Modulhandbücher („Modulhandbuch Bachelor Informatik 2014“, „… 2016“),
        die nicht in den Fassungstabellen der Ordnungen stehen. Das Dokument mit dem
        jüngsten Jahr gilt als aktuell.
        """
        groups: dict[str, list[tuple[int, Document]]] = {}
        for document in documents:
            if document.kind != "document" or document.version is not None:
                continue
            years = [int(year) for year in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", document.title)]
            if not years:
                continue
            key = re.sub(r"(?<!\d)(?:19|20)\d{2}(?:/\d{2,4})?(?!\d)|\b(?:ws|ss|wise|sose)\b", " ", document.title.lower())
            key = " ".join(re.sub(r"[^a-zäöüß]+", " ", key).split())
            groups.setdefault(key, []).append((max(years), document))
        for members in groups.values():
            if len({year for year, _ in members}) < 2:
                continue
            newest = max(year for year, _ in members)
            for year, document in members:
                document.version = "current" if year == newest else "older"
                document.title += " (aktuelle Fassung)" if year == newest else " (ältere Fassung)"
                self.stats.versioned_by_year += 1

    def _finish(self, parsed: list[tuple[Document, list[tuple[int, str]]]]) -> list[Document]:
        """Gliedert die Dokumente in Abschnitte, ergänzt abgeleitete Dokumente und verwirft leere und doppelte."""
        # Erst hier importiert, da das Modul selbst die Datenklassen dieses Moduls verwendet
        from chatbot.derived import build_semester_overviews

        for document, blocks in parsed:
            document.sections = build_sections(blocks, document.title)
        # Die Übersichten nutzen alle Steckbriefe einschließlich der Duplikate, damit für jeden
        # Studiengang die Adresse seines Modulhandbuchs bekannt ist.
        overviews = build_semester_overviews([document for document, _ in parsed], self.config.semester_overviews)
        self.stats.overviews = len(overviews)
        self._mark_versions_by_year([document for document, _ in parsed])

        documents = []
        seen: set[str] = set()
        for document, _ in parsed:
            if not document.sections:
                self.stats.empty += 1
                continue
            # Dieselben Inhalte können unter mehreren Adressen liegen, etwa auf alter und neuer Webseite.
            digest = hashlib.sha1("\n".join(s.text for s in document.sections).encode("utf-8")).hexdigest()
            if digest in seen:
                self.stats.duplicates += 1
                continue
            seen.add(digest)
            documents.append(document)
        return documents + overviews
