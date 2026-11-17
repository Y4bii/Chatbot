"""Zerlegung der aufbereiteten Dokumente in Chunks.

Seiten werden entlang ihrer Überschriften zerlegt, sodass ein Chunk in der Regel einen
zusammenhängenden Abschnitt enthält. Sehr kurze Abschnitte, etwa Teaser-Kacheln auf
Übersichtsseiten, werden mit benachbarten Abschnitten unter derselben übergeordneten
Überschrift zusammengelegt. Abschnitte, die die maximale Eingabelänge des
Embedding-Modells überschreiten, werden an Absatz- und Satzgrenzen mit Überlappung
weiter unterteilt. PDF-Dokumente werden in Teile fester Länge mit Überlappung zerlegt.

Jedem Chunk werden der Titel des Dokuments und die übergeordneten Überschriften
vorangestellt, damit er ohne seine Umgebung verständlich bleibt.
"""

import bisect
import hashlib
import re
from dataclasses import dataclass

from tokenizers import Tokenizer

from chatbot.config import ChunkingConfig
from chatbot.preprocessing import Document, Section

_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+")
CONTEXT_SEPARATOR = " › "
# Überschrift eines Semesters im Regelstudienplan, z. B. „4. Semester (30KP, 5 Prüfungen)“
_SEMESTER_HEADING = re.compile(r"^\d+\.\s*Semester\b")


@dataclass
class Chunk:
    """Ein Chunk mit den Metadaten seiner Quelle."""

    id: str
    text: str
    url: str
    title: str
    breadcrumb: list[str]
    headings: list[str]
    kind: str
    token_count: int
    # Fassung einer Regelung ("current", "older") oder None
    version: str | None = None
    # Bei PDF-Dokumenten die Seite, auf der der Chunk beginnt
    page: int | None = None


class Chunker:
    """Zerlegt Dokumente in Chunks mit begrenzter Tokenzahl."""

    def __init__(self, config: ChunkingConfig) -> None:
        """
        Args:
            config: Einstellungen für die Zerlegung. Der Tokenizer wird beim ersten
                Aufruf von Hugging Face geladen und lokal zwischengespeichert.
        """
        self.config = config
        self._tokenizer = Tokenizer.from_pretrained(config.tokenizer)
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count_tokens(self, text: str) -> int:
        """Anzahl der Token ohne Sondertoken."""
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def chunk(self, document: Document) -> list[Chunk]:
        """Zerlegt ein Dokument in Chunks."""
        chunks = []
        sections = document.sections if document.kind == "document" else self._merge_short(document)
        for section_index, section in enumerate(sections):
            prefix = self._prefix(document, section.headings)
            budget = self._budget(prefix)
            pages: list[int | None]
            if document.kind == "document":
                windows = self._fixed_windows_with_starts(section.text, budget, self.config.overlap_tokens)
                pieces = [window for window, _ in windows]
                pages = [_page_at(document.page_starts, start) for _, start in windows]
            else:
                pieces = self._split(section.text, budget)
                pages = [None] * len(pieces)
            for piece_index, (piece, page) in enumerate(zip(pieces, pages)):
                text = prefix + piece
                # Der Titel geht mit ein, da abgeleitete Dokumente dieselbe Adresse teilen können.
                key = f"{document.url}#{document.title}#{section_index}.{piece_index}"
                chunk_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=text,
                        url=document.url,
                        title=document.title,
                        breadcrumb=document.breadcrumb,
                        headings=section.headings,
                        kind=document.kind,
                        token_count=self.count_tokens(text),
                        version=document.version,
                        page=page,
                    )
                )
        return chunks

    @staticmethod
    def _prefix(document: Document, headings: list[str]) -> str:
        """Kontext, der jedem Chunk vorangestellt wird: Titel und übergeordnete Überschriften.

        Bei PDF-Dokumenten steht zusätzlich die verweisende Seite davor, sofern sie nicht
        schon im Titel vorkommt. So ist etwa ein Studienverlaufsplan seinem Studiengang
        zugeordnet, auch wenn sein Titel ihn nicht nennt.
        """
        context = [document.title, *headings]
        if document.kind == "document" and document.breadcrumb:
            page = document.breadcrumb[-1]
            if page.lower() not in document.title.lower():
                context.insert(0, page)
        return CONTEXT_SEPARATOR.join(context) + "\n\n"

    def _budget(self, prefix: str) -> int:
        """Verfügbare Token für den Text eines Chunks nach Abzug des Präfixes.

        Beim Zusammensetzen von Präfix und Text kann die Tokenzahl an der Nahtstelle
        leicht von der Summe abweichen, daher bleiben zwei Token Puffer.
        """
        return self.config.max_tokens - self.count_tokens(prefix) - 2

    def _merge_short(self, document: Document) -> list[Section]:
        """Legt kurze Abschnitte mit Nachbarn unter derselben übergeordneten Überschrift zusammen.

        Zwei aufeinanderfolgende Abschnitte werden verbunden, wenn sie dieselbe
        übergeordnete Überschrift haben, mindestens einer kürzer als
        ``min_section_tokens`` ist und das Ergebnis die maximale Länge nicht
        überschreitet. Die eigene Überschrift eines Abschnitts wird dabei in den Text
        übernommen, damit die Gliederung erhalten bleibt.

        Semester eines Regelstudienplans werden nie zusammengelegt. Sonst stand etwa das
        4. Semester im Chunk mit der Überschrift des 3. Semesters, und die Suche nach dem
        4. Semester fand ihn nicht.
        """
        merged: list[Section] = []
        # Gibt je Eintrag in ``merged`` an, ob er bereits zusammengelegt wurde und seine
        # Überschrift damit schon im Text steht.
        combined: list[bool] = []
        for section in document.sections:
            if merged and section.headings and merged[-1].headings and not (
                self._is_semester(section) or self._is_semester(merged[-1])
            ):
                previous = merged[-1]
                parent = previous.headings if combined[-1] else previous.headings[:-1]
                if section.headings[:-1] == parent:
                    short = min(self.count_tokens(previous.text), self.count_tokens(section.text))
                    previous_text = previous.text if combined[-1] else f"{previous.headings[-1]}\n{previous.text}"
                    text = f"{previous_text}\n\n{section.headings[-1]}\n{section.text}"
                    prefix = self._prefix(document, parent)
                    if short < self.config.min_section_tokens and self.count_tokens(text) <= self._budget(prefix):
                        merged[-1] = Section(headings=list(parent), text=text)
                        combined[-1] = True
                        continue
            merged.append(section)
            combined.append(False)
        return merged

    @staticmethod
    def _is_semester(section: Section) -> bool:
        """Ist der Abschnitt ein Semester eines Regelstudienplans?"""
        return bool(section.headings) and bool(_SEMESTER_HEADING.match(section.headings[-1]))

    def _split(self, text: str, budget: int) -> list[str]:
        """Teilt einen Abschnitt an Absatz-, Zeilen- und Satzgrenzen in Teile mit Überlappung."""
        if self.count_tokens(text) <= budget:
            return [text]

        # Jede Einheit führt das Trennzeichen mit, das im Original vor ihr stand, damit
        # Absätze, Zeilen und Sätze beim Zusammensetzen erhalten bleiben.
        units: list[_Unit] = []
        for paragraph_index, paragraph in enumerate(text.split("\n\n")):
            for line_index, line in enumerate(paragraph.split("\n")):
                for unit_index, unit in enumerate(self._units(line, budget)):
                    if unit_index:
                        separator = " "
                    elif line_index:
                        separator = "\n"
                    else:
                        separator = "\n\n" if paragraph_index else ""
                    units.append(_Unit(unit, self.count_tokens(unit), separator))

        pieces: list[str] = []
        current: list[_Unit] = []
        size = 0
        for unit in units:
            if current and size + unit.tokens > budget:
                pieces.append(_join(current))
                current = self._overlap_tail(current)
                # Die Überlappung wird gekürzt, wenn sie zusammen mit der nächsten Einheit nicht passt.
                while current and sum(item.tokens for item in current) + unit.tokens > budget:
                    current.pop(0)
                size = sum(item.tokens for item in current)
            current.append(unit)
            size += unit.tokens
        if current:
            pieces.append(_join(current))
        return [piece for piece in pieces if piece]

    def _units(self, line: str, budget: int) -> list[str]:
        """Zerlegt eine zu lange Zeile in Sätze und, falls nötig, in Teile fester Länge."""
        if self.count_tokens(line) <= budget:
            return [line]
        units = []
        for sentence in _SENTENCE_END.split(line):
            if self.count_tokens(sentence) <= budget:
                units.append(sentence)
            else:
                units.extend(self._fixed_windows(sentence, budget, overlap=0))
        return units

    def _overlap_tail(self, units: list["_Unit"]) -> list["_Unit"]:
        """Letzte Einheiten eines Teils, die als Überlappung in den nächsten Teil übernommen werden."""
        tail: list[_Unit] = []
        size = 0
        for unit in reversed(units):
            if size + unit.tokens > self.config.overlap_tokens:
                break
            tail.insert(0, unit)
            size += unit.tokens
        return tail

    def _fixed_windows(self, text: str, budget: int, overlap: int | None = None) -> list[str]:
        """Teilt einen Text in Teile fester Tokenzahl mit Überlappung."""
        overlap = self.config.overlap_tokens if overlap is None else overlap
        return [window for window, _ in self._fixed_windows_with_starts(text, budget, overlap)]

    def _fixed_windows_with_starts(self, text: str, budget: int, overlap: int) -> list[tuple[str, int]]:
        """Wie :meth:`_fixed_windows`, liefert zu jedem Teil zusätzlich seine Startposition im Text."""
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        if len(offsets) <= budget:
            return [(text.strip(), 0)] if text.strip() else []
        step = max(budget - overlap, 1)
        windows = []
        for start in range(0, len(offsets), step):
            end = min(start + budget, len(offsets))
            window = text[offsets[start][0] : offsets[end - 1][1]].strip()
            if window:
                windows.append((window, offsets[start][0]))
            if end == len(offsets):
                break
        return windows


def _page_at(page_starts: list[tuple[int, int]], position: int) -> int | None:
    """Seitenzahl an einer Zeichenposition, ermittelt aus den Seitenanfängen eines PDF-Dokuments."""
    if not page_starts:
        return None
    index = bisect.bisect_right([start for start, _ in page_starts], position) - 1
    return page_starts[max(index, 0)][1]


@dataclass
class _Unit:
    """Kleinste Einheit beim Teilen eines Abschnitts mit dem vorangehenden Trennzeichen."""

    text: str
    tokens: int
    separator: str


def _join(units: list[_Unit]) -> str:
    """Setzt Einheiten mit ihren Trennzeichen wieder zu Text zusammen."""
    return "".join((unit.separator if index else "") + unit.text for index, unit in enumerate(units)).strip()
