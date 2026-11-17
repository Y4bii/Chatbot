"""Abruf der öffentlichen Seiten und Dokumente der Hochschulwebseite.

Ausgehend von der Startadresse werden alle Seiten im Kernbereich (z. B. ``/studium/``)
vollständig abgerufen. Seiten außerhalb des Kernbereichs werden nur bis zu einer
festgelegten Zahl an Klicks verfolgt. PDF-Dokumente werden von allen abgerufenen
Seiten übernommen, sofern sie auf einer Domain der Hochschule liegen. Links werden
nur aus dem Inhaltsbereich einer Seite verfolgt, nicht aus Menü, Kopf- oder Fußzeile.

Jeder Abruf wird mit Datum unter ``data/raw/<datum>/`` abgelegt und anschließend nicht
mehr verändert, damit sich die Ergebnisse der Evaluation vergleichen lassen. Ein
unterbrochener Abruf lässt sich fortsetzen: Aus den bereits gespeicherten Seiten wird
die Warteschlange wiederhergestellt, sodass keine Seite erneut abgerufen wird.
"""

import hashlib
import heapq
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from chatbot.config import CrawlerConfig
from chatbot.robots import RobotsRules

logger = logging.getLogger(__name__)

HTML_TYPES = ("text/html", "application/xhtml+xml")
PDF_TYPE = "application/pdf"

# Dateien, deren Links ohne Abruf übergangen werden
IGNORED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".mp3", ".mp4", ".webm", ".zip", ".ics", ".vcf",
}
# Office-Dokumente werden nicht aufgenommen, aber gezählt, um ihren Anteil abschätzen zu können
OFFICE_EXTENSIONS = {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".rtf"}


@dataclass
class CrawlRecord:
    """Ein Eintrag im Verzeichnis eines Abrufs (``manifest.jsonl``)."""

    url: str
    final_url: str
    kind: str
    status: int
    depth: int
    referrer: str | None
    retrieved_at: str
    file: str | None = None
    title: str | None = None
    content_area: str | None = None
    note: str | None = None


@dataclass
class CrawlStats:
    """Zusammenfassung eines Abrufs."""

    requests: int = 0
    pages: int = 0
    documents: int = 0
    errors: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    office_links: set[str] = field(default_factory=set)

    def skip(self, reason: str) -> None:
        """Zählt einen übersprungenen Abruf mit Begründung."""
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def normalize_url(url: str) -> str | None:
    """Vereinheitlicht eine Adresse.

    Fragmente (``#...``) werden entfernt und der Host kleingeschrieben.

    Returns:
        Die Adresse oder ``None``, wenn es keine HTTP(S)-Adresse ist.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    # Die Seiten der Hochschule sind per HTTPS erreichbar. Einheitlich HTTPS zu
    # verwenden verhindert doppelte Einträge für dieselbe Seite.
    return urlunsplit(("https", parts.netloc.lower(), parts.path or "/", parts.query, ""))


def _extension(url: str) -> str:
    """Dateiendung des Pfads in Kleinbuchstaben, z. B. ``.pdf``."""
    return PurePosixPath(urlsplit(url).path).suffix.lower()


class Crawler:
    """Ruft die Webseite ab und speichert Seiten und PDF-Dokumente."""

    def __init__(self, config: CrawlerConfig, output_dir: Path) -> None:
        """
        Args:
            config: Einstellungen für den Abruf.
            output_dir: Zielordner des Abrufs. Er darf noch keinen Abruf enthalten.
        """
        self.config = config
        self.output_dir = output_dir
        self.stats = CrawlStats()
        self._client = httpx.Client(
            headers={"User-Agent": config.user_agent},
            timeout=config.timeout,
            follow_redirects=True,
        )
        self._robots: dict[str, RobotsRules] = {}
        self._last_request = 0.0
        # Warteschlange nach Linktiefe geordnet, damit Seiten über den kürzesten Weg
        # erreicht werden. Der Zähler hält die Reihenfolge bei gleicher Tiefe stabil.
        self._queue: list[tuple[int, int, str, str | None]] = []
        self._counter = 0
        self._depths: dict[str, int] = {}
        self._done: set[str] = set()
        self._restored_requests = 0
        # Zeitraum eines fortgesetzten Abrufs, falls dieser ohne crawl_info.json abgebrochen wurde
        self._restored_period: tuple[str, str] | None = None

    def run(
        self, start_url: str | None = None, max_requests: int | None = None, resume: bool = False
    ) -> CrawlStats:
        """Führt den Abruf durch oder setzt einen unterbrochenen Abruf fort.

        Args:
            start_url: Abweichende Startadresse, z. B. für einen Testlauf.
            max_requests: Abweichende Obergrenze der Abrufe, z. B. für einen Testlauf.
            resume: Einen vorhandenen Abruf im Zielordner fortsetzen.

        Returns:
            Die Zusammenfassung des Abrufs. Bei einer Fortsetzung zählen die bisherigen
            Abrufe mit, auch für die Obergrenze.
        """
        manifest_path = self.output_dir / "manifest.jsonl"
        if manifest_path.exists() and not resume:
            raise FileExistsError(f"Im Zielordner liegt bereits ein Abruf: {self.output_dir}")
        if resume and not manifest_path.exists():
            raise FileNotFoundError(f"Kein Abruf zum Fortsetzen gefunden: {self.output_dir}")
        (self.output_dir / "pages").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "documents").mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(timezone.utc)
        limit = max_requests or self.config.max_requests
        start = normalize_url(start_url or self.config.start_url)
        if resume:
            self._restore(manifest_path)
        else:
            self._enqueue(start, 0 if self._is_core(start) else 1, None)

        with manifest_path.open("a" if resume else "w", encoding="utf-8") as manifest:
            while self._queue and self.stats.requests < limit:
                depth, _, url, referrer = heapq.heappop(self._queue)
                # Einträge, die später über einen kürzeren Weg erneut eingereiht wurden, überspringen
                if url in self._done or depth > self._depths[url]:
                    continue
                self._done.add(url)
                if not self._robots_allow(url):
                    self.stats.skip("durch robots.txt ausgeschlossen")
                    continue
                record = self._fetch(url, depth, referrer)
                manifest.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                manifest.flush()

        self._client.close()
        self._write_info(start, started_at, limit)
        return self.stats

    def _restore(self, manifest_path: Path) -> None:
        """Stellt den Stand eines unterbrochenen Abrufs aus seinem Verzeichnis wieder her.

        Bereits abgerufene Adressen werden als erledigt markiert. Die Warteschlange
        entsteht neu aus den Links der gespeicherten Seiten, nach denselben Regeln wie
        beim ursprünglichen Abruf.
        """
        records = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Eine beim Abbruch unvollständig geschriebene letzte Zeile wird übergangen.
                logger.warning("Unvollständiger Eintrag im Verzeichnis übersprungen")

        for record in records:
            self._done.update((record["url"], record["final_url"]))
            self._depths.setdefault(record["url"], record["depth"])
            self.stats.requests += 1
            if record["kind"] == "page":
                self.stats.pages += 1
            elif record["kind"] == "document":
                self.stats.documents += 1
            elif record["kind"] == "error":
                self.stats.errors += 1
            elif record.get("note"):
                self.stats.skip(record["note"])
        self._restored_requests = self.stats.requests
        if records:
            self._restored_period = (records[0]["retrieved_at"], records[-1]["retrieved_at"])

        for record in records:
            # Seiten, die inzwischen ausgeschlossen sind, liefern keine Links mehr.
            if record["kind"] != "page" or not self._is_site_page(record["final_url"]):
                continue
            soup = BeautifulSoup((self.output_dir / record["file"]).read_bytes(), "html.parser")
            depth = 0 if self._is_core(record["final_url"]) else max(record["depth"], 1)
            _, links = self._content_links(soup, record["final_url"])
            for link in links:
                self._handle_link(link, depth, record["final_url"])
        logger.info(
            "Abruf fortgesetzt: %d bisherige Abrufe, %d Adressen in der Warteschlange",
            self._restored_requests,
            len(self._queue),
        )

    # Bereich des Abrufs

    def _is_site_page(self, url: str) -> bool:
        """Liegt die Adresse auf einer der Webseiten und nicht in einem ausgeschlossenen Pfad?"""
        parts = urlsplit(url)
        return parts.netloc in self.config.site_hosts and not any(
            parts.path.startswith(prefix) for prefix in self.config.excluded_path_prefixes
        )

    def _is_core(self, url: str) -> bool:
        """Liegt die Adresse im Kernbereich, der vollständig abgerufen wird?"""
        parts = urlsplit(url)
        return parts.netloc in self.config.site_hosts and parts.path.startswith(self.config.core_path_prefix)

    def _is_document_host(self, url: str) -> bool:
        """Liegt das Dokument auf der Domain der Hochschule oder einer ihrer Subdomains?"""
        host = urlsplit(url).netloc
        domain = self.config.document_domain
        return host == domain or host.endswith("." + domain)

    def _enqueue(self, url: str, depth: int, referrer: str | None) -> None:
        """Reiht eine Adresse ein, sofern sie nicht schon über einen kürzeren Weg bekannt ist."""
        if url in self._done or depth >= self._depths.get(url, depth + 1):
            return
        self._depths[url] = depth
        self._counter += 1
        heapq.heappush(self._queue, (depth, self._counter, url, referrer))

    def _handle_link(self, link: str, depth: int, referrer: str) -> None:
        """Entscheidet, ob ein Link einer abgerufenen Seite weiterverfolgt wird.

        Args:
            link: Normalisierte Adresse des Links.
            depth: Linktiefe der Seite, auf der der Link steht.
            referrer: Adresse dieser Seite.
        """
        extension = _extension(link)
        if extension == ".pdf":
            if self._is_document_host(link):
                self._enqueue(link, depth, referrer)
            return
        if extension in OFFICE_EXTENSIONS:
            if self._is_document_host(link):
                self.stats.office_links.add(link)
            return
        if extension in IGNORED_EXTENSIONS or not self._is_site_page(link):
            return
        # Seiten mit Query-Parametern sind meist Filter- oder Kalenderansichten mit
        # unbegrenzt vielen Varianten und werden daher nicht verfolgt.
        if urlsplit(link).query:
            return
        link_depth = 0 if self._is_core(link) else depth + 1
        if link_depth <= self.config.max_outside_depth:
            self._enqueue(link, link_depth, referrer)

    # Abruf

    def _wait(self) -> None:
        """Hält den Mindestabstand zwischen zwei Abrufen ein."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.config.request_delay:
            time.sleep(self.config.request_delay - elapsed)
        self._last_request = time.monotonic()

    def _robots_allow(self, url: str) -> bool:
        """Prüft die ``robots.txt`` des Hosts, die beim ersten Zugriff abgerufen wird."""
        parts = urlsplit(url)
        rules = self._robots.get(parts.netloc)
        if rules is None:
            rules = self._load_robots(parts.netloc)
            self._robots[parts.netloc] = rules
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        return rules.allows(path)

    def _load_robots(self, host: str) -> RobotsRules:
        """Ruft die ``robots.txt`` eines Hosts ab.

        Nach RFC 9309 gilt: Fehlt die Datei (4xx), ist alles erlaubt. Ist sie wegen
        eines Server- oder Verbindungsfehlers nicht erreichbar, ist nichts erlaubt.
        """
        self._wait()
        try:
            response = self._client.get(f"https://{host}/robots.txt")
        except httpx.HTTPError as error:
            logger.warning("robots.txt von %s nicht erreichbar: %s", host, error)
            return RobotsRules.disallow_all()
        if response.status_code == 200:
            return RobotsRules(response.text, self.config.user_agent)
        if 400 <= response.status_code < 500:
            return RobotsRules("", self.config.user_agent)
        logger.warning("robots.txt von %s liefert HTTP %d", host, response.status_code)
        return RobotsRules.disallow_all()

    def _download(self, url: str) -> tuple[httpx.Response, bytes | None]:
        """Lädt eine Adresse herunter.

        Returns:
            Die Antwort und ihren Inhalt. Der Inhalt ist ``None``, wenn die Datei die
            zulässige Größe überschreitet.
        """
        self._wait()
        limit = int(self.config.max_document_mb * 1024 * 1024)
        with self._client.stream("GET", url) as response:
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > limit:
                    return response, None
                chunks.append(chunk)
        return response, b"".join(chunks)

    def _fetch(self, url: str, depth: int, referrer: str | None) -> CrawlRecord:
        """Ruft eine Adresse ab, speichert den Inhalt und reiht gefundene Links ein."""
        self.stats.requests += 1
        logger.info("[%d] Tiefe %d: %s", self.stats.requests, depth, url)
        record = CrawlRecord(
            url=url,
            final_url=url,
            kind="skipped",
            status=0,
            depth=depth,
            referrer=referrer,
            retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        try:
            response, body = self._download(url)
        except httpx.HTTPError as error:
            self.stats.errors += 1
            record.kind, record.note = "error", str(error)
            logger.warning("Fehler bei %s: %s", url, error)
            return record

        record.status = response.status_code
        record.final_url = normalize_url(str(response.url)) or url
        if record.final_url != url:
            # Nach einer Weiterleitung kann das Ziel bereits abgerufen worden sein.
            if record.final_url in self._done:
                record.note = "Weiterleitung auf bereits abgerufene Adresse"
                self.stats.skip(record.note)
                return record
            self._done.add(record.final_url)

        if response.status_code != 200:
            self.stats.errors += 1
            record.kind, record.note = "error", f"HTTP {response.status_code}"
            return record
        if body is None:
            record.note = f"größer als {self.config.max_document_mb} MB"
            self.stats.skip(record.note)
            return record

        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type == PDF_TYPE:
            return self._save_document(record, body)
        if content_type in HTML_TYPES:
            return self._save_page(record, body)
        record.note = f"Dateityp {content_type or 'unbekannt'}"
        self.stats.skip(record.note)
        return record

    def _save_document(self, record: CrawlRecord, body: bytes) -> CrawlRecord:
        """Speichert ein PDF-Dokument, sofern es von der Hochschule stammt."""
        if not self._is_document_host(record.final_url):
            record.note = "Dokument außerhalb der Hochschule"
            self.stats.skip(record.note)
            return record
        record.kind = "document"
        record.file = self._write("documents", record.final_url, ".pdf", body)
        self.stats.documents += 1
        return record

    def _save_page(self, record: CrawlRecord, body: bytes) -> CrawlRecord:
        """Speichert eine Seite und reiht die Links aus ihrem Inhaltsbereich ein."""
        if not self._is_site_page(record.final_url):
            record.note = "Seite außerhalb der Webseite"
            self.stats.skip(record.note)
            return record

        soup = BeautifulSoup(body, "html.parser")
        language = (soup.html.get("lang", "") if soup.html else "").lower()
        if language and not language.startswith(self.config.language):
            record.note = f"Sprache {language}"
            self.stats.skip(record.note)
            return record

        record.kind = "page"
        record.title = soup.title.get_text(strip=True) if soup.title else None
        record.file = self._write("pages", record.final_url, ".html", body)
        self.stats.pages += 1

        # Liegt die Seite außerhalb des Kernbereichs, zählt sie selbst als Klick.
        depth = 0 if self._is_core(record.final_url) else max(record.depth, 1)
        record.content_area, links = self._content_links(soup, record.final_url)
        for link in links:
            self._handle_link(link, depth, record.final_url)
        return record

    def _content_links(self, soup: BeautifulSoup, base_url: str) -> tuple[str, list[str]]:
        """Sammelt die Links aus dem Inhaltsbereich einer Seite.

        Returns:
            Den verwendeten Selektor (oder ``"Fallback"``) und die normalisierten Links.
        """
        area, used = None, "Fallback"
        for selector in self.config.content_selectors:
            area = soup.select_one(selector)
            if area is not None:
                used = selector
                break
        if area is None:
            # Ohne erkennbaren Inhaltsbereich gilt die ganze Seite ohne Kopf, Fuß und Navigation.
            area = soup.body or soup
            ignored = ("header", "footer", "nav")
        else:
            ignored = self.config.ignored_selectors
        for selector in ignored:
            for element in area.select(selector):
                element.decompose()

        # Relative Links beziehen sich auf eine im <base>-Tag angegebene Adresse, sofern vorhanden.
        if soup.base is not None and soup.base.get("href"):
            base_url = urljoin(base_url, soup.base["href"])
        links = []
        for anchor in area.find_all("a", href=True):
            link = normalize_url(urljoin(base_url, anchor["href"]))
            if link:
                links.append(link)
        return used, links

    # Ablage

    def _write(self, subdir: str, url: str, suffix: str, body: bytes) -> str:
        """Speichert einen Inhalt unter einem aus der Adresse abgeleiteten Dateinamen."""
        name = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16] + suffix
        (self.output_dir / subdir / name).write_bytes(body)
        return f"{subdir}/{name}"

    def _write_info(self, start_url: str, started_at: datetime, limit: int) -> None:
        """Hält Zeitpunkt, Einstellungen und Ergebnis des Abrufs in ``crawl_info.json`` fest.

        Bei einer Fortsetzung bleiben die Angaben früherer Läufe erhalten.
        """
        info_path = self.output_dir / "crawl_info.json"
        previous = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        runs = previous.get("runs", [])
        if not previous and self._restored_period:
            # Der vorherige Lauf wurde abgebrochen; sein Zeitraum stammt aus dem Verzeichnis.
            runs.append(
                {
                    "started_at": self._restored_period[0],
                    "finished_at": self._restored_period[1],
                    "requests": self._restored_requests,
                    "max_requests": None,
                    "note": "abgebrochen, Angaben aus manifest.jsonl",
                }
            )
            previous = {"started_at": self._restored_period[0]}
        elif previous and not runs:
            # Ein früherer Lauf ohne Liste der Läufe wird als erster Lauf übernommen.
            runs.append(
                {
                    "started_at": previous.get("started_at"),
                    "finished_at": previous.get("finished_at"),
                    "requests": previous.get("stats", {}).get("requests"),
                    "max_requests": previous.get("max_requests"),
                }
            )
        runs.append(
            {
                "started_at": started_at.isoformat(timespec="seconds"),
                "finished_at": finished_at,
                "requests": self.stats.requests - self._restored_requests,
                "max_requests": limit,
            }
        )
        info = {
            "start_url": start_url,
            "started_at": previous.get("started_at", started_at.isoformat(timespec="seconds")),
            "finished_at": finished_at,
            "runs": runs,
            "max_requests": limit,
            "config": asdict(self.config),
            "stats": {
                "requests": self.stats.requests,
                "pages": self.stats.pages,
                "documents": self.stats.documents,
                "errors": self.stats.errors,
                "skipped": self.stats.skipped,
                "office_documents_not_included": sorted(self.stats.office_links),
                "queue_remaining": len(self._queue),
            },
        }
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
