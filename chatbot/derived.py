"""Abgeleitete Dokumente: Semesterübersichten aus den Modul-Steckbriefen.

Fragen wie „Welche Module habe ich im 5. Semester?“ betreffen viele Modul-Steckbriefe
zugleich. Eine Suche, die nur wenige Chunks liefert, kann diese Angaben nicht bündeln,
und die grafischen Studienverlaufspläne verlieren beim Auslesen ihre Tabellenstruktur.
Jeder Steckbrief nennt aber, in welchem Studiengang und Fachsemester das Modul vorgesehen
ist. Daraus entsteht je Studiengang und Studienordnungsjahr ein Dokument mit einem
Abschnitt je Fachsemester. Es enthält nur Angaben aus den Steckbriefen.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from chatbot.config import SemesterOverviewConfig
from chatbot.preprocessing import Document, Section

# "Bachelor Hebammenwissenschaft 2021" -> Studiengang und Jahr
_PROGRAM_YEAR = re.compile(r"^(?P<program>.+?)\s+(?P<year>\d{4})$")


@dataclass
class _Module:
    """Ein Modul aus Sicht eines Studiengangs."""

    name: str
    credits: str
    kind: str
    semesters: str
    numbers: list[int]
    exam: str

    def line(self) -> str:
        """Zeile in der Semesterübersicht."""
        details = [f"{self.credits} KP" if self.credits else "", self.kind]
        if len(self.numbers) > 1:
            details.append(self.semesters)
        text = f"- {self.name}: {', '.join(detail for detail in details if detail)}"
        return f"{text}; Prüfung: {self.exam}" if self.exam else text


def _normalize(text: str) -> str:
    """Vergleichsform eines Studiengangsnamens, z. B. für „Bachelor Studiengang Hebammenwissenschaft“."""
    return re.sub(r"studiengang|[^a-zäöüß]", "", text.lower())


def _first_line(text: str) -> str:
    """Erste Zeile eines Abschnitts ohne Aufzählungszeichen und ohne angehängte Adressen."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"\s*\(https?://[^)]*\)", "", line.lstrip("- ")).strip()


def build_semester_overviews(documents: list[Document], config: SemesterOverviewConfig) -> list[Document]:
    """Leitet aus den Modul-Steckbriefen je Studiengang und Jahr eine Semesterübersicht ab.

    Args:
        documents: Aufbereitete Dokumente einschließlich doppelter Steckbriefe, damit für
            jeden Studiengang die Adresse seines Modulhandbuchs bekannt ist.
        config: Aufbau der Steckbriefe dieser Hochschule.

    Returns:
        Die Semesterübersichten. Gibt es für einen Studiengang mehrere Jahre, ist die
        jüngste als aktuelle, die übrigen als ältere Fassung gekennzeichnet.
    """
    if not config.enabled:
        return []

    modules: dict[tuple[str, str], dict[str, _Module]] = defaultdict(dict)
    handbooks: dict[str, Counter] = defaultdict(Counter)
    breadcrumbs: dict[str, list[str]] = {}
    retrieved_at = ""

    for document in documents:
        if document.kind != "page" or config.module_url_marker not in document.url:
            continue
        retrieved_at = max(retrieved_at, document.retrieved_at)
        by_heading = {section.headings[-1]: section.text for section in document.sections if section.headings}

        # Modulhandbuch des Studiengangs, auf dessen Seiten der Steckbrief liegt
        # z. B. ".../bachelor-studiengang-hebammenwissenschaft" + "/modulhandbuch/"
        handbook_path = config.module_url_marker.rstrip("/").rsplit("/", 1)[0] + "/"
        handbook_url = document.url[: document.url.index(config.module_url_marker)] + handbook_path
        if config.handbook_breadcrumb in document.breadcrumb:
            position = document.breadcrumb.index(config.handbook_breadcrumb)
            if position > 0:
                key = _normalize(document.breadcrumb[position - 1])
                handbooks[key][handbook_url] += 1
                breadcrumbs.setdefault(key, document.breadcrumb[: position + 1])

        name = re.sub(r"^Modul\s+", "", document.title)
        credits = _first_line(by_heading.get(config.credits_heading, ""))
        exam = _first_line(by_heading.get(config.exam_heading, ""))[:150]
        for raw_line in by_heading.get(config.semester_heading, "").splitlines():
            parts = [part.strip() for part in raw_line.lstrip("- ").split(",")]
            if len(parts) < 3:
                continue
            match = _PROGRAM_YEAR.match(parts[0])
            program, year = (match.group("program"), match.group("year")) if match else (parts[0], "")
            semesters = parts[-1]
            numbers = sorted({int(number) for number in re.findall(r"(\d+)\.", semesters)})
            modules[(program, year)].setdefault(
                name, _Module(name, credits, parts[1], semesters, numbers, exam)
            )

    # Jüngstes Jahr je Studiengang als aktuelle Fassung
    newest: dict[str, str] = {}
    for program, year in modules:
        newest[program] = max(newest.get(program, ""), year)
    years_per_program = Counter(program for program, _ in modules)

    overviews = []
    for (program, year), entries in sorted(modules.items()):
        by_semester: dict[int, list[_Module]] = defaultdict(list)
        anytime: list[_Module] = []
        for module in entries.values():
            for number in module.numbers or []:
                by_semester[number].append(module)
            if not module.numbers:
                anytime.append(module)

        label = f"{program} {year}".strip()
        order = lambda module: (module.kind != "Pflicht", module.name)  # noqa: E731
        sections = [
            Section(
                headings=[f"{number}. Fachsemester"],
                text=f"Module im {number}. Fachsemester ({label}):\n"
                + "\n".join(module.line() for module in sorted(by_semester[number], key=order)),
            )
            for number in sorted(by_semester)
        ]
        if anytime:
            sections.append(
                Section(
                    headings=["Module ohne festes Fachsemester"],
                    text=f"Module ohne festes Fachsemester, z. B. Wahlpflicht- und Wahlmodule ({label}):\n"
                    + "\n".join(module.line() for module in sorted(anytime, key=order)),
                )
            )

        key = _normalize(program)
        url = handbooks[key].most_common(1)[0][0] if handbooks.get(key) else ""
        if not url:
            continue  # ohne zugehöriges Modulhandbuch fehlt eine überprüfbare Quelle
        version = None
        title = f"Studienverlauf: {label}"
        if years_per_program[program] > 1:
            version = "current" if year == newest[program] else "older"
            title += " (aktuelle Fassung)" if version == "current" else " (ältere Fassung)"
        overviews.append(
            Document(
                url=url,
                title=title,
                breadcrumb=breadcrumbs.get(key, []),
                kind="overview",
                retrieved_at=retrieved_at,
                sections=sections,
                version=version,
            )
        )
    return overviews
