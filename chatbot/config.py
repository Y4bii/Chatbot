"""Laden der hochschulspezifischen Konfiguration.

Alle Angaben, die sich auf eine bestimmte Hochschule beziehen, stehen in einer
TOML-Datei unter ``config/``. Dieses Modul liest sie ein und stellt sie den übrigen
Komponenten als unveränderliche Objekte bereit.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "uzl.toml"


@dataclass(frozen=True)
class CrawlerConfig:
    """Einstellungen für den Abruf der Webseite."""

    start_url: str
    site_hosts: tuple[str, ...]
    core_path_prefix: str
    max_outside_depth: int
    document_domain: str
    excluded_path_prefixes: tuple[str, ...]
    language: str
    content_selectors: tuple[str, ...]
    ignored_selectors: tuple[str, ...]
    request_delay: float
    timeout: float
    max_requests: int
    max_document_mb: float
    user_agent: str


@dataclass(frozen=True)
class SemesterOverviewConfig:
    """Aufbau der Modul-Steckbriefe, aus denen Semesterübersichten abgeleitet werden."""

    enabled: bool
    module_url_marker: str
    handbook_breadcrumb: str
    semester_heading: str
    credits_heading: str
    exam_heading: str


@dataclass(frozen=True)
class PreprocessingConfig:
    """Einstellungen für die Aufbereitung der abgerufenen Seiten und Dokumente."""

    breadcrumb_selectors: tuple[str, ...]
    breadcrumb_root: str
    generic_titles: tuple[str, ...]
    removed_selectors: tuple[str, ...]
    boilerplate_min_pages: int
    boilerplate_min_chars: int
    min_pdf_chars_per_page: int
    excluded_documents: tuple[str, ...]
    semester_overviews: SemesterOverviewConfig


@dataclass(frozen=True)
class ChunkingConfig:
    """Einstellungen für die Zerlegung in Chunks."""

    tokenizer: str
    max_tokens: int
    overlap_tokens: int
    min_section_tokens: int


@dataclass(frozen=True)
class StudentServiceConfig:
    """Kontaktdaten des Studierendenservice für Fallback und Übergabe."""

    name: str
    email: str
    phone: str
    url: str


@dataclass(frozen=True)
class SmallTalkConfig:
    """Muster für Floskeln einer Art (z. B. Begrüßung) und die feste Antwort darauf."""

    patterns: tuple[str, ...]
    response: str


@dataclass(frozen=True)
class ProgramConfig:
    """Ein Studiengang mit Erkennungsmustern, Abkürzungen und Abschlüssen."""

    name: str
    patterns: tuple[str, ...]
    degrees: tuple[str, ...]
    abbreviations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClarificationConfig:
    """Muster und Texte für Rückfragen bei Fragen, die vom Studiengang abhängen."""

    patterns: tuple[str, ...]
    degree_pattern: str
    program_response: str
    degree_response: str
    relative_semester_pattern: str
    semester_number_pattern: str
    semester_response: str


@dataclass(frozen=True)
class ChatbotConfig:
    """Name und Verhalten des Chatbots."""

    name: str
    subtitle: str
    fallback_marker: str
    fallback_response: str
    history_turns: int
    max_sources: int
    unclear_max_words: int
    unclear_response: str
    addition_max_words: int
    small_talk: dict[str, SmallTalkConfig]
    clarification: ClarificationConfig
    programs: tuple[ProgramConfig, ...]


@dataclass(frozen=True)
class RetrievalConfig:
    """Einstellungen der hybriden Suche."""

    top_k: int
    candidates: int
    rrf_k: int
    # Suchtiefe im HNSW-Graphen des Vektorindex
    ef_search: int


@dataclass(frozen=True)
class LLMModelConfig:
    """Ein Sprachmodell mit dem genauen Stand seines Repositorys."""

    repo: str
    revision: str


@dataclass(frozen=True)
class LLMConfig:
    """Auswahl des Sprachmodells und Adresse des Inferenzservers."""

    active: str
    server_url: str
    temperature: float
    max_tokens: int
    timeout: float
    chat_template_args: dict
    models: dict[str, LLMModelConfig]

    @property
    def model(self) -> LLMModelConfig:
        """Das aktive Modell."""
        return self.models[self.active]


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Ein Embedding-Modell mit den Präfixen, die es vor Anfragen und Chunks erwartet."""

    repo: str
    query_prefix: str
    document_prefix: str
    trust_remote_code: bool


@dataclass(frozen=True)
class EmbeddingConfig:
    """Auswahl und Einstellungen der Embedding-Modelle."""

    active: str
    max_tokens: int
    batch_size: int
    models: dict[str, EmbeddingModelConfig]

    @property
    def model(self) -> EmbeddingModelConfig:
        """Das aktive Modell."""
        return self.models[self.active]


@dataclass(frozen=True)
class KeywordIndexConfig:
    """Einstellungen der lexikalischen Suche."""

    language: str


@dataclass(frozen=True)
class JudgeConfig:
    """Externes Bewerter-LLM für die Bewertung der Testläufe."""

    provider: str
    model: str
    # Denktiefe des Bewerters
    effort: str
    # Bewerter-Prompt, relativ zum Projektordner
    prompt: str
    # Wartezeit zwischen zwei Anfragen und Wiederholungen bei Ratenbegrenzung
    pause_seconds: float
    retry_seconds: float
    max_attempts: int


@dataclass(frozen=True)
class TestsConfig:
    """Einstellungen für die technischen Tests."""

    # Testdatensatz und Ordner der Läufe, relativ zum Projektordner
    testset: str
    runs_dir: str
    # Fragen vor der Messung, nicht aus dem Testdatensatz
    warmup_questions: tuple[str, ...]
    judge: JudgeConfig


@dataclass(frozen=True)
class Config:
    """Gesamte Konfiguration einer Hochschule."""

    university_name: str
    crawler: CrawlerConfig
    preprocessing: PreprocessingConfig
    chunking: ChunkingConfig
    student_service: StudentServiceConfig
    chatbot: ChatbotConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    keyword_index: KeywordIndexConfig
    tests: TestsConfig


def _with_tuples(section: dict) -> dict:
    """Wandelt Listen in Tupel um, damit die Konfiguration unveränderlich bleibt."""
    return {key: tuple(value) if isinstance(value, list) else value for key, value in section.items()}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Liest die Konfiguration aus einer TOML-Datei.

    Args:
        path: Pfad zur Konfigurationsdatei.

    Returns:
        Die eingelesene Konfiguration.
    """
    with path.open("rb") as file:
        data = tomllib.load(file)

    return Config(
        university_name=data["university"]["name"],
        crawler=CrawlerConfig(**_with_tuples(data["crawler"])),
        preprocessing=PreprocessingConfig(
            **_with_tuples({key: value for key, value in data["preprocessing"].items() if key != "semester_overviews"}),
            semester_overviews=SemesterOverviewConfig(**data["preprocessing"]["semester_overviews"]),
        ),
        chunking=ChunkingConfig(**data["chunking"]),
        student_service=StudentServiceConfig(**data["student_service"]),
        chatbot=ChatbotConfig(
            **{
                key: value
                for key, value in data["chatbot"].items()
                if key not in ("small_talk", "clarification", "programs")
            },
            small_talk={
                kind: SmallTalkConfig(**_with_tuples(entry))
                for kind, entry in data["chatbot"].get("small_talk", {}).items()
            },
            clarification=ClarificationConfig(**_with_tuples(data["chatbot"]["clarification"])),
            programs=tuple(ProgramConfig(**_with_tuples(entry)) for entry in data["chatbot"].get("programs", [])),
        ),
        retrieval=RetrievalConfig(**data["retrieval"]),
        llm=LLMConfig(
            **{key: value for key, value in data["llm"].items() if key != "models"},
            models={key: LLMModelConfig(**model) for key, model in data["llm"]["models"].items()},
        ),
        embedding=EmbeddingConfig(
            active=data["embedding"]["active"],
            max_tokens=data["embedding"]["max_tokens"],
            batch_size=data["embedding"]["batch_size"],
            models={key: EmbeddingModelConfig(**model) for key, model in data["embedding"]["models"].items()},
        ),
        keyword_index=KeywordIndexConfig(**data["keyword_index"]),
        tests=TestsConfig(
            **_with_tuples({key: value for key, value in data["tests"].items() if key != "judge"}),
            judge=JudgeConfig(**data["tests"]["judge"]),
        ),
    )
