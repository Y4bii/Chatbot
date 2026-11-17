"""Zentrale Steuerung der Beantwortung einer Anfrage.

Die Steuerung koordiniert die übrigen Komponenten:

1. Folgefragen werden mithilfe der letzten Nachrichten vom Sprachmodell in eine
   eigenständige Frage umformuliert.
2. Die hybride Suche liefert die passendsten Chunks.
3. Aus Systemprompt, nummerierten Chunks und der eigenständigen Frage entsteht der
   Prompt für das Sprachmodell.
4. Die Chunk-Nummern in der Antwort werden durch Titel und Adresse der Quellen ersetzt.
   Das Modell erzeugt Adressen also nie selbst.
5. Beginnt die Antwort mit der Fallback-Markierung, wird sie entfernt und um die
   Kontaktdaten des Studierendenservice ergänzt.

Anfragen und Gesprächsverläufe werden nicht gespeichert. Der Verlauf wird bei jeder
Anfrage vollständig übergeben.
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from chatbot.config import Config, ProgramConfig
from chatbot.llm import LLMClient
from chatbot.retrieval import HybridRetriever, RetrievedChunk

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Folge von Verweisen einschließlich Trennzeichen dazwischen, z. B. "[1], [2] und [3]"
_CITATION_GROUP = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\](?:\s*(?:,|und)?\s*\[\d+(?:\s*,\s*\d+)*\])*")
# Wendungen, mit denen das Modell im Satz auf einen Auszug verweist, z. B. ", wie in [1] beschrieben".
# Sie werden samt Verweis entfernt, da sonst ein unvollständiger Satz übrig bliebe.
_CITATION_PHRASE = re.compile(
    r",?\s*(?:wie\s+(?:in|im|unter)\s+(?:den\s+)?(?:Auszügen?\s+)?|siehe\s+(?:Auszug\s+)?|(?:in|im)\s+Auszug\s+"
    # „in den Auszügen unter [1]“ oder „unter [1]“ ohne „wie“, außerdem „findest du in [1]“
    r"|(?:in\s+den\s+Auszügen\s+)?unter\s+|in\s+(?=\[\d)|wie\s+(?=\[\d))"
    r"\[\d+(?:\s*,\s*\d+)*\](?:\s*(?:,|und)?\s*\[\d+(?:\s*,\s*\d+)*\])*"
    # Fundstelle innerhalb des Auszugs, z. B. „wie in [1] Absatz 6 beschrieben“
    r"(?:\s+(?:Absatz|Abs\.|§|Satz|Seite|S\.)\s*\d+\w?)*"
    # Abschließendes Verb wie "beschrieben" oder "dargelegt": ein Wort direkt vor einem Satzzeichen
    r"(?:\s+(?:zu\s+finden|\w+)(?=\s*[.,;:!?)]|\s*$))?"
    # Bei einem Einschub zwischen zwei Kommas entfällt auch das schließende Komma.
    r"(?:\s*,(?=\s))?",
    re.IGNORECASE,
)
# Hinweise in einer Frage auf einen bestimmten Studienbeginn oder eine ältere Fassung
_STUDY_START = re.compile(
    r"\b(Studienbeginn|Studienstart|angefangen|begonnen|eingeschrieben|immatrikuliert|Kohorte"
    r"|älter\w*|alte[nrs]?\s+(?:PO|Prüfungsordnung|Studiengangsordnung|Fassung|Ordnung)"
    r"|seit\s+(?:dem\s+)?(?:WS|SS|WiSe|SoSe|Wintersemester|Sommersemester)|20\d\d|\d{2}/\d{2})\b",
    re.IGNORECASE,
)
# "laut [2]" bzw. "gemäß [2]" trägt die Satzstellung und wird daher durch "laut der Quelle" ersetzt.
_CITATION_ACCORDING = re.compile(
    r"\b(laut|gemäß)\s+(?:Auszug\s+)?\[\d+(?:\s*,\s*\d+)*\](?:\s*(?:,|und)?\s*\[\d+(?:\s*,\s*\d+)*\])*",
    re.IGNORECASE,
)
# Erwähnungen der Auszüge im Antworttext, z. B. „in den bereitgestellten Auszügen“. Die Person
# sieht keine Auszüge, sondern die Seiten der Universität. Der Artikel bleibt erhalten, damit
# der Satz grammatisch stimmt („den Auszügen“ → „den Seiten der Universität“).
_EXCERPT_ADJECTIVE = r"(?:(?:bereitgestellten|vorliegenden|gegebenen|genannten|übergebenen|angegebenen)\s+)?"
_EXCERPT_MENTION = re.compile(rf"\b([Dd](?:en|ie|er))\s+{_EXCERPT_ADJECTIVE}Auszügen?\b")
# „in den Auszügen“ wird zu „auf den Seiten der Universität“, nicht „in den Seiten“
_EXCERPT_MENTION_IN = re.compile(rf"\b([Ii])n\s+den\s+{_EXCERPT_ADJECTIVE}Auszügen\b")


@dataclass
class Source:
    """Eine Quelle der Antwort, wie sie in der Oberfläche angezeigt wird."""

    title: str
    url: str
    breadcrumb: list[str]
    # "page" für Webseiten, "document" für PDF-Dokumente
    kind: str = "page"
    # Bei PDF-Dokumenten die Seite, auf der der zitierte Abschnitt beginnt
    page: int | None = None


@dataclass
class Answer:
    """Antwort des Chatbots mit Quellen und Messwerten."""

    text: str
    sources: list[Source]
    fallback: bool
    search_query: str
    citations_from_model: bool
    small_talk: str | None = None
    invalid_citations: list[int] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)


class Controller:
    """Beantwortet Anfragen auf Grundlage der Indizes und des Sprachmodells."""

    def __init__(
        self,
        config: Config,
        retriever: HybridRetriever,
        llm: LLMClient,
        prompt_dir: Path = DEFAULT_PROMPT_DIR,
    ) -> None:
        """
        Args:
            config: Gesamte Konfiguration.
            retriever: Hybride Suche über die Indizes.
            llm: Verbindung zum Inferenzserver.
            prompt_dir: Ordner mit ``system.txt``, ``context.txt`` und ``rewrite.txt``.
        """
        self.config = config
        self.retriever = retriever
        self.llm = llm
        self._placeholders = {
            "university_name": config.university_name,
            "chatbot_name": config.chatbot.name,
            "fallback_marker": config.chatbot.fallback_marker,
        }
        self._system_prompt = (
            (prompt_dir / "system.txt").read_text(encoding="utf-8").format(**self._placeholders).strip()
        )
        self._context_template = (prompt_dir / "context.txt").read_text(encoding="utf-8")
        self._rewrite_template = (prompt_dir / "rewrite.txt").read_text(encoding="utf-8")
        clarification = config.chatbot.clarification
        self._clarification_patterns = [re.compile(pattern) for pattern in clarification.patterns]
        self._degree_pattern = re.compile(clarification.degree_pattern)
        self._relative_semester = re.compile(clarification.relative_semester_pattern)
        self._semester_number = re.compile(clarification.semester_number_pattern)
        self._program_patterns = [
            (program, [re.compile(pattern) for pattern in program.patterns]) for program in config.chatbot.programs
        ]

    def answer(self, question: str, history: list[dict[str, str]] | None = None) -> Answer:
        """Beantwortet eine Anfrage.

        Args:
            question: Aktuelle Anfrage.
            history: Bisheriger Gesprächsverlauf als Nachrichten mit ``role`` (``user``
                oder ``assistant``) und ``content``, älteste zuerst.

        Returns:
            Die Antwort mit Quellen und Messwerten.
        """
        started = time.perf_counter()
        small_talk = self._small_talk(question)
        if small_talk is not None:
            return small_talk

        recent = self._recent_history(history or [])
        timings: dict[str, float] = {}
        tokens = {"prompt": 0, "completion": 0}

        search_query = question
        # Eigene Worte des Nutzers; bei kurzen Angaben samt der ergänzten Frage
        own_text = question
        if recent:
            completed = self._complete_previous(question, recent)
            if completed is None:
                search_query = self._rewrite(question, recent, tokens)
            else:
                # Die ergänzte Frage kann ihrerseits eine Folgefrage sein („Welche brauche ich
                # im 3. Semester?“ nach einer Frage zu Modulen). Sie wird dann mit dem Verlauf
                # vor ihr umformuliert, wie es ohne die Rückfrage geschehen wäre.
                own_text, earlier = completed
                search_query = self._rewrite(own_text, earlier, tokens) if earlier else own_text
            timings["rewrite"] = time.perf_counter() - started
        clarification = self._clarification(search_query, own_text, recent)
        if clarification is not None:
            return self._fixed_answer(clarification, search_query, "rueckfrage")
        search_query = self._expand_abbreviations(search_query)

        step = time.perf_counter()
        top_k = self.config.retrieval.top_k
        # Mehr Treffer anfordern, damit nach dem Aussortieren älterer Fassungen genug übrig bleiben
        retrieved = self._prefer_current(self.retriever.search(search_query, top_k * 3), f"{question} {search_query}")
        retrieved = retrieved[:top_k]
        timings["retrieval"] = time.perf_counter() - step

        step = time.perf_counter()
        # Die Antwort entsteht aus der (bei Folgefragen umformulierten) eigenständigen Frage
        # ohne den Gesprächsverlauf. Mit Verlauf übernahm das Modell im Test frühere Antworten
        # teils wörtlich; den Bezug zum Gespräch stellt bereits die Umformulierung her.
        # Die Regeln werden nach der Frage kurz wiederholt, da kleine Modelle Anweisungen
        # am Ende eines langen Prompts zuverlässiger befolgen.
        context = self._context_template.format(
            chunks=self._format_chunks(retrieved), question=search_query, **self._placeholders
        ).strip()
        messages = [{"role": "system", "content": self._system_prompt}, {"role": "user", "content": context}]
        completion = self.llm.chat(messages)
        timings["generation"] = time.perf_counter() - step
        tokens["prompt"] += completion.prompt_tokens
        tokens["completion"] += completion.completion_tokens

        answer = self._postprocess(completion.text, retrieved, search_query)
        timings["total"] = time.perf_counter() - started
        answer.timings = {name: round(seconds, 3) for name, seconds in timings.items()}
        answer.tokens = tokens
        return answer

    # Begrüßung, Dank und Verabschiedung

    def _small_talk(self, question: str) -> Answer | None:
        """Beantwortet Eingaben, die nur aus einer Floskel bestehen, mit einem festen Text.

        Ohne diese Erkennung würde etwa „Hallo“ wie eine Sachfrage behandelt: Die dichte
        Suche liefert immer die ähnlichsten Chunks, und das Modell formuliert daraus eine
        Antwort zu einem zufälligen Thema.

        Returns:
            Die feste Antwort oder ``None``, wenn die Eingabe keine reine Floskel ist.
        """
        normalized = " ".join(re.sub(r"[^\w\s-]", " ", question.lower()).split())
        for kind, entry in self.config.chatbot.small_talk.items():
            if any(re.fullmatch(pattern, normalized) for pattern in entry.patterns):
                return self._fixed_answer(entry.response.format(**self._placeholders), question, kind)

        # Auffangnetz: kurze Eingaben ohne Fragezeichen, zu denen kein Wort in der Wissensbasis
        # vorkommt, sind keine beantwortbare Frage (z. B. „Heyho“). Sonst würde die dichte
        # Suche zufällig ähnliche Chunks liefern und das Modell daraus eine Antwort bauen.
        if not normalized.strip("-_ "):
            # Eingaben ganz ohne Wörter, etwa „???“
            return self._fixed_answer(self.config.chatbot.unclear_response, question, "unclear")
        short = len(normalized.split()) <= self.config.chatbot.unclear_max_words
        if short and "?" not in question and not self.retriever.has_keyword_match(question):
            return self._fixed_answer(self.config.chatbot.unclear_response, question, "unclear")
        return None

    @staticmethod
    def _fixed_answer(text: str, question: str, kind: str) -> Answer:
        """Feste Antwort ohne Suche, Quellen und Sprachmodell."""
        return Answer(
            text=text,
            sources=[],
            fallback=False,
            search_query=question,
            citations_from_model=False,
            small_talk=kind,
            timings={"total": 0.0},
        )

    # Studiengänge

    def _program(self, text: str) -> ProgramConfig | None:
        """Der erste Studiengang, dessen Muster im (klein geschriebenen) Text vorkommt."""
        for program, patterns in self._program_patterns:
            if any(pattern.search(text) for pattern in patterns):
                return program
        return None

    def _clarification(self, query: str, own_text: str, recent: list[dict[str, str]]) -> str | None:
        """Gegenfrage, wenn die Antwort erkennbar vom Studiengang abhängt, dieser aber fehlt.

        Das Sprachmodell stellte trotz Regel im Prompt keine Rückfragen. Daher wird hier ohne
        Modell geprüft: Deutet die Frage auf Angaben je Studiengang hin (etwa „dieses Semester“
        oder „Wahlpflicht“) und nennen weder sie noch die vorherigen Fragen einen Studiengang,
        fragt der Chatbot danach. Gibt es den genannten Studiengang als Bachelor und als Master,
        ohne dass ein Abschluss genannt ist, fragt er nach dem Abschluss. Bezieht sich die Frage
        auf das eigene Semester („dieses Semester“), ohne es zu nennen, fragt er danach.

        Studiengang und Abschluss werden nur in den Worten des Nutzers gesucht, nicht in der
        umformulierten Frage: Das Modell übernahm beim Umformulieren den Abschluss eines zuvor
        besprochenen Studiengangs. Maßgeblich ist der zuletzt genannte Studiengang, und ein
        Abschluss zählt nur aus Nachrichten, die nach dem Wechsel zu diesem Studiengang kamen.

        Args:
            query: Suchanfrage, bei Folgefragen umformuliert; nur für die Hinweise auf
                Angaben je Studiengang.
            own_text: Aktuelle Nachricht, bei kurzen Angaben samt der ergänzten Frage.
            recent: Letzte Nachrichten des Gesprächs.
        """
        if not any(pattern.search(query.lower()) for pattern in self._clarification_patterns):
            return None
        settings = self.config.chatbot.clarification
        program, same_program = self._current_program(own_text, recent)
        if program is None:
            return settings.program_response
        if len(program.degrees) > 1 and not self._degree_pattern.search(same_program):
            return settings.degree_response.format(name=program.name)
        previous = " ".join(message["content"] for message in recent if message["role"] == "user")
        text = f"{previous} {own_text}".lower()
        if self._relative_semester.search(text) and not self._semester_number.search(text):
            return settings.semester_response
        return None

    def _current_program(self, own_text: str, recent: list[dict[str, str]]) -> tuple[ProgramConfig | None, str]:
        """Der zuletzt genannte Studiengang und die Nachrichten des Nutzers, die zu ihm gehören.

        Die Nachrichten werden von der neuesten an gesammelt, bis eine einen anderen
        Studiengang nennt. Angaben wie der Abschluss gelten nur aus diesen Nachrichten.

        Returns:
            Der Studiengang oder ``None`` und die zugehörigen Nachrichten in Kleinschreibung.
        """
        previous = [message["content"].lower() for message in recent if message["role"] == "user"]
        program = None
        same_program: list[str] = []
        for message in [own_text.lower(), *reversed(previous)]:
            named = self._program(message)
            if program is not None and named is not None and named is not program:
                break
            program = program or named
            same_program.append(message)
        return program, " ".join(same_program)

    def _expand_abbreviations(self, query: str) -> str:
        """Ergänzt Abkürzungen von Studiengängen um den vollen Namen, z. B. „MML“."""
        for program in self.config.chatbot.programs:
            if program.name.lower() in query.lower():
                continue
            for abbreviation in program.abbreviations:
                pattern = re.compile(rf"\b{re.escape(abbreviation)}\b", re.IGNORECASE)
                query, count = pattern.subn(lambda match: f"{match.group(0)} ({program.name})", query, count=1)
                if count:
                    break
        return query

    # Folgefragen

    def _recent_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """Die letzten Fragen und Antworten des Gesprächs, nur mit den benötigten Feldern."""
        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in history
            if message.get("role") in ("user", "assistant") and message.get("content")
        ]
        return messages[-2 * self.config.chatbot.history_turns :]

    def _is_addition(self, text: str) -> bool:
        """Ist eine Nachricht nur eine kurze Angabe ohne Fragezeichen, etwa „Bachelor“?"""
        return "?" not in text and len(text.split()) <= self.config.chatbot.addition_max_words

    def _complete_previous(
        self, question: str, recent: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]] | None:
        """Hängt eine kurze Angabe an die vorherige Frage an.

        Antworten auf Rückfragen („Bachelor IT-Sicherheit, 3. Semester“) formulierte das
        Sprachmodell teils aus seiner eigenen letzten Antwort statt aus der Frage um.
        Folgen mehrere Angaben aufeinander, werden alle an die letzte echte Frage angehängt.

        Returns:
            Die ergänzte Frage und die Nachrichten vor der ursprünglichen Frage, oder
            ``None``, wenn die Nachricht keine solche Angabe ist.
        """
        if not self._is_addition(question):
            return None
        parts = [question.strip()]
        for position in range(len(recent) - 1, -1, -1):
            message = recent[position]
            if message["role"] != "user":
                continue
            parts.insert(0, message["content"].strip())
            if not self._is_addition(message["content"]):
                return " ".join(parts), recent[:position]
        # War schon die erste Nachricht kurz und ohne Fragezeichen, etwa ein Stichwort wie
        # „Klausurzulassung [Kurs]“, gilt sie als Frage, die nun ergänzt wird.
        return (" ".join(parts), []) if len(parts) > 1 else None

    def _rewrite(self, question: str, recent: list[dict[str, str]], tokens: dict[str, int]) -> str:
        """Formuliert eine Folgefrage in eine eigenständige Frage für die Suche um."""
        labels = {"user": "Frage", "assistant": "Antwort"}
        history = "\n".join(f"{labels[message['role']]}: {message['content']}" for message in recent)
        prompt = self._rewrite_template.format(history=history, question=question).strip()
        completion = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=128)
        tokens["prompt"] += completion.prompt_tokens
        tokens["completion"] += completion.completion_tokens
        rewritten = completion.text.strip()
        # Das Modell übernimmt teils die Beschriftung aus dem Prompt („Eigenständige Frage:“).
        rewritten = re.sub(r"^(eigenständige|neue)\s+frage\s*:\s*", "", rewritten, flags=re.IGNORECASE)
        rewritten = rewritten.strip().strip("\"„“").strip()
        # Unbrauchbare Ergebnisse, etwa leere oder ausufernde Texte, werden verworfen.
        if not rewritten or len(rewritten) > 3 * len(question) + 200:
            return question
        # Ebenso Ergebnisse, die eine frühere Antwort wörtlich übernehmen. Das Modell setzte
        # nach einem Themenwechsel teils die letzte Antwort vor die neue Frage, sodass die
        # Suche wieder Treffer zum alten Thema lieferte.
        if self._copies_answer(rewritten, question, recent):
            return question
        # Nennt die Frage selbst einen Studiengang, darf die Umformulierung ihn nicht gegen den
        # zuvor besprochenen austauschen; das Modell hielt sich nicht verlässlich an diese Regel.
        # Ebenso darf sie keinen Abschluss hinzufügen, den der Nutzer für diesen Studiengang
        # nicht genannt hat.
        named = self._program(question.lower())
        if named is not None:
            if self._program(rewritten.lower()) is not named:
                return question
            _, same_program = self._current_program(question, recent)
            if not self._degrees(rewritten) <= self._degrees(same_program):
                return question
        # Lässt die Umformulierung eine Zahl der Nachricht weg, etwa das Fachsemester aus
        # „Bachelor Robotik, 4. Semester“, wird die Nachricht angehängt, damit die Angabe
        # für Suche und Antwort erhalten bleibt.
        if not set(re.findall(r"\d+", question)) <= set(re.findall(r"\d+", rewritten)):
            return f"{rewritten} {question}"
        return rewritten

    def _degrees(self, text: str) -> set[str]:
        """Die im Text genannten Abschlüsse, vereinheitlicht zu „bachelor“ und „master“."""
        return {
            "bachelor" if match.group(0).startswith("b") else "master"
            for match in self._degree_pattern.finditer(text.lower())
        }

    @staticmethod
    def _copies_answer(rewritten: str, question: str, recent: list[dict[str, str]], length: int = 8) -> bool:
        """Enthält die Umformulierung eine längere Wortfolge aus einer Antwort des Chatbots?

        Wortfolgen, die auch in den Nachrichten des Nutzers stehen, zählen nicht, da eine
        Umformulierung Teile früherer Fragen übernehmen darf.
        """

        def sequences(text: str) -> set[tuple[str, ...]]:
            words = re.findall(r"\w+", text.lower())
            return {tuple(words[i : i + length]) for i in range(len(words) - length + 1)}

        own = sequences(question)
        for message in recent:
            if message["role"] == "user":
                own |= sequences(message["content"])
        copied = sequences(rewritten) - own
        return any(
            copied & sequences(message["content"]) for message in recent if message["role"] == "assistant"
        )

    # Fassungen von Regelungen

    @staticmethod
    def _prefer_current(retrieved: list[RetrievedChunk], text: str) -> list[RetrievedChunk]:
        """Sortiert Chunks aus älteren Fassungen von Ordnungen aus, sofern die Frage keinen Studienbeginn nennt.

        Ältere Fassungen gelten nur für Studierende mit früherem Studienbeginn. Ohne einen
        solchen Hinweis in der Frage stützt sich die Antwort daher auf die aktuelle Fassung.
        Bleibt nach dem Aussortieren nichts übrig, werden die Chunks unverändert verwendet.
        """
        if _STUDY_START.search(text):
            return retrieved
        current = [result for result in retrieved if result.chunk.version != "older"]
        return current or retrieved

    # Prompt

    @staticmethod
    def _format_chunks(retrieved: list[RetrievedChunk]) -> str:
        """Nummeriert die Chunks, der relevanteste zuerst. Jeder Chunk beginnt mit Titel und Überschriften."""
        return "\n\n".join(f"[{number}] {result.chunk.text}" for number, result in enumerate(retrieved, start=1))

    # Nachbearbeitung

    def _postprocess(self, text: str, retrieved: list[RetrievedChunk], search_query: str) -> Answer:
        """Entfernt die Fallback-Markierung und ersetzt Chunk-Nummern durch Quellen."""
        marker = self.config.chatbot.fallback_marker
        # Laut Prompt steht die Markierung am Anfang. Kleine Modelle setzen sie aber auch
        # ans Ende oder formatieren sie fett, daher wird sie an jeder Stelle erkannt.
        fallback = marker in text
        if fallback:
            text = text.replace(marker, " ")
            text = re.sub(r"(\*\*|__)\s*(\*\*|__)", "", text).strip("*_:–- \n")

        cited: list[int] = []
        invalid: list[int] = []
        for match in _CITATION.finditer(text):
            for number in (int(value) for value in match.group(1).split(",")):
                if 1 <= number <= len(retrieved):
                    if number not in cited:
                        cited.append(number)
                elif number not in invalid:
                    invalid.append(number)
        # Verweise entfernen und verbleibende Satzzeichen bereinigen, etwa ".,," zu "."
        # Steht das Komma nur hinter der Wendung, bleibt es erhalten: „X in den Auszügen unter [1], Y“
        # wird zu „X, Y“. Ist die Wendung beidseitig durch Kommas abgetrennt, entfallen beide:
        # „X, wie in [1] beschrieben, und Y“ wird zu „X und Y“.
        def replace_phrase(match: re.Match) -> str:
            phrase = match.group(0)
            at_sentence_start = not match.string[: match.start()].strip() or match.string[: match.start()].rstrip()[-1] in ".!?"
            # „Wie in [1] beschrieben, gibt es …“ am Satzanfang: Ersatz, damit der Satz vollständig bleibt
            if at_sentence_start and phrase.rstrip().endswith(","):
                return "Laut der Quelle" if match.start() == 0 else " Laut der Quelle"
            keep_comma = phrase.rstrip().endswith(",") and not phrase.startswith(",")
            return "," if keep_comma else ""

        text = _CITATION_PHRASE.sub(replace_phrase, text)
        text = _CITATION_ACCORDING.sub(lambda match: f"{match.group(1)} der Quelle", text)
        text = _CITATION_GROUP.sub("", text)
        text = _EXCERPT_MENTION_IN.sub(
            lambda match: f"{'A' if match.group(1) == 'I' else 'a'}uf den Seiten der Universität", text
        )
        text = _EXCERPT_MENTION.sub(lambda match: f"{match.group(1)} Seiten der Universität", text)
        # „den Auszügen der Universität zu Lübeck“ ergäbe sonst „den Seiten der Universität der Universität …“
        text = re.sub(r"Seiten der Universität\s+der Universität(?:\s+zu\s+\w+)?", "Seiten der Universität", text)
        # „den Auszügen der Prüfungsverfahrensordnung“ wird zu „der Prüfungsverfahrensordnung“
        # ebenso „zur …“ → „der …“ und „zum …“ → „dem …“
        articles = {"der": "der", "des": "des", "zur": "der", "zum": "dem"}
        text = re.sub(
            r"\b([Dd])(?:en|ie|er)\s+Seiten der Universität\s+(der|des|zur|zum)\s+",
            lambda match: f"{match.group(1)}{articles[match.group(2)][1:]} ",
            text,
        )
        text = re.sub(r"\(\s*\)", "", text)
        # Sätze, die nach dem Entfernen des Verweises nur noch auf die Quelle zeigen, etwa
        # „Die genauen Schritte findest du [1].“, entfallen; die Quelle steht unter der Antwort.
        text = re.sub(r"(?:^|(?<=[.!?]))\s*[^.!?]*\b(?:findest du|kannst du nachlesen|kannst du nachschauen)\s*[.!?]", "", text)
        # Ebenso „Informationen dazu findest du auf den Seiten der Universität.“
        text = re.sub(
            r"(?:^|(?<=[.!?]))\s*[^.!?]*\b(?:findest du|nachlesen|nachschauen|informieren)\b[^.!?]*\bSeiten der Universität\s*[.!?]",
            "",
            text,
        )
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.!?;:]),+", r"\1", text)
        text = re.sub(r",{2,}", ",", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        # Bestand die Antwort nur aus Verweisen wie „[1]“, bleibt nach dem Entfernen nichts
        # übrig. Statt einer leeren Antwort gilt das als Fallback.
        if not re.search(r"\w", text):
            fallback = True

        max_sources = self.config.chatbot.max_sources
        # Unter den zitierten Chunks werden die mit dem besten Suchrang angezeigt. Die Chunks
        # sind nach Relevanz nummeriert, eine kleinere Nummer bedeutet also einen besseren Rang.
        sources = self._sources([retrieved[number - 1] for number in sorted(cited)])[:max_sources]
        citations_from_model = bool(sources)
        # Eine reine Rückfrage ohne Verweise beantwortet noch nichts und erhält keine Quelle.
        clarification = not cited and text.rstrip().endswith("?")
        if not sources and not fallback and not clarification:
            # Nennt das Modell keine Nummern, dienen die bestplatzierten Chunks als Quellen.
            sources = self._sources(retrieved)[:max_sources]

        if fallback:
            # Fester Text statt der Formulierung des Modells: Dessen Begründungen, warum nichts
            # gefunden wurde, waren teils erfunden, z. B. dass ein Studiengang nicht angeboten wird.
            text = f"{self.config.chatbot.fallback_response}\n\n{self._contact_text()}"

        return Answer(
            text=text,
            sources=sources,
            fallback=fallback,
            search_query=search_query,
            citations_from_model=citations_from_model,
            invalid_citations=invalid,
            retrieved=retrieved,
        )

    @staticmethod
    def _sources(results: list[RetrievedChunk]) -> list[Source]:
        """Quellen der Chunks, je Adresse nur einmal, in der gegebenen Reihenfolge."""
        sources: list[Source] = []
        seen: set[str] = set()
        for result in results:
            chunk = result.chunk
            if chunk.url not in seen:
                seen.add(chunk.url)
                sources.append(
                    Source(
                        title=chunk.title,
                        url=chunk.url,
                        breadcrumb=chunk.breadcrumb,
                        kind=chunk.kind,
                        page=chunk.page,
                    )
                )
        return sources

    def _contact_text(self) -> str:
        """Kontaktdaten des Studierendenservice aus der Konfiguration."""
        service = self.config.student_service
        details = [
            f"E-Mail: {service.email}" if service.email else "",
            f"Telefon: {service.phone}" if service.phone else "",
            service.url,
        ]
        details = [detail for detail in details if detail]
        suffix = f" ({', '.join(details)})" if details else ""
        return f"Weiterhelfen kann dir der {service.name}{suffix}."
