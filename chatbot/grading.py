"""Bewertung eines Testlaufs.

Zwei Arten von Kennzahlen werden ermittelt:

- **Bewerter-LLM:** Ein externes Sprachmodell beurteilt je Testfall Verhalten, Korrektheit
  gegenüber der Musterantwort, Treue zu den Auszügen, Relevanz und ob die angezeigte Quelle
  passt. Der Prompt liegt offen in ``technische-tests/bewerter-prompt.txt``. Übermittelt
  werden nur Testfragen, Auszüge der öffentlichen Webseite und Antworten des Chatbots.
- **Eigene Kennzahlen:** ohne Sprachmodell aus den gespeicherten Messwerten, etwa ob die
  Musterquelle unter den gefundenen Chunks ist, ob der Fallback richtig ausgelöst wurde,
  Antwortzeiten und Anrede.

Die Ergebnisse liegen im Ordner des Laufs: ``bewertung.jsonl`` je Testfall,
``bewertung.csv`` zum Lesen, ``kennzahlen.json`` als Zusammenfassung und ``bewertung.json``
mit Bewerter-Modell und Zeitpunkt.
"""

import csv
import json
import logging
import re
import shutil
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field

from chatbot.config import JudgeConfig

logger = logging.getLogger(__name__)

# Erkennung des erwarteten Verhaltens aus der Spalte „Erwartetes Verhalten“
_EXPECTS_FALLBACK = re.compile(r"keine Information|nur Informationen der Universität|nicht zu den Aufgaben")
# Höfliche Anrede mitten im Satz; am Satzanfang ist „Sie“ nicht von „sie“ zu unterscheiden
_FORMAL_ADDRESS = re.compile(r"[^.!?:\n„\"]\s+(Sie|Ihnen|Ihr|Ihre|Ihren|Ihrem|Ihrer)\b")
# Grenze für die Antwortzeit
TIME_LIMIT_SECONDS = 20.0


class Judgement(BaseModel):
    """Urteil des Bewerters zu einem Testfall; die Kriterien stehen im Bewerter-Prompt."""

    begruendung: str = Field(description="Ein bis drei kurze Sätze, die die Bewertung begründen.")
    verhalten: Literal["erfuellt", "teilweise", "nicht_erfuellt"]
    rueckfrage: bool
    korrektheit: Literal["korrekt", "teilweise", "falsch", "nicht_anwendbar"]
    treue: Literal["vollstaendig", "teilweise", "nicht", "nicht_anwendbar"]
    relevanz: Literal["hoch", "mittel", "niedrig"]
    kontext_relevanz: Literal["hoch", "mittel", "niedrig"]
    quelle_passend: Literal["ja", "nein", "keine_quelle"]


def normalize_url(url: str) -> str:
    """Vereinheitlicht eine Adresse für den Vergleich: ohne Sprungmarke und Schrägstrich am Ende."""
    parts = urlsplit(url.strip())
    path = unquote(parts.path).rstrip("/")
    return f"{parts.netloc.lower()}{path}"


def expected_behaviour(case: dict[str, str]) -> str:
    """Erwartetes Verhalten als Kategorie: ``rueckfrage``, ``fallback`` oder ``antwort``."""
    if case.get("Antwort auf Rückfrage", "").strip():
        return "rueckfrage"
    if _EXPECTS_FALLBACK.search(case.get("Erwartetes Verhalten", "")):
        return "fallback"
    return "antwort"


def _source_link(source: dict) -> str:
    """Adresse einer angezeigten Quelle, bei PDF-Dokumenten mit Seite."""
    return f"{source['url']}#page={source['seite']}" if source.get("seite") else source["url"]


def build_prompt(template: str, result: dict) -> str:
    """Setzt Testfall, Gespräch und Auszüge der letzten Antwort in den Bewerter-Prompt ein."""
    case, turns = result["testfall"], result["verlauf"]
    last = turns[-1]
    conversation = []
    for turn in turns:
        conversation.append(f"Person: {turn['nachricht']}")
        sources = ", ".join(_source_link(source) for source in turn["quellen"]) or "keine"
        conversation.append(f"Chatbot: {turn['antwort']}\n(angezeigte Quelle: {sources})")
    excerpts = "\n\n".join(
        f"[{chunk['rang']}] {chunk['titel']} – {chunk['url']}{f' (Seite {chunk['seite']})' if chunk.get('seite') else ''}\n{chunk['text']}"
        for chunk in last["gefundene_chunks"]
    )
    return template.format(
        frage=case["Frage"],
        kontext=case.get("Kontext") or "–",
        antwort_auf_rueckfrage=case.get("Antwort auf Rückfrage") or "–",
        musterantwort=case.get("Musterantwort") or "– (keine, es wird ein Fallback erwartet)",
        musterquelle=case.get("Quelle") or "–",
        erwartetes_verhalten=case.get("Erwartetes Verhalten") or "–",
        gespraech="\n\n".join(conversation),
        auszuege=excerpts or "– (keine)",
        angezeigte_quelle=", ".join(_source_link(source) for source in last["quellen"]) or "keine",
    )


def own_metrics(result: dict) -> dict:
    """Kennzahlen eines Testfalls, die ohne Sprachmodell berechnet werden."""
    case = result["testfall"]
    measured = [turn for turn in result["verlauf"] if turn["gemessen"]]
    last = measured[-1]
    expected = expected_behaviour(case)
    reference = normalize_url(case["Quelle"]) if case.get("Quelle", "").strip() else None
    rank = None
    if reference:
        rank = next(
            (chunk["rang"] for chunk in last["gefundene_chunks"] if normalize_url(chunk["url"]) == reference), None
        )
    shown = normalize_url(last["quellen"][0]["url"]) if last["quellen"] else None
    answers = " ".join(turn["antwort"] for turn in measured)
    return {
        "erwartet": expected,
        "fallback": last["fallback"],
        "fallback_richtig": last["fallback"] == (expected == "fallback"),
        "musterquelle_rang": rank,
        "quelle_identisch": (shown == reference) if reference else None,
        "anrede_sie": bool(_FORMAL_ADDRESS.search(answers)),
        "woerter": len(last["antwort"].split()),
        "zeiten": [turn["zeiten"]["gesamt"] for turn in measured],
        "ungueltige_zitate": sum(len(turn["ungueltige_zitate"]) for turn in measured),
    }


class DailyLimitReached(RuntimeError):
    """Kontingent oder Guthaben des Bewerters ist erschöpft; weitere Anfragen sind zwecklos."""


def _judgement_schema() -> dict:
    """JSON-Schema des Urteils für strukturierte Ausgaben; zusätzliche Felder sind nicht erlaubt."""
    schema = Judgement.model_json_schema()
    schema["additionalProperties"] = False
    return schema


class ClaudeJudge:
    """Bewerter über die Claude API. Der Schlüssel wird aus ``ANTHROPIC_API_KEY`` gelesen."""

    def __init__(self, config: JudgeConfig) -> None:
        import anthropic

        self.config = config
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=300.0)

    def judge(self, prompt: str) -> tuple[Judgement, dict]:
        """Bewertet einen Testfall. Bei Ratenbegrenzung oder Serverfehlern wird gewartet und wiederholt.

        Returns:
            Das Urteil und die Tokenzahlen samt antwortendem Modell.

        Raises:
            DailyLimitReached: Das Guthaben ist aufgebraucht.
        """
        anthropic = self.anthropic
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.client.beta.messages.create(
                    model=self.config.model,
                    max_tokens=16000,
                    output_config={
                        "effort": self.config.effort,
                        "format": {"type": "json_schema", "schema": _judgement_schema()},
                    },
                    # Lehnt das Modell eine Anfrage aus Sicherheitsgründen ab, beantwortet sie ein
                    # Ersatzmodell; welches Modell geantwortet hat, wird je Fall festgehalten.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    messages=[{"role": "user", "content": prompt}],
                )
                if response.stop_reason == "refusal":
                    raise RuntimeError(f"Bewerter hat die Anfrage abgelehnt ({response.stop_details})")
                text = next(block.text for block in response.content if block.type == "text")
                # Kennung der Antwort, um den Aufruf in der Abrechnung wiederzufinden
                usage = {
                    "anfrage_id": response.id,
                    "modell": response.model,
                    "eingabe": response.usage.input_tokens,
                    "ausgabe": response.usage.output_tokens,
                }
                return Judgement.model_validate_json(text), usage
            except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as error:
                if attempt == self.config.max_attempts:
                    raise
                wait = self.config.retry_seconds * attempt
                logger.warning("Bewerter: %s, neuer Versuch in %d s", type(error).__name__, wait)
                time.sleep(wait)
            except anthropic.BadRequestError as error:
                if "credit balance" in str(error).lower():
                    raise DailyLimitReached(str(error)) from error
                raise
        raise RuntimeError("unerreichbar")


def grade_run(
    run_dir: Path, prompt_path: Path, config: JudgeConfig, only: set[int] | None = None, dry_run: bool = False
) -> dict:
    """Bewertet einen Testlauf und schreibt die Ergebnisse in seinen Ordner.

    Args:
        run_dir: Ordner des Testlaufs mit ``ergebnisse.jsonl``.
        prompt_path: Bewerter-Prompt mit Platzhaltern.
        config: Anbieter, Modell und Wiederholungen des Bewerters.
        only: Optional nur diese Nummern bewerten.
        dry_run: Nur eigene Kennzahlen berechnen und den ersten Prompt ausgeben, ohne Anfrage
            an den Bewerter und ohne Dateien zu schreiben.

    Returns:
        Die zusammengefassten Kennzahlen.
    """
    template = prompt_path.read_text(encoding="utf-8")
    with (run_dir / "ergebnisse.jsonl").open(encoding="utf-8") as file:
        results = [json.loads(line) for line in file if line.strip()]
    if only:
        results = [result for result in results if int(result["testfall"]["Nr."]) in only]

    if dry_run:
        print(build_prompt(template, results[0]))
        graded = [{"nr": r["testfall"]["Nr."], "eigene": own_metrics(r), "bewerter": None} for r in results]
        return summarize(graded)

    if config.provider != "anthropic":
        raise ValueError(f"Bewerter-Anbieter {config.provider!r} ist nicht umgesetzt")
    judge = ClaudeJudge(config)
    started_at = datetime.now().astimezone()
    # Jedes Urteil wird sofort gespeichert. Ein erneuter Aufruf übernimmt bereits bewertete
    # Fälle und setzt fort, etwa nach aufgebrauchtem Guthaben.
    progress_path = run_dir / "bewertung.jsonl"
    done: dict[str, dict] = {}
    if progress_path.exists():
        with progress_path.open(encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                if entry.get("bewerter") and entry.get("modell", config.model) == config.model:
                    done[entry["nr"]] = entry
    graded = []
    limit_reached = False
    with progress_path.open("w", encoding="utf-8") as file:
        for entry in done.values():
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for number, result in enumerate(results, start=1):
            case = result["testfall"]
            if case["Nr."] in done:
                graded.append(done[case["Nr."]])
                continue
            entry = {"nr": case["Nr."], "frage": case["Frage"], "eigene": own_metrics(result), "bewerter": None}
            if not limit_reached:
                logger.info("Bewertung %s (%d/%d)", case["Nr."], number, len(results))
                try:
                    judgement, usage = judge.judge(build_prompt(template, result))
                    entry.update(bewerter=judgement.model_dump(), token=usage, modell=config.model)
                    file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    file.flush()
                    time.sleep(config.pause_seconds)
                except DailyLimitReached as error:
                    logger.error("Guthaben oder Kontingent des Bewerters erschöpft, Abbruch: %s", error)
                    limit_reached = True
                    entry["fehler"] = "Guthaben oder Kontingent erschöpft"
                except Exception as error:  # noqa: BLE001 – ein Fehler soll den Lauf nicht abbrechen
                    logger.error("Bewertung %s fehlgeschlagen: %s", case["Nr."], error)
                    entry["fehler"] = str(error)
            else:
                entry["fehler"] = "nicht bewertet (Guthaben oder Kontingent erschöpft)"
            graded.append(entry)

    summary = summarize(graded)
    _write_overview(run_dir / "bewertung.csv", graded)
    (run_dir / "kennzahlen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy(prompt_path, run_dir / "bewerter-prompt.txt")
    meta = {
        "beginn": started_at.isoformat(timespec="seconds"),
        "anbieter": config.provider,
        "modell": config.model,
        "testfaelle": len(graded),
        "bewertet": sum(1 for entry in graded if entry.get("bewerter")),
        "fehler": sum(1 for entry in graded if "fehler" in entry),
        "vollstaendig": not limit_reached and all(entry.get("bewerter") for entry in graded),
        # Verbrauch aller Urteile, auch der aus einem früheren, abgebrochenen Aufruf
        "token": {
            "eingabe": sum(entry["token"]["eingabe"] for entry in graded if entry.get("token")),
            "ausgabe": sum(entry["token"]["ausgabe"] for entry in graded if entry.get("token")),
        },
    }
    (run_dir / "bewertung.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _share(values: list[bool]) -> float | None:
    """Anteil der wahren Werte, gerundet; ``None`` ohne Werte."""
    return round(sum(values) / len(values), 3) if values else None


def _count(values: list[str]) -> dict[str, int]:
    """Häufigkeit je Ausprägung."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def summarize(graded: list[dict]) -> dict:
    """Fasst die Kennzahlen aller Testfälle zusammen."""
    own = [entry["eigene"] for entry in graded]
    by_expected = {kind: [o for o in own if o["erwartet"] == kind] for kind in ("antwort", "rueckfrage", "fallback")}
    with_reference = [o for o in own if o["quelle_identisch"] is not None]
    ranks = [o["musterquelle_rang"] for o in with_reference]
    times = [seconds for o in own for seconds in o["zeiten"]]

    summary = {
        "testfaelle": len(graded),
        "erwartet": {kind: len(items) for kind, items in by_expected.items()},
        "suche": {
            # Strenge Untergrenze: nur exakt gleiche Adresse zählt, gleichwertige Seiten nicht
            "musterquelle_unter_auszuegen": _share([rank is not None for rank in ranks]),
            "mrr": round(statistics.fmean([1 / rank if rank else 0 for rank in ranks]), 3) if ranks else None,
            "angezeigte_quelle_identisch": _share([o["quelle_identisch"] for o in with_reference]),
        },
        "fallback": {
            "bei_erwartetem_fallback_ausgeloest": _share([o["fallback"] for o in by_expected["fallback"]]),
            "faelschlich_ausgeloest": _share([o["fallback"] for o in by_expected["antwort"] + by_expected["rueckfrage"]]),
        },
        "zeit": {
            "nachrichten": len(times),
            "median": round(statistics.median(times), 1) if times else None,
            "maximum": round(max(times), 1) if times else None,
            "ueber_20_s": _share([seconds > TIME_LIMIT_SECONDS for seconds in times]),
        },
        "anrede_sie": sum(o["anrede_sie"] for o in own),
        "ungueltige_zitate": sum(o["ungueltige_zitate"] for o in own),
        "woerter_median": statistics.median([o["woerter"] for o in own]) if own else None,
    }
    judged = [entry for entry in graded if entry.get("bewerter")]
    if judged:
        verdicts = [entry["bewerter"] for entry in judged]
        expected_of = {entry["nr"]: entry["eigene"]["erwartet"] for entry in judged}
        summary["bewerter"] = {
            "bewertet": len(judged),
            "fehler": sum(1 for entry in graded if "fehler" in entry),
            "verhalten": _count([v["verhalten"] for v in verdicts]),
            "verhalten_je_erwartung": {
                kind: _count([e["bewerter"]["verhalten"] for e in judged if expected_of[e["nr"]] == kind])
                for kind in ("antwort", "rueckfrage", "fallback")
            },
            "rueckfrage_gestellt_wenn_erwartet": _share(
                [e["bewerter"]["rueckfrage"] for e in judged if expected_of[e["nr"]] == "rueckfrage"]
            ),
            "korrektheit": _count([v["korrektheit"] for v in verdicts]),
            "treue": _count([v["treue"] for v in verdicts]),
            "relevanz": _count([v["relevanz"] for v in verdicts]),
            "kontext_relevanz": _count([v.get("kontext_relevanz", "") for v in verdicts]),
            "quelle_passend": _count([v["quelle_passend"] for v in verdicts]),
        }
    return summary


def _write_overview(path: Path, graded: list[dict]) -> None:
    """Schreibt je Testfall eine Zeile mit Urteil und eigenen Kennzahlen."""
    header = [
        "Nr.", "Frage", "Erwartet", "Verhalten", "Rückfrage", "Korrektheit", "Treue", "Relevanz",
        "Kontextrelevanz", "Quelle passend", "Musterquelle unter Auszügen (Rang)", "Fallback", "Zeit (s)", "Begründung",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        for entry in graded:
            own, verdict = entry["eigene"], entry.get("bewerter") or {}
            writer.writerow([
                entry["nr"],
                entry.get("frage", ""),
                own["erwartet"],
                verdict.get("verhalten", "Fehler" if "fehler" in entry else ""),
                {True: "ja", False: "nein"}.get(verdict.get("rueckfrage"), ""),
                verdict.get("korrektheit", ""),
                verdict.get("treue", ""),
                verdict.get("relevanz", ""),
                verdict.get("kontext_relevanz", ""),
                verdict.get("quelle_passend", ""),
                own["musterquelle_rang"] or ("–" if own["quelle_identisch"] is None else "nicht gefunden"),
                "ja" if own["fallback"] else "nein",
                str(round(sum(own["zeiten"]), 1)).replace(".", ","),
                verdict.get("begruendung", entry.get("fehler", "")),
            ])
