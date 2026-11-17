"""Testlauf: stellt alle Fragen des Testdatensatzes und speichert Antworten und Messwerte.

Jede Frage wird als eigenes Gespräch geführt. Nennt die Spalte „Kontext“ eine vorherige
Frage, wird diese zuerst gestellt. Enthält die Spalte „Antwort auf Rückfrage“ eine Angabe,
folgt sie als zweite Nachricht. Vor der Messung werden einige Aufwärmfragen gestellt,
die nicht in die Ergebnisse eingehen, da die erste Anfrage nach dem Start deutlich länger
dauert.

Ein Lauf liegt in einem eigenen Ordner mit:

- ``lauf.json``: Zeitpunkt, Modelle mit Revision, Index, Dauer
- ``ergebnisse.jsonl``: alle Nachrichten mit Antwort, Quellen, gefundenen Chunks und Zeiten
- ``ergebnisse.csv``: lesbare Übersicht je Frage
- ``prompts/`` und ``konfiguration.toml``: die verwendeten Fassungen
"""

import csv
import json
import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from chatbot.config import Config
from chatbot.controller import Answer, Controller

logger = logging.getLogger(__name__)

# Vorherige Frage in der Spalte „Kontext“, z. B. Folgefrage zu: „Wie verlängere ich …“
_PREVIOUS_QUESTION = re.compile(r"Folgefrage zu:\s*„(.+?)“")


def load_testset(path: Path) -> list[dict[str, str]]:
    """Liest den Testdatensatz (Semikolon als Trennzeichen, UTF-8 mit oder ohne BOM)."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=";"))


def _turn(message: str, answer: Answer, seconds: float, measured: bool) -> dict:
    """Fasst eine Nachricht und die Antwort des Chatbots für die Ergebnisse zusammen."""
    return {
        "nachricht": message,
        "gemessen": measured,
        "suchanfrage": answer.search_query,
        "antwort": answer.text,
        "fallback": answer.fallback,
        "small_talk": answer.small_talk,
        "quellen": [
            {"titel": source.title, "url": source.url, "seite": source.page} for source in answer.sources
        ],
        "zitate_vom_modell": answer.citations_from_model,
        "ungueltige_zitate": answer.invalid_citations,
        "gefundene_chunks": [
            {
                "rang": rank,
                "url": result.chunk.url,
                "seite": result.chunk.page,
                "titel": result.chunk.title,
                "fassung": result.chunk.version,
                "rang_dicht": result.dense_rank,
                "rang_bm25": result.lexical_rank,
                # Wortlaut, wie ihn das Sprachmodell erhalten hat; Grundlage für die Bewertung der Treue
                "text": result.chunk.text,
            }
            for rank, result in enumerate(answer.retrieved, start=1)
        ],
        # Die Gesamtzeit wird außerhalb der Steuerung gemessen, damit sie auch Begrüßungen erfasst.
        "zeiten": {
            "gesamt": round(seconds, 2),
            **{key: round(value, 2) for key, value in answer.timings.items() if key != "total"},
        },
        "token": answer.tokens,
    }


def _ask(controller: Controller, message: str, history: list[dict[str, str]], measured: bool) -> dict:
    """Stellt eine Nachricht, misst die Zeit und ergänzt den Gesprächsverlauf."""
    started = time.perf_counter()
    answer = controller.answer(message, history)
    seconds = time.perf_counter() - started
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer.text}]
    return _turn(message, answer, seconds, measured)


def run_testset(
    controller: Controller,
    config: Config,
    testset: Path,
    output_dir: Path,
    index_dir: Path,
    embedding_key: str,
    project_dir: Path,
    config_path: Path,
    only: set[int] | None = None,
) -> Path:
    """Führt einen Testlauf durch und speichert die Ergebnisse.

    Args:
        controller: Steuerung des Chatbots, wie sie auch die Testseite nutzt.
        config: Gesamte Konfiguration.
        testset: CSV-Datei des Testdatensatzes.
        output_dir: Ordner, unter dem der Lauf angelegt wird.
        index_dir: Verwendeter Index (wird nur festgehalten).
        embedding_key: Schlüssel des Embedding-Modells.
        project_dir: Hauptordner des Projekts, in dem ``prompts/`` liegt.
        config_path: Verwendete Konfigurationsdatei, wird in den Lauf kopiert.
        only: Optional nur diese Nummern stellen, etwa zum Ausprobieren.

    Returns:
        Der Ordner des Laufs.
    """
    rows = load_testset(testset)
    if only:
        rows = [row for row in rows if int(row["Nr."]) in only]
    started_at = datetime.now().astimezone()
    name = f"{started_at:%Y-%m-%d_%H%M}_{config.llm.active}_{embedding_key}"
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(project_dir / "prompts", run_dir / "prompts")
    shutil.copy(config_path, run_dir / "konfiguration.toml")

    for question in config.tests.warmup_questions:
        logger.info("Aufwärmen: %s", question)
        controller.answer(question, [])

    results = []
    run_started = time.perf_counter()
    for number, row in enumerate(rows, start=1):
        logger.info("Frage %s (%d/%d): %s", row["Nr."], number, len(rows), row["Frage"])
        history: list[dict[str, str]] = []
        turns = []
        previous = _PREVIOUS_QUESTION.search(row.get("Kontext", ""))
        if previous:
            turns.append(_ask(controller, previous.group(1), history, measured=False))
        turns.append(_ask(controller, row["Frage"], history, measured=True))
        if row.get("Antwort auf Rückfrage", "").strip():
            turns.append(_ask(controller, row["Antwort auf Rückfrage"].strip(), history, measured=True))
        results.append({"testfall": row, "verlauf": turns})
    duration = time.perf_counter() - run_started

    with (run_dir / "ergebnisse.jsonl").open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    _write_overview(run_dir / "ergebnisse.csv", results)

    llm = config.llm.model
    embedding = config.embedding.models[embedding_key]
    info = {
        "beginn": started_at.isoformat(timespec="seconds"),
        "dauer_sekunden": round(duration, 1),
        "fragen": len(results),
        "testdatensatz": str(testset.relative_to(project_dir)) if testset.is_relative_to(project_dir) else str(testset),
        "sprachmodell": {"schluessel": config.llm.active, "repo": llm.repo, "revision": llm.revision},
        "embedding_modell": {"schluessel": embedding_key, "repo": embedding.repo},
        "index": str(index_dir),
        "temperatur": config.llm.temperature,
        "aufwaermfragen": list(config.tests.warmup_questions),
    }
    (run_dir / "lauf.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return run_dir


def _source_link(source: dict) -> str:
    """Adresse einer Quelle, bei PDF-Dokumenten mit Seite wie in der Oberfläche."""
    return f"{source['url']}#page={source['seite']}" if source.get("seite") else source["url"]


def _write_overview(path: Path, results: list[dict]) -> None:
    """Schreibt je Frage eine Zeile mit den Antworten des Chatbots, zum Lesen in Excel oder Numbers."""
    header = [
        "Nr.", "Frage", "Antwort", "Antwort auf Rückfrage", "Antwort danach",
        "Quelle des Chatbots", "Musterquelle", "Fallback", "Zeit (s)",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        for result in results:
            case = result["testfall"]
            measured = [turn for turn in result["verlauf"] if turn["gemessen"]]
            first, second = measured[0], (measured[1] if len(measured) > 1 else None)
            last = second or first
            writer.writerow([
                case["Nr."],
                case["Frage"],
                first["antwort"],
                case.get("Antwort auf Rückfrage", ""),
                second["antwort"] if second else "",
                _source_link(last["quellen"][0]) if last["quellen"] else "",
                case.get("Quelle", ""),
                "ja" if last["fallback"] else "nein",
                str(round(sum(turn["zeiten"]["gesamt"] for turn in measured), 1)).replace(".", ","),
            ])
