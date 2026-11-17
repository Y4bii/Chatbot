"""Kommandozeile des Studien-Assistenten.

Beispiele::

    uv run python main.py crawl
    uv run python main.py crawl --max-requests 10 --output data/raw/test
    uv run python main.py prepare --raw data/raw/2026-11-15
    uv run python main.py index --processed data/processed/2026-11-15
    uv run python main.py search "Wann ist die Rückmeldung?" --index data/index/2026-11-15
    uv run python main.py test --index data/index/2026-11-15 --only 2 33
    uv run python main.py grade --run technische-tests/laeufe/<lauf> --dry-run
"""

import argparse
import gc
import json
import logging
import statistics
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

from chatbot.chunking import Chunker
from chatbot.config import DEFAULT_CONFIG_PATH, load_config
from chatbot.crawler import Crawler
from chatbot.preprocessing import Preprocessor

PROJECT_DIR = Path(__file__).resolve().parent
# Chunks unterhalb dieser Tokenzahl gelten in der Auswertung als kurz.
SHORT_CHUNK_TOKENS = 64


def _write_jsonl(path: Path, items: list) -> None:
    """Schreibt Datenklassen zeilenweise als JSON."""
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def crawl(args: argparse.Namespace) -> None:
    """Ruft die Webseite ab und gibt eine Zusammenfassung aus."""
    config = load_config(args.config)
    if args.resume and not args.output:
        raise SystemExit("Zum Fortsetzen bitte den Ordner des Abrufs mit --output angeben.")
    output_dir = args.output or PROJECT_DIR / "data" / "raw" / date.today().isoformat()
    stats = Crawler(config.crawler, output_dir).run(
        start_url=args.start_url, max_requests=args.max_requests, resume=args.resume
    )
    print(f"\nAbruf gespeichert unter {output_dir}")
    print(f"Abrufe: {stats.requests}, Seiten: {stats.pages}, PDFs: {stats.documents}, Fehler: {stats.errors}")
    for reason, count in sorted(stats.skipped.items()):
        print(f"Übersprungen ({reason}): {count}")
    print(f"Office-Dokumente (nicht aufgenommen): {len(stats.office_links)}")


def prepare(args: argparse.Namespace) -> None:
    """Bereitet einen Abruf auf, zerlegt ihn in Chunks und gibt eine Zusammenfassung aus."""
    config = load_config(args.config)
    output_dir = args.output or PROJECT_DIR / "data" / "processed" / args.raw.name
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = Preprocessor(config.preprocessing, config.crawler, config.university_name)
    documents = preprocessor.run(args.raw)
    chunker = Chunker(config.chunking)
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]

    _write_jsonl(output_dir / "documents.jsonl", documents)
    _write_jsonl(output_dir / "chunks.jsonl", chunks)
    tokens = [chunk.token_count for chunk in chunks]
    info = {
        "raw_dir": str(args.raw),
        "preprocessing": asdict(preprocessor.stats),
        "chunks": {
            "count": len(chunks),
            "from_pages": sum(chunk.kind == "page" for chunk in chunks),
            "from_documents": sum(chunk.kind == "document" for chunk in chunks),
            "tokens_min": min(tokens, default=0),
            "tokens_median": statistics.median(tokens) if tokens else 0,
            "tokens_max": max(tokens, default=0),
            "short": sum(t < SHORT_CHUNK_TOKENS for t in tokens),
            "over_limit": sum(t > config.chunking.max_tokens for t in tokens),
        },
        "config": {"preprocessing": asdict(config.preprocessing), "chunking": asdict(config.chunking)},
    }
    (output_dir / "prepare_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nAufbereitung gespeichert unter {output_dir}")
    print(json.dumps({"preprocessing": info["preprocessing"], "chunks": info["chunks"]}, ensure_ascii=False, indent=2))


def build_index(args: argparse.Namespace) -> None:
    """Baut den Stichwortindex und je Embedding-Modell einen Vektorindex und misst die Dauer."""
    # Erst hier importiert, da PyTorch und die Modelle nur für diesen Befehl benötigt werden
    from chatbot.embedding import Embedder
    from chatbot.index import KeywordIndex, VectorIndex, load_chunks, save_chunks

    config = load_config(args.config)
    chunks = load_chunks(args.processed / "chunks.jsonl")
    output_dir = args.output or PROJECT_DIR / "data" / "index" / args.processed.name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_chunks(output_dir / "chunks.jsonl", chunks)
    info: dict = {"processed_dir": str(args.processed), "chunks": len(chunks), "models": {}}

    started = time.perf_counter()
    keyword_index = KeywordIndex(config.keyword_index.language)
    keyword_index.build(chunks)
    (output_dir / "bm25").mkdir(exist_ok=True)
    keyword_index.save(output_dir / "bm25")
    info["keyword_index_seconds"] = round(time.perf_counter() - started, 2)

    for key in args.models or list(config.embedding.models):
        model_config = config.embedding.models[key]
        started = time.perf_counter()
        embedder = Embedder(model_config, config.embedding.max_tokens, config.embedding.batch_size)
        loaded = time.perf_counter()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_documents(texts)
        embedded = time.perf_counter()
        VectorIndex(output_dir / "chroma", key, config.retrieval.ef_search).build(chunks, embeddings)
        indexed = time.perf_counter()
        # Kontrolle mit dem Tokenizer des Modells: Chunks über der Grenze würden abgeschnitten.
        token_counts = [embedder.count_document_tokens(text) for text in texts]
        info["models"][key] = {
            "repo": model_config.repo,
            "device": embedder.device,
            "dimensions": int(embeddings.shape[1]),
            "load_seconds": round(loaded - started, 2),
            "embed_seconds": round(embedded - loaded, 2),
            "index_seconds": round(indexed - embedded, 2),
            "max_tokens_with_prefix": max(token_counts, default=0),
            "truncated_chunks": sum(count > config.embedding.max_tokens for count in token_counts),
        }
        print(f"{key}: {json.dumps(info['models'][key], ensure_ascii=False)}")
        # Modell freigeben, bevor das nächste geladen wird
        del embedder, embeddings
        gc.collect()

    (output_dir / "index_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nIndizes gespeichert unter {output_dir}")


def search(args: argparse.Namespace) -> None:
    """Sucht passende Chunks zu einer Frage und zeigt die Rangplätze beider Suchverfahren."""
    from chatbot.retrieval import HybridRetriever

    config = load_config(args.config)
    retriever = HybridRetriever(args.index, config, args.model)
    started = time.perf_counter()
    results = retriever.search(args.question, args.top_k)
    print(f"\nModell: {retriever.model_key}, Dauer: {time.perf_counter() - started:.2f} s\n")
    for position, result in enumerate(results, start=1):
        dense = result.dense_rank or "–"
        lexical = result.lexical_rank or "–"
        print(f"{position}. [{result.score:.4f} | dicht {dense} | BM25 {lexical}] {result.chunk.text.splitlines()[0]}")
        print(f"   {result.chunk.url}")


def ask(args: argparse.Namespace) -> None:
    """Gespräch mit dem Chatbot im Terminal. Der Verlauf liegt nur im Arbeitsspeicher."""
    from chatbot.controller import Controller
    from chatbot.llm import LLMClient
    from chatbot.retrieval import HybridRetriever

    config = load_config(args.config)
    controller = Controller(config, HybridRetriever(args.index, config, args.embedding), LLMClient(config.llm))
    history: list[dict[str, str]] = []
    questions = [args.question] if args.question else None
    print(f"{config.chatbot.name} (KI) – leere Eingabe beendet das Gespräch.")
    while True:
        question = questions.pop(0) if questions else (None if questions is not None else input("\nDu: ").strip())
        if not question:
            break
        answer = controller.answer(question, history)
        print(f"\n{config.chatbot.name}: {answer.text}")
        for source in answer.sources:
            print(f"  Quelle: {source.title} · {' › '.join(source.breadcrumb)}\n          {source.url}")
        details = f"Suchanfrage: {answer.search_query} | Zeiten: {answer.timings} | Token: {answer.tokens}"
        if answer.fallback:
            details += " | Fallback"
        if not answer.citations_from_model:
            details += " | keine Chunk-Nummern vom Modell"
        print(f"  [{details}]")
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer.text}]


def web(args: argparse.Namespace) -> None:
    """Startet die Testseite mit dem Chatfenster unter http://127.0.0.1:<port>."""
    import uvicorn

    from chatbot.api import create_app
    from chatbot.controller import Controller
    from chatbot.llm import LLMClient
    from chatbot.retrieval import HybridRetriever

    config = load_config(args.config)
    controller = Controller(config, HybridRetriever(args.index, config, args.embedding), LLMClient(config.llm))
    # Ohne Zugriffsprotokoll, damit auch keine Anfragezeilen festgehalten werden
    uvicorn.run(create_app(controller, config), host="127.0.0.1", port=args.port, access_log=False)


def test(args: argparse.Namespace) -> None:
    """Stellt alle Fragen des Testdatensatzes und speichert die Antworten als Testlauf."""
    from chatbot.controller import Controller
    from chatbot.llm import LLMClient
    from chatbot.retrieval import HybridRetriever
    from chatbot.testrun import run_testset

    config = load_config(args.config)
    embedding_key = args.embedding or config.embedding.active
    controller = Controller(config, HybridRetriever(args.index, config, embedding_key), LLMClient(config.llm))
    run_dir = run_testset(
        controller,
        config,
        testset=args.testset or PROJECT_DIR / config.tests.testset,
        output_dir=args.output or PROJECT_DIR / config.tests.runs_dir,
        index_dir=args.index,
        embedding_key=embedding_key,
        project_dir=PROJECT_DIR,
        config_path=args.config,
        only=set(args.only) if args.only else None,
    )
    print(f"Testlauf gespeichert unter {run_dir}")


def grade(args: argparse.Namespace) -> None:
    """Bewertet einen Testlauf mit dem Bewerter-LLM und eigenen Kennzahlen."""
    from chatbot.grading import grade_run

    config = load_config(args.config)
    summary = grade_run(
        args.run,
        PROJECT_DIR / config.tests.judge.prompt,
        config.tests.judge,
        only=set(args.only) if args.only else None,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def serve(args: argparse.Namespace) -> None:
    """Lädt das gewählte Sprachmodell, falls nötig, und startet den lokalen Inferenzserver."""
    import os

    from chatbot.llm import download_model, server_command

    config = load_config(args.config)
    key = args.model or config.llm.active
    model_dir = download_model(config.llm.models[key])
    command = server_command(config.llm, model_dir)
    print(f"Starte Inferenzserver für {key}: {' '.join(command)}", flush=True)
    # Der Prozess wird durch den Server ersetzt, damit er sich direkt beenden lässt.
    os.execv(command[0], command)


def main() -> None:
    """Wertet die Argumente aus und führt den gewählten Befehl aus."""
    parser = argparse.ArgumentParser(description="Studien-Assistent")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Konfigurationsdatei")
    commands = parser.add_subparsers(dest="command", required=True)

    crawl_parser = commands.add_parser("crawl", help="Webseite abrufen und mit Datum speichern")
    crawl_parser.add_argument("--output", type=Path, help="Zielordner (Standard: data/raw/<datum>)")
    crawl_parser.add_argument("--start-url", help="abweichende Startadresse, z. B. für Tests")
    crawl_parser.add_argument("--max-requests", type=int, help="Obergrenze der Abrufe, z. B. für Tests")
    crawl_parser.add_argument("--resume", action="store_true", help="unterbrochenen Abruf in --output fortsetzen")
    crawl_parser.set_defaults(handler=crawl)

    prepare_parser = commands.add_parser("prepare", help="Abruf aufbereiten und in Chunks zerlegen")
    prepare_parser.add_argument("--raw", type=Path, required=True, help="Ordner des Abrufs")
    prepare_parser.add_argument("--output", type=Path, help="Zielordner (Standard: data/processed/<abruf>)")
    prepare_parser.set_defaults(handler=prepare)

    index_parser = commands.add_parser("index", help="Stichwort- und Vektorindizes aufbauen")
    index_parser.add_argument("--processed", type=Path, required=True, help="Ordner der Aufbereitung")
    index_parser.add_argument("--output", type=Path, help="Zielordner (Standard: data/index/<aufbereitung>)")
    index_parser.add_argument("--models", nargs="+", help="Schlüssel der Embedding-Modelle (Standard: alle)")
    index_parser.set_defaults(handler=build_index)

    search_parser = commands.add_parser("search", help="Passende Chunks zu einer Frage suchen")
    search_parser.add_argument("question", help="Frage")
    search_parser.add_argument("--index", type=Path, required=True, help="Ordner der Indizes")
    search_parser.add_argument("--model", help="Embedding-Modell (Standard: aktives Modell)")
    search_parser.add_argument("--top-k", type=int, help="Anzahl der Ergebnisse")
    search_parser.set_defaults(handler=search)

    serve_parser = commands.add_parser("serve", help="Lokalen Inferenzserver für das Sprachmodell starten")
    serve_parser.add_argument("--model", help="Schlüssel des Sprachmodells (Standard: aktives Modell)")
    serve_parser.set_defaults(handler=serve)

    ask_parser = commands.add_parser("ask", help="Fragen an den Chatbot stellen (Inferenzserver muss laufen)")
    ask_parser.add_argument("question", nargs="?", help="Einzelne Frage; ohne Angabe startet ein Gespräch")
    ask_parser.add_argument("--index", type=Path, required=True, help="Ordner der Indizes")
    ask_parser.add_argument("--embedding", help="Embedding-Modell (Standard: aktives Modell)")
    ask_parser.set_defaults(handler=ask)

    web_parser = commands.add_parser("web", help="Testseite mit Chatfenster starten (Inferenzserver muss laufen)")
    web_parser.add_argument("--index", type=Path, required=True, help="Ordner der Indizes")
    web_parser.add_argument("--embedding", help="Embedding-Modell (Standard: aktives Modell)")
    web_parser.add_argument("--port", type=int, default=8000, help="Port der Testseite (Standard: 8000)")
    web_parser.set_defaults(handler=web)

    test_parser = commands.add_parser("test", help="Testdatensatz durchlaufen (Inferenzserver muss laufen)")
    test_parser.add_argument("--index", type=Path, required=True, help="Ordner der Indizes")
    test_parser.add_argument("--embedding", help="Embedding-Modell (Standard: aktives Modell)")
    test_parser.add_argument("--testset", type=Path, help="Testdatensatz (Standard: aus der Konfiguration)")
    test_parser.add_argument("--only", type=int, nargs="+", help="nur diese Nummern, z. B. zum Ausprobieren")
    test_parser.add_argument("--output", type=Path, help="Ordner für den Lauf (Standard: aus der Konfiguration)")
    test_parser.set_defaults(handler=test)

    grade_parser = commands.add_parser("grade", help="Testlauf bewerten (Bewerter-LLM und eigene Kennzahlen)")
    grade_parser.add_argument("--run", type=Path, required=True, help="Ordner des Testlaufs")
    grade_parser.add_argument("--only", type=int, nargs="+", help="nur diese Nummern bewerten")
    grade_parser.add_argument(
        "--dry-run", action="store_true", help="ohne Anfrage an den Bewerter: ersten Prompt und eigene Kennzahlen zeigen"
    )
    grade_parser.set_defaults(handler=grade)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.handler(args)


if __name__ == "__main__":
    main()
