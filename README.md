# Studien-Assistent

Prototyp eines lokal betriebenen Chatbots auf Basis von Retrieval-Augmented Generation (RAG). Er beantwortet Fragen zu Studium und Studienorganisation auf Grundlage der öffentlich zugänglichen Inhalte im Bereich `/studium` der Webseite der Universität zu Lübeck.

Der Prototyp entstand im Rahmen einer Bachelorarbeit am Institut für Multimediale und Interaktive Systeme (IMIS) der Universität zu Lübeck. Er dient als Machbarkeitsnachweis und ist kein Produktivsystem.

## Funktionsweise

Das System besteht aus zwei getrennten Abläufen:

1. **Indexierung:** Ein Crawler ruft die Seiten unterhalb von `/studium` und die dort verlinkten PDF-Dokumente ab. Die Inhalte werden aufbereitet, in Abschnitte (Chunks) zerlegt und in einem Vektor- und einem Stichwortindex abgelegt.
2. **Beantwortung einer Anfrage:** Zu jeder Frage werden passende Chunks über eine hybride Suche gefunden. Das Sprachmodell formuliert daraus eine Antwort mit Quellenangaben.

Alle Komponenten laufen lokal. Bei der Beantwortung einer Anfrage werden keine externen Dienste aufgerufen, und Anfragen werden nicht gespeichert.

## Voraussetzungen

- Mac mit Apple Silicon, entwickelt und getestet auf einem MacBook Pro mit M1 Pro und 16 GB Arbeitsspeicher
- [uv](https://docs.astral.sh/uv/) zur Verwaltung von Python und den Abhängigkeiten

## Einrichtung

```bash
uv sync
```

Der Befehl installiert Python 3.12 und alle Abhängigkeiten in die virtuelle Umgebung `.venv` im Projektordner.

## Kurzüberblick

Abgerufene Inhalte und Indizes sind nicht Teil des Repositorys. Für den ersten Start werden die Schritte in dieser Reihenfolge ausgeführt:

1. `crawl`: Webseite abrufen
2. `prepare`: Inhalte aufbereiten und in Chunks zerlegen
3. `index`: Indizes aufbauen
4. `serve`: Sprachmodell starten (eigenes Terminal, läuft dauerhaft)
5. `web` oder `ask`: Fragen stellen

`<datum>` steht jeweils für den Ordnernamen, den der vorherige Schritt angelegt hat. Die einzelnen Befehle sind im Folgenden beschrieben.

## Verwendung

### Webseite abrufen

```bash
uv run python main.py crawl
```

Der Crawler ruft ausgehend von der Startadresse alle Seiten im Kernbereich (`/studium/`) ab. Seiten außerhalb des Kernbereichs verfolgt er bis zu einer festgelegten Zahl an Klicks. Dazu kommen alle verlinkten PDF-Dokumente der Hochschule. Er beachtet die `robots.txt` und wartet zwischen zwei Abrufen.

Der Abruf wird unter `data/raw/<datum>/` gespeichert:

- `pages/`: abgerufene Seiten
- `documents/`: abgerufene PDF-Dokumente
- `manifest.jsonl`: ein Eintrag je Abruf mit Adresse, Titel, Linktiefe und Zeitpunkt
- `crawl_info.json`: Einstellungen und Zusammenfassung des Abrufs

Ein unterbrochener Abruf lässt sich fortsetzen, ohne bereits abgerufene Seiten erneut zu laden:

```bash
uv run python main.py crawl --resume --output data/raw/<datum>
```

Für einen kurzen Testlauf lässt sich die Zahl der Abrufe begrenzen:

```bash
uv run python main.py crawl --max-requests 10 --output data/raw/test
```

### Aufbereiten und in Chunks zerlegen

```bash
uv run python main.py prepare --raw data/raw/<datum>
```

Aus jeder Seite wird der Inhaltsbereich übernommen, aus jedem PDF-Dokument der Text. Die Inhalte werden entlang der Überschriften in Chunks von höchstens 480 Token zerlegt. Das Ergebnis liegt unter `data/processed/<datum>/` (`documents.jsonl`, `chunks.jsonl`, `prepare_info.json`).

### Indizes aufbauen

```bash
uv run python main.py index --processed data/processed/<datum>
```

Baut den Stichwortindex (BM25) und für jedes Embedding-Modell aus der Konfiguration einen Vektorindex in einer lokalen Chroma-Datenbank. Die Embedding-Modelle werden beim ersten Aufruf von Hugging Face geladen. Das Ergebnis liegt unter `data/index/<datum>/`, die gemessenen Zeiten stehen in `index_info.json`.

### Suchen

```bash
uv run python main.py search "Wann ist die Rückmeldung?" --index data/index/<datum>
```

Führt die hybride Suche aus: dichte Suche und BM25 parallel, zusammengeführt mit Reciprocal Rank Fusion. Angezeigt werden die gefundenen Chunks mit ihren Rangplätzen in beiden Suchverfahren. Mit `--model e5` lässt sich das Embedding-Modell wechseln.

### Sprachmodell starten

Das Sprachmodell läuft in einem eigenen Inferenzserver auf Basis von [mlx-lm](https://github.com/ml-explore/mlx-lm). `mlx-lm` wird getrennt vom Projekt installiert, da es andere Bibliotheksversionen benötigt als die Embedding-Modelle:

```bash
uv tool install "mlx-lm==0.32.2"
uv run python main.py serve
```

Der Befehl lädt das aktive Modell aus der Konfiguration in der festgelegten Revision herunter, sofern es noch nicht vorhanden ist. Danach startet er den Server unter `http://127.0.0.1:8080`, nur auf diesem Rechner erreichbar. Mit `--model eurollm` oder `--model qwen3` lässt sich ein anderes Modell starten.

### Fragen stellen

```bash
uv run python main.py ask --index data/index/<datum>
uv run python main.py ask "Wann ist die Rückmeldung?" --index data/index/<datum>
```

Startet ein Gespräch im Terminal oder beantwortet eine einzelne Frage. Der Inferenzserver muss laufen. Zu jeder Antwort werden die Quellen, die für die Suche verwendete Anfrage sowie Zeiten und Tokenzahlen ausgegeben. Der Gesprächsverlauf wird nicht gespeichert.

### Testseite mit Chatfenster

```bash
uv run python main.py web --index data/index/<datum>
```

Startet die Testseite unter `http://127.0.0.1:8000`. Der Inferenzserver muss laufen. Das Chatfenster öffnet sich über die Schaltfläche unten rechts. Der Gesprächsverlauf liegt nur im Browser und bleibt für die Dauer der Sitzung erhalten. Die Übergabe an den Studierendenservice ist nur nachgebildet, es werden keine Daten versendet oder gespeichert.

### Testlauf

```bash
uv run python main.py test --index data/index/<datum>
```

Stellt alle Fragen aus `technische-tests/testdatensatz.csv` und speichert Antworten, Quellen, gefundene Chunks und Zeiten unter `technische-tests/laeufe/`. Der Inferenzserver muss laufen; die Testseite sollte während des Laufs beendet sein, damit die Zeiten nicht verfälscht werden. Mit `--only 2 33` werden nur einzelne Nummern gestellt.

### Bewertung eines Testlaufs

```bash
uv run python main.py grade --run technische-tests/laeufe/<lauf>
```

Lässt jeden Testfall von einem externen Bewerter-LLM (Claude Opus 5.5, kostenpflichtig, siehe `[tests.judge]` in der Konfiguration) beurteilen und berechnet eigene Kennzahlen wie Treffer der Musterquelle, Fallback und Antwortzeiten. Übermittelt werden nur Testfragen, Auszüge der öffentlichen Webseite und Antworten des Chatbots. Mit `--dry-run` wird nichts gesendet; angezeigt werden der erste Prompt und die eigenen Kennzahlen.

Der Zugang zur Claude API erfolgt über die Anmeldung mit `ant auth login` oder über die Umgebungsvariable `ANTHROPIC_API_KEY`.

## Konfiguration

Alle Angaben, die sich auf die Universität zu Lübeck beziehen, sind in `config/uzl.toml` gebündelt, etwa:

- die Startadresse des Crawlers
- der Name der Universität
- die Kontaktdaten des Studierendenservice
- die verwendeten Modelle

Für den Einsatz an einer anderen Hochschule wird diese Datei angepasst und die Indexierung neu ausgeführt.

## Projektstruktur

| Pfad | Inhalt |
|---|---|
| `main.py` | Einstiegspunkt mit den Befehlen der Kommandozeile |
| `config/` | Konfiguration der Hochschule und der Modelle |
| `chatbot/config.py` | Laden der Konfiguration |
| `chatbot/crawler.py` | Abruf der Seiten und PDF-Dokumente |
| `chatbot/robots.py` | Auswertung der `robots.txt` nach RFC 9309 |
| `chatbot/preprocessing.py` | Aufbereitung der Seiten und PDF-Dokumente |
| `chatbot/derived.py` | Semesterübersichten aus den Modul-Steckbriefen |
| `chatbot/chunking.py` | Zerlegung in Chunks |
| `chatbot/embedding.py` | Embedding-Modelle für die dichte Suche |
| `chatbot/index.py` | Vektorindex (Chroma) und Stichwortindex (BM25) |
| `chatbot/retrieval.py` | Hybride Suche mit Reciprocal Rank Fusion |
| `chatbot/llm.py` | Anbindung des lokalen Inferenzservers |
| `chatbot/controller.py` | Steuerung: Umformulierung, Suche, Prompt, Quellen, Fallback |
| `chatbot/api.py` | Webschnittstelle zwischen Oberfläche und Steuerung |
| `chatbot/testrun.py` | Testlauf über den Testdatensatz |
| `chatbot/grading.py` | Bewertung eines Testlaufs mit Bewerter-LLM und eigenen Kennzahlen |
| `web/` | Testseite und Chatfenster (HTML, CSS, JavaScript) |
| `prompts/` | Systemprompt, Aufbau des Kontexts und Prompt zur Umformulierung von Folgefragen |
| `technische-tests/` | Testdatensatz, Bewerter-Prompt und der bewertete Testlauf mit Ergebnissen |
| `data/` | Abgerufene Inhalte und Indizes (nicht im Repository) |
