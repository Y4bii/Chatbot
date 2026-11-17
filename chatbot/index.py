"""Vektor- und Stichwortindex der Chunks.

Der Vektorindex liegt in einer lokalen Chroma-Datenbank, die für die
Nächste-Nachbarn-Suche einen HNSW-Graphen verwendet. Für jedes Embedding-Modell gibt
es eine eigene Sammlung, damit sich die Modelle vergleichen lassen. Der Stichwortindex
verwendet BM25 mit deutschem Stemming und deutschen Stoppwörtern.
"""

import json
from dataclasses import asdict
from pathlib import Path

import bm25s
import chromadb
import numpy as np
import Stemmer
from bm25s import stopwords as bm25_stopwords
from chromadb.config import Settings

from chatbot.chunking import Chunk

# Chroma nimmt nur eine begrenzte Zahl an Einträgen je Aufruf an.
_CHROMA_BATCH = 2000


def load_chunks(path: Path) -> list[Chunk]:
    """Liest Chunks aus einer JSONL-Datei."""
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_chunks(path: Path, chunks: list[Chunk]) -> None:
    """Schreibt Chunks in eine JSONL-Datei."""
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


class VectorIndex:
    """Vektorindex für ein Embedding-Modell in einer lokalen Chroma-Datenbank."""

    def __init__(self, path: Path, model_key: str, ef_search: int) -> None:
        """
        Args:
            path: Ordner der Datenbank.
            model_key: Schlüssel des Embedding-Modells, bestimmt den Namen der Sammlung.
            ef_search: Suchtiefe im HNSW-Graphen.
        """
        # Die Telemetrie ist abgeschaltet, damit keine Daten an externe Dienste gehen.
        self._client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self._name = f"chunks_{model_key}"
        self._ef_search = ef_search
        self._collection = None

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Legt die Sammlung neu an und füllt sie mit den Chunks und ihren Vektoren."""
        if self._name in [collection.name for collection in self._client.list_collections()]:
            self._client.delete_collection(self._name)
        collection = self._client.create_collection(self._name, metadata={"hnsw:space": "cosine"})
        collection.modify(configuration={"hnsw": {"ef_search": self._ef_search}})
        for start in range(0, len(chunks), _CHROMA_BATCH):
            batch = chunks[start : start + _CHROMA_BATCH]
            collection.add(
                ids=[chunk.id for chunk in batch],
                embeddings=embeddings[start : start + len(batch)].tolist(),
                documents=[chunk.text for chunk in batch],
                metadatas=[
                    {"url": chunk.url, "title": chunk.title, "kind": chunk.kind, "breadcrumb": json.dumps(chunk.breadcrumb, ensure_ascii=False)}
                    for chunk in batch
                ],
            )

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Sucht die ``k`` ähnlichsten Chunks.

        Returns:
            Paare aus Chunk-ID und Kosinus-Ähnlichkeit, absteigend sortiert.
        """
        result = self._get_collection().query(query_embeddings=[query_embedding.tolist()], n_results=k)
        return [(chunk_id, 1.0 - distance) for chunk_id, distance in zip(result["ids"][0], result["distances"][0])]

    def _get_collection(self):
        """Die Sammlung, mit der konfigurierten Suchtiefe.

        Bei der Voreinstellung von Chroma lieferte die Näherungssuche für manche Anfragen
        keinen der tatsächlich ähnlichsten Chunks, etwa Modulhandbücher (Ähnlichkeit 0,41)
        statt der Seite „Vorlesungszeiten“ (0,61). Ältere Indizes werden beim ersten Zugriff
        auf die konfigurierte Suchtiefe umgestellt.
        """
        if self._collection is None:
            collection = self._client.get_collection(self._name)
            if collection.configuration_json.get("hnsw", {}).get("ef_search") != self._ef_search:
                collection.modify(configuration={"hnsw": {"ef_search": self._ef_search}})
                collection = self._client.get_collection(self._name)
            self._collection = collection
        return self._collection


class KeywordIndex:
    """Stichwortindex mit BM25."""

    def __init__(self, language: str) -> None:
        """
        Args:
            language: Sprache für Stemming und Stoppwörter, z. B. ``german``.
        """
        self._stemmer = Stemmer.Stemmer(language)
        self._stopwords = list(getattr(bm25_stopwords, f"STOPWORDS_{language.upper()}"))
        self._retriever: bm25s.BM25 | None = None
        self._ids: list[str] = []

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        return bm25s.tokenize(
            texts, stopwords=self._stopwords, stemmer=self._stemmer, return_ids=False, show_progress=False
        )

    def build(self, chunks: list[Chunk]) -> None:
        """Baut den Index aus den Texten der Chunks einschließlich Titel und Überschriften."""
        self._ids = [chunk.id for chunk in chunks]
        self._retriever = bm25s.BM25()
        self._retriever.index(self._tokenize([chunk.text for chunk in chunks]), show_progress=False)

    def save(self, path: Path) -> None:
        """Speichert den Index und die Zuordnung zu den Chunk-IDs."""
        self._retriever.save(str(path))
        (path / "chunk_ids.json").write_text(json.dumps(self._ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, language: str) -> "KeywordIndex":
        """Lädt einen gespeicherten Index."""
        index = cls(language)
        index._retriever = bm25s.BM25.load(str(path))
        index._ids = json.loads((path / "chunk_ids.json").read_text(encoding="utf-8"))
        return index

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Sucht die ``k`` am besten passenden Chunks.

        Returns:
            Paare aus Chunk-ID und BM25-Wert, absteigend sortiert. Chunks ohne
            übereinstimmendes Wort werden nicht zurückgegeben.
        """
        tokens = self._tokenize([query])
        if not tokens[0]:
            return []
        documents, scores = self._retriever.retrieve(tokens, k=min(k, len(self._ids)), show_progress=False)
        return [(self._ids[doc], float(score)) for doc, score in zip(documents[0], scores[0]) if score > 0]
