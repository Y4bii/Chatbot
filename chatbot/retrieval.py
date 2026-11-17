"""Hybride Suche nach passenden Chunks.

Zu einer Anfrage laufen eine dichte Suche über den Vektorindex und eine lexikalische
Suche über den Stichwortindex. Ihre Ergebnislisten werden mit Reciprocal Rank Fusion
zusammengeführt: Jeder Chunk erhält für jede Liste, in der er vorkommt, den Wert
``1 / (k + Rang)``, und die Summe bestimmt die gemeinsame Reihenfolge. Das Verfahren
benötigt kein Training und berücksichtigt nur Rangplätze, sodass die unterschiedlichen
Skalen von Kosinus-Ähnlichkeit und BM25 keine Rolle spielen.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from chatbot.chunking import Chunk
from chatbot.config import Config
from chatbot.embedding import Embedder
from chatbot.index import KeywordIndex, VectorIndex, load_chunks


@dataclass
class RetrievedChunk:
    """Ein gefundener Chunk mit seinen Rangplätzen in beiden Suchverfahren."""

    chunk: Chunk
    score: float
    dense_rank: int | None
    lexical_rank: int | None


def reciprocal_rank_fusion(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Führt mehrere Ranglisten zusammen.

    Args:
        rankings: Ranglisten aus Chunk-IDs, jeweils beginnend mit dem besten Treffer.
        k: Konstante, die den Einfluss der vorderen Ränge dämpft.

    Returns:
        Paare aus Chunk-ID und Wert, absteigend sortiert.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    """Sucht passende Chunks über dichte und lexikalische Suche."""

    def __init__(self, index_dir: Path, config: Config, model_key: str | None = None) -> None:
        """Lädt die Indizes und das Embedding-Modell.

        Args:
            index_dir: Ordner mit den Indizes, erzeugt durch den Befehl ``index``.
            config: Gesamte Konfiguration.
            model_key: Embedding-Modell; ohne Angabe das aktive Modell der Konfiguration.
        """
        self.config = config.retrieval
        self.model_key = model_key or config.embedding.active
        self._chunks = {chunk.id: chunk for chunk in load_chunks(index_dir / "chunks.jsonl")}
        self._keyword_index = KeywordIndex.load(index_dir / "bm25", config.keyword_index.language)
        self._vector_index = VectorIndex(index_dir / "chroma", self.model_key, config.retrieval.ef_search)
        self._embedder = Embedder(
            config.embedding.models[self.model_key], config.embedding.max_tokens, config.embedding.batch_size
        )

    def has_keyword_match(self, query: str) -> bool:
        """Findet die Stichwortsuche mindestens einen Chunk mit einem Wort der Anfrage?"""
        return bool(self._keyword_index.search(query, 1))

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Sucht die passendsten Chunks zu einer Anfrage.

        Args:
            query: Eigenständige Anfrage, bei Folgefragen bereits umformuliert.
            top_k: Anzahl der Ergebnisse; ohne Angabe der Wert aus der Konfiguration.

        Returns:
            Die gefundenen Chunks, der relevanteste zuerst.
        """
        candidates = self.config.candidates
        # Beide Suchverfahren laufen parallel.
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                self._vector_index.search, self._embedder.embed_query(query), candidates
            )
            lexical_future = executor.submit(self._keyword_index.search, query, candidates)
            dense = [chunk_id for chunk_id, _ in dense_future.result()]
            lexical = [chunk_id for chunk_id, _ in lexical_future.result()]
        fused = reciprocal_rank_fusion([dense, lexical], self.config.rrf_k)

        results = []
        for chunk_id, score in fused[: top_k or self.config.top_k]:
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[chunk_id],
                    score=score,
                    dense_rank=dense.index(chunk_id) + 1 if chunk_id in dense else None,
                    lexical_rank=lexical.index(chunk_id) + 1 if chunk_id in lexical else None,
                )
            )
        return results
