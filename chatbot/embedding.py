"""Embedding-Modelle für die dichte Suche.

Ein Embedding-Modell wandelt Chunks und Anfragen in Vektoren um, deren Ähnlichkeit die
Kosinus-Ähnlichkeit angibt. Die Modelle erwarten unterschiedliche Präfixe vor Anfragen
und Chunks. Diese stehen in der Konfiguration, sodass sich ein Modell ohne Änderung am
Code austauschen lässt.
"""

import threading

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from chatbot.config import EmbeddingModelConfig


class Embedder:
    """Kapselt ein Embedding-Modell einschließlich seiner Präfixe."""

    def __init__(self, config: EmbeddingModelConfig, max_tokens: int, batch_size: int) -> None:
        """Lädt das Modell. Beim ersten Aufruf wird es von Hugging Face heruntergeladen.

        Args:
            config: Modell und Präfixe.
            max_tokens: Maximale Eingabelänge in Token.
            batch_size: Anzahl der Texte, die gemeinsam eingebettet werden.
        """
        self.config = config
        self.batch_size = batch_size
        # Auf Apple Silicon rechnet das Modell auf der Grafikeinheit (Metal Performance Shaders).
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(
            config.repo, device=self.device, trust_remote_code=config.trust_remote_code
        )
        self.model.max_seq_length = max_tokens
        # Das Modell ist nicht threadsicher: Nomic hält einen Zwischenspeicher für die
        # Positionskodierung, der bei gleichzeitigen Anfragen inkonsistent wird.
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Bettet Chunks ein. Die Vektoren sind auf Länge 1 normiert."""
        return self.model.encode(
            [self.config.document_prefix + text for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Bettet eine Anfrage ein. Der Vektor ist auf Länge 1 normiert.

        Gleichzeitige Anfragen werden nacheinander verarbeitet.
        """
        with self._lock:
            return self.model.encode(
                self.config.query_prefix + text, normalize_embeddings=True, convert_to_numpy=True
            )

    def count_document_tokens(self, text: str) -> int:
        """Tokenzahl eines Chunks mit Präfix und Sondertoken, wie sie das Modell verarbeitet."""
        return len(self.model.tokenizer(self.config.document_prefix + text)["input_ids"])
