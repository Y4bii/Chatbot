"""Anbindung des Sprachmodells über einen lokalen Inferenzserver.

Das Sprachmodell läuft in einem eigenen Prozess (``mlx_lm.server``) und wird über die
verbreitete, OpenAI-kompatible Programmierschnittstelle angesprochen, die auch andere
Inferenz-Frameworks wie llama.cpp oder Ollama anbieten. Gemeint ist nur das Format der
Anfragen: Der Server läuft auf diesem Rechner, und es werden keine Daten an externe
Dienste übermittelt. Das Modell lässt sich so wechseln, ohne den übrigen Code zu ändern.
"""

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from huggingface_hub import snapshot_download

from chatbot.config import LLMConfig, LLMModelConfig


@dataclass
class Completion:
    """Antwort des Sprachmodells mit Messwerten."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def download_model(model: LLMModelConfig) -> Path:
    """Lädt ein Modell in der festgelegten Revision herunter, sofern noch nicht vorhanden.

    Returns:
        Lokaler Ordner des Modells im Zwischenspeicher von Hugging Face.
    """
    return Path(snapshot_download(repo_id=model.repo, revision=model.revision))


def server_command(config: LLMConfig, model_dir: Path) -> list[str]:
    """Befehl zum Starten des Inferenzservers mit den Einstellungen aus der Konfiguration."""
    executable = shutil.which("mlx_lm.server") or str(Path.home() / ".local" / "bin" / "mlx_lm.server")
    address = urlsplit(config.server_url)
    return [
        executable,
        "--model", str(model_dir),
        "--host", address.hostname or "127.0.0.1",
        "--port", str(address.port or 8080),
        "--temp", str(config.temperature),
        "--max-tokens", str(config.max_tokens),
        "--chat-template-args", json.dumps(config.chat_template_args),
    ]


class LLMClient:
    """Sendet Chat-Anfragen an den lokalen Inferenzserver."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = httpx.Client(base_url=config.server_url, timeout=config.timeout)

    def chat(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> Completion:
        """Erzeugt eine Antwort auf einen Gesprächsverlauf.

        Args:
            messages: Nachrichten mit ``role`` (``system``, ``user``, ``assistant``) und ``content``.
            max_tokens: Abweichende Obergrenze der erzeugten Token.

        Returns:
            Die Antwort mit Tokenzahlen und Dauer.
        """
        started = time.perf_counter()
        response = self._client.post(
            "/chat/completions",
            json={
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "chat_template_kwargs": self.config.chat_template_args,
            },
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"].strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            seconds=time.perf_counter() - started,
        )
