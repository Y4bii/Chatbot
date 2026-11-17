"""Webschnittstelle zwischen Benutzeroberfläche und Steuerung.

Der Server liefert die Testseite mit dem Chatfenster aus und nimmt Anfragen entgegen.
Er ist zustandslos: Der Gesprächsverlauf liegt nur im Browser und wird mit jeder Anfrage
übermittelt. Anfragen, Verläufe und Antworten werden nicht gespeichert oder protokolliert.
Die Übergabe an den Studierendenservice ist nur im Browser nachgebildet und erreicht den
Server nicht.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chatbot.config import Config
from chatbot.controller import Controller

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class Message(BaseModel):
    """Eine Nachricht des bisherigen Gesprächs."""

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    """Anfrage der Oberfläche mit dem bisherigen Gesprächsverlauf."""

    question: str = Field(min_length=1, max_length=1000)
    history: list[Message] = Field(default_factory=list, max_length=50)


def _source_url(url: str, page: int | None) -> str:
    """Adresse einer Quelle; bei PDF-Dokumenten mit Sprung auf die Seite (``#page=N``)."""
    return f"{url}#page={page}" if page else url


def create_app(controller: Controller, config: Config) -> FastAPI:
    """Erzeugt die Webanwendung.

    Args:
        controller: Steuerung, die die Anfragen beantwortet.
        config: Gesamte Konfiguration, aus der Name und Kontaktdaten stammen.
    """
    # Die automatisch erzeugte Dokumentation lädt Skripte von externen Servern und entfällt daher.
    app = FastAPI(title=config.chatbot.name, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def no_cache(request, call_next):
        """Verhindert, dass der Browser veraltete Fassungen der Oberfläche aus dem Zwischenspeicher lädt."""
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/config")
    def chatbot_config() -> dict:
        """Angaben, die die Oberfläche aus der Konfiguration übernimmt."""
        service = config.student_service
        return {
            "name": config.chatbot.name,
            "subtitle": config.chatbot.subtitle,
            "university": config.university_name,
            "contact": {"name": service.name, "email": service.email, "phone": service.phone, "url": service.url},
        }

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict:
        """Beantwortet eine Anfrage und gibt Antwort und Quellen zurück."""
        answer = controller.answer(request.question, [message.model_dump() for message in request.history])
        return {
            "text": answer.text,
            "sources": [
                {"title": source.title, "url": _source_url(source.url, source.page), "kind": source.kind}
                for source in answer.sources
            ],
            "fallback": answer.fallback,
            # Dient bei einer Übergabe als Betreff, der das Anliegen ohne Verlauf zusammenfasst (4.5).
            # Bei Begrüßung, Dank oder Verabschiedung gibt es kein Anliegen und damit keinen Betreff.
            "search_query": None if answer.small_talk else answer.search_query,
        }

    @app.get("/")
    def index() -> FileResponse:
        """Testseite mit dem Chatfenster."""
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
