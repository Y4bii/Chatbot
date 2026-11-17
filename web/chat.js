/**
 * Verhalten des Chatfensters.
 *
 * Der Gesprächsverlauf liegt nur im Browser (sessionStorage) und bleibt für die Dauer
 * der Sitzung erhalten, auch wenn das Fenster geschlossen wird (Abschnitt 4.6). Er wird
 * mit jeder Anfrage an den Server übermittelt, der selbst nichts speichert. Die Übergabe
 * an den Studierendenservice ist nur nachgebildet und verlässt den Browser nicht.
 */
(() => {
  "use strict";

  const STORAGE_KEY = "studien-assistent-verlauf";
  // Anzahl der Nachrichten, die höchstens als Verlauf an den Server gehen
  const MAX_HISTORY = 20;

  const byId = (id) => document.getElementById(id);
  const launcher = byId("chat-launcher");
  const panel = byId("chat-panel");
  const log = byId("chat-log");
  const status = byId("chat-status");
  const chatForm = byId("chat-form");
  const input = byId("chat-input");
  const sendButton = chatForm.querySelector(".send-button");
  const contactView = byId("contact-view");
  const handoverView = byId("handover-view");

  let settings = {
    name: "Studien-Assistent",
    university: "Universität",
    contact: { name: "Studierendenservice", email: "", phone: "", url: "" },
  };
  let messages = loadMessages();
  let busy = false;

  // Verlauf im Browser

  function loadMessages() {
    try {
      return JSON.parse(sessionStorage.getItem(STORAGE_KEY)) || [];
    } catch {
      return [];
    }
  }

  function saveMessages() {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // Ohne sessionStorage funktioniert der Chat weiter, nur ohne Erhalt beim Neuladen.
    }
  }

  /** Bisherige Fragen und Antworten in der Form, die der Server erwartet. */
  function conversation() {
    return messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .map(({ role, content }) => ({ role, content }));
  }

  // Darstellung

  /** Fügt Text absatzweise ein; Zeilenumbrüche bleiben erhalten, HTML wird nicht ausgewertet. */
  function appendParagraphs(container, text) {
    text.split(/\n{2,}/).forEach((block) => {
      const paragraph = document.createElement("p");
      block.split("\n").forEach((line, index) => {
        if (index) paragraph.appendChild(document.createElement("br"));
        paragraph.appendChild(document.createTextNode(line));
      });
      container.appendChild(paragraph);
    });
  }

  function speakerLabel(text) {
    const label = document.createElement("span");
    label.className = "visually-hidden";
    label.textContent = text;
    return label;
  }

  function renderSources(sources) {
    const box = document.createElement("div");
    box.className = "sources";
    sources.forEach((source) => {
      const line = document.createElement("p");
      line.append("Quelle: ");
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title;
      line.appendChild(link);
      // PDF-Dokumente kennzeichnen, sofern der Titel das nicht schon tut
      if (source.kind === "document" && !/pdf/i.test(source.title)) line.append(" (PDF)");
      box.appendChild(line);
    });
    return box;
  }

  /** Angebot zur Übergabe, wenn der Chatbot keine Antwort gefunden hat (Abbildung 4.2). */
  function renderFallbackOffer(index) {
    const offer = document.createElement("div");
    offer.className = "fallback-offer";
    const text = document.createElement("p");
    text.textContent = `Möchtest du deine Anfrage an den ${settings.contact.name} übergeben?`;
    const actions = document.createElement("div");
    actions.className = "overlay-actions";
    const accept = document.createElement("button");
    accept.type = "button";
    accept.className = "primary-button";
    accept.dataset.action = "open-handover";
    accept.textContent = "Anfrage übergeben";
    const decline = document.createElement("button");
    decline.type = "button";
    decline.className = "secondary-button";
    decline.dataset.action = "dismiss-offer";
    decline.dataset.index = String(index);
    decline.textContent = "Nein, danke";
    actions.append(accept, decline);
    offer.append(text, actions);
    return offer;
  }

  function renderMessage(message, index) {
    const bubble = document.createElement("div");
    bubble.className = `message message-${message.role}`;
    if (message.role === "user") bubble.appendChild(speakerLabel("Du: "));
    if (message.role === "assistant") bubble.appendChild(speakerLabel(`${settings.name}: `));
    appendParagraphs(bubble, message.content);
    if (message.sources && message.sources.length) bubble.appendChild(renderSources(message.sources));
    if (message.fallback && !message.offerDismissed && index !== undefined) {
      bubble.appendChild(renderFallbackOffer(index));
    }
    log.appendChild(bubble);
    log.scrollTop = log.scrollHeight;
  }

  function greeting() {
    return {
      role: "greeting",
      content:
        `Hallo! Ich bin ein KI-Assistent und beantworte Fragen zu Studium und Studienorganisation ` +
        `auf Grundlage der Webseite der ${settings.university}. Meine Antworten können Fehler enthalten. ` +
        `Prüfe wichtige Angaben bitte in den angegebenen Quellen. Bitte gib keine persönlichen Daten ` +
        `wie deinen Namen oder deine Matrikelnummer ein.`,
    };
  }

  function renderAll() {
    log.replaceChildren();
    const welcome = greeting();
    const bubble = document.createElement("div");
    bubble.className = "message message-assistant";
    appendParagraphs(bubble, welcome.content);
    log.appendChild(bubble);
    messages.forEach((message, index) => renderMessage(message, index));
  }

  function addMessage(message) {
    messages.push(message);
    saveMessages();
    renderMessage(message, messages.length - 1);
  }

  /** Hinweis im Verlauf, etwa bei Fehlern; wird nicht als Gesprächsverlauf an den Server übermittelt. */
  function addNote(text) {
    addMessage({ role: "note", content: text });
  }

  function setBusy(value) {
    busy = value;
    sendButton.disabled = value;
    log.setAttribute("aria-busy", String(value));
    status.textContent = value ? `${settings.name} schreibt …` : "";
  }

  // Konfiguration

  function renderContact() {
    const details = byId("contact-details");
    details.replaceChildren();
    const entries = [
      ["E-Mail", settings.contact.email, settings.contact.email && `mailto:${settings.contact.email}`],
      ["Telefon", settings.contact.phone, settings.contact.phone && `tel:${settings.contact.phone.replace(/[^\d+]/g, "")}`],
      ["Webseite", settings.contact.url, settings.contact.url],
    ].filter(([, value]) => value);
    if (!entries.length) {
      const empty = document.createElement("dd");
      empty.textContent = "Die Kontaktdaten sind im Prototyp noch nicht hinterlegt.";
      details.appendChild(empty);
      return;
    }
    entries.forEach(([label, value, href]) => {
      const term = document.createElement("dt");
      term.textContent = label;
      const definition = document.createElement("dd");
      const link = document.createElement("a");
      link.href = href;
      link.textContent = value;
      if (label === "Webseite") {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
      definition.appendChild(link);
      details.append(term, definition);
    });
  }

  async function loadSettings() {
    try {
      const response = await fetch("/api/config");
      if (response.ok) settings = await response.json();
    } catch {
      // Ohne Verbindung bleiben die Standardtexte stehen.
    }
    const texts = {
      name: settings.name,
      subtitle: settings.subtitle,
      university: settings.university,
      "contact-name": settings.contact.name,
      launcher: `${settings.name} öffnen`,
    };
    Object.entries(texts).forEach(([key, value]) => {
      if (!value) return;
      document.querySelectorAll(`[data-text="${key}"]`).forEach((node) => {
        node.textContent = value;
      });
    });
    renderContact();
    renderAll();
  }

  // Öffnen und Schließen

  function openPanel() {
    panel.hidden = false;
    launcher.setAttribute("aria-expanded", "true");
    log.scrollTop = log.scrollHeight;
    input.focus();
  }

  function closePanel() {
    closeOverlay(false);
    panel.hidden = true;
    launcher.setAttribute("aria-expanded", "false");
    launcher.focus();
  }

  function showOverlay(view, headingId) {
    contactView.hidden = view !== contactView;
    handoverView.hidden = view !== handoverView;
    // Die Ebene beginnt unter der Kopfzeile, deren Höhe je nach Umbruch des Untertitels variiert.
    view.style.top = `${panel.querySelector(".chat-header").offsetHeight}px`;
    // Verlauf und Eingabe liegen unter der Ebene und sind solange nicht bedienbar.
    log.inert = true;
    chatForm.inert = true;
    byId(headingId).focus();
  }

  function closeOverlay(restoreFocus = true) {
    contactView.hidden = true;
    handoverView.hidden = true;
    log.inert = false;
    chatForm.inert = false;
    if (restoreFocus) input.focus();
  }

  // Übergabe an den Studierendenservice (nur Demonstration)

  function openHandover() {
    // Betreff ist die zuletzt gestellte Sachfrage (bei Folgefragen umformuliert). Begrüßungen
    // und Dank liefern keine Suchanfrage und werden übersprungen. Der Betreff ist bearbeitbar.
    const subject =
      [...messages].reverse().find((message) => message.role === "assistant" && message.searchQuery)?.searchQuery ||
      "";
    handoverView.reset();
    byId("handover-subject").value = subject;
    ["handover-email-error", "handover-consent-error"].forEach((id) => (byId(id).hidden = true));
    ["handover-email", "handover-consent"].forEach((id) => byId(id).removeAttribute("aria-invalid"));

    const history = byId("handover-history");
    history.replaceChildren();
    const turns = conversation();
    if (!turns.length) history.textContent = "Noch kein Gesprächsverlauf.";
    turns.forEach((message) => {
      const line = document.createElement("p");
      const speaker = document.createElement("strong");
      speaker.textContent = message.role === "user" ? "Du: " : `${settings.name}: `;
      line.append(speaker, message.content);
      history.appendChild(line);
    });
    showOverlay(handoverView, "handover-title");
  }

  function validateField(id, valid) {
    byId(`${id}-error`).hidden = valid;
    if (valid) byId(id).removeAttribute("aria-invalid");
    else byId(id).setAttribute("aria-invalid", "true");
    return valid;
  }

  handoverView.addEventListener("submit", (event) => {
    event.preventDefault();
    const email = byId("handover-email");
    const emailValid = validateField("handover-email", email.value.trim() !== "" && email.checkValidity());
    const consentValid = validateField("handover-consent", byId("handover-consent").checked);
    if (!emailValid) return email.focus();
    if (!consentValid) return byId("handover-consent").focus();
    // Im Prototyp wird nichts versendet: Das Formular wird geleert und die Übergabe nur bestätigt.
    handoverView.reset();
    closeOverlay();
    // Offene Übergabe-Angebote sind damit erledigt.
    messages.forEach((message) => {
      if (message.fallback) message.offerDismissed = true;
    });
    log.querySelectorAll(".fallback-offer").forEach((offer) => offer.remove());
    addNote(
      `Demo: Die Übergabe an den ${settings.contact.name} wurde nur nachgestellt. ` +
        "Es wurden keine Daten versendet oder gespeichert."
    );
  });

  // Anfragen

  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || busy) return;
    const history = conversation().slice(-MAX_HISTORY);
    addMessage({ role: "user", content: question });
    input.value = "";
    setBusy(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      addMessage({
        role: "assistant",
        content: data.text,
        sources: data.sources,
        fallback: data.fallback,
        searchQuery: data.search_query,
      });
    } catch {
      addNote(
        "Entschuldigung, es ist ein technischer Fehler aufgetreten. Bitte versuche es erneut " +
          `oder wende dich über „Kontakt“ an den ${settings.contact.name}.`
      );
    } finally {
      setBusy(false);
      input.focus();
    }
  });

  // Bedienelemente

  launcher.addEventListener("click", () => (panel.hidden ? openPanel() : closePanel()));
  byId("chat-close").addEventListener("click", closePanel);
  byId("contact-button").addEventListener("click", () => showOverlay(contactView, "contact-title"));

  panel.addEventListener("click", (event) => {
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const action = button.dataset.action;
    if (action === "open-handover") openHandover();
    if (action === "close-overlay") closeOverlay();
    if (action === "dismiss-offer") {
      messages[Number(button.dataset.index)].offerDismissed = true;
      saveMessages();
      button.closest(".fallback-offer").remove();
      input.focus();
    }
  });

  panel.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (!contactView.hidden || !handoverView.hidden) closeOverlay();
    else closePanel();
  });

  loadSettings();
})();
