"""Auswertung von ``robots.txt`` nach RFC 9309.

Das Modul ``urllib.robotparser`` der Standardbibliothek unterstützt die Platzhalter
``*`` und ``$`` nicht, die etwa in ``Disallow: /*tx_powermail_pi1`` verwendet werden.
Daher werden die Regeln hier selbst ausgewertet.
"""

import re
from dataclasses import dataclass, field


@dataclass
class _Group:
    """Eine Gruppe von Regeln für einen oder mehrere User-Agents."""

    agents: list[str] = field(default_factory=list)
    rules: list[tuple[bool, str]] = field(default_factory=list)


class RobotsRules:
    """Regeln einer ``robots.txt`` für einen bestimmten User-Agent."""

    def __init__(self, text: str, user_agent: str) -> None:
        """Liest die Regeln ein.

        Args:
            text: Inhalt der ``robots.txt``.
            user_agent: Kennung des Crawlers. Maßgeblich ist der Teil vor einem ``/``.
        """
        self._rules = self._select_rules(self._parse(text), user_agent)
        self._disallow_all = False

    @classmethod
    def disallow_all(cls) -> "RobotsRules":
        """Erzeugt Regeln, die jeden Abruf verbieten.

        Nach RFC 9309 ist das anzunehmen, wenn die ``robots.txt`` wegen eines
        Serverfehlers nicht abrufbar ist.
        """
        rules = cls("", "")
        rules._disallow_all = True
        return rules

    def allows(self, path: str) -> bool:
        """Prüft, ob ein Pfad abgerufen werden darf.

        Es gilt die Regel mit dem längsten passenden Muster, bei Gleichstand ``Allow``.

        Args:
            path: Pfad einschließlich Query, z. B. ``/studium/?a=1``.
        """
        if self._disallow_all:
            return False
        best_length, allowed = -1, True
        for is_allow, pattern in self._rules:
            if _matches(pattern, path):
                if len(pattern) > best_length or (len(pattern) == best_length and is_allow):
                    best_length, allowed = len(pattern), is_allow
        return allowed

    @staticmethod
    def _parse(text: str) -> list[_Group]:
        """Zerlegt den Text in Gruppen. Aufeinanderfolgende User-Agent-Zeilen bilden eine Gruppe."""
        groups: list[_Group] = []
        current: _Group | None = None
        previous_was_agent = False
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if current is None or not previous_was_agent:
                    current = _Group()
                    groups.append(current)
                current.agents.append(value.lower())
                previous_was_agent = True
                continue
            previous_was_agent = False
            # Ein leeres Disallow verbietet nichts und wird übergangen.
            if key in ("allow", "disallow") and current is not None and value:
                current.rules.append((key == "allow", value))
        return groups

    @staticmethod
    def _select_rules(groups: list[_Group], user_agent: str) -> list[tuple[bool, str]]:
        """Wählt die Regeln für den User-Agent, ersatzweise die für ``*``."""
        token = user_agent.split("/", 1)[0].lower()
        if token and any(token in group.agents for group in groups):
            return [rule for group in groups if token in group.agents for rule in group.rules]
        return [rule for group in groups if "*" in group.agents for rule in group.rules]


def _matches(pattern: str, path: str) -> bool:
    """Prüft, ob ein Muster mit ``*`` (beliebige Zeichen) und ``$`` (Ende) auf den Pfad passt."""
    regex = re.escape(pattern).replace(r"\*", ".*")
    if regex.endswith(r"\$"):
        regex = regex[:-2] + "$"
    return re.match(regex, path) is not None
