"""Lightweight conversation memory for multi-turn RAG sessions."""

import json
import os
from typing import Dict, List, Optional, Tuple


class ConversationMemory:
    """Store recent question/answer turns per session.

    When ``max_turns`` is 0 the memory is disabled (calls are no-ops). Sessions
    are persisted as a JSON file so the CLI can keep a conversation alive across
    invocations.
    """

    def __init__(self, max_turns: int = 0, path: Optional[str] = None):
        self.max_turns = max_turns
        self.path = path
        self.sessions: Dict[str, List[Tuple[str, str]]] = {}
        if path and os.path.exists(path):
            self.load()

    def add(self, session_id: str, question: str, answer: str) -> None:
        if self.max_turns <= 0 or not session_id:
            return
        history = self.sessions.setdefault(session_id, [])
        history.append((question, answer))
        # Keep at most ``max_turns`` question/answer pairs.
        self.sessions[session_id] = history[-self.max_turns:]
        self.save()

    def history(self, session_id: str) -> List[Tuple[str, str]]:
        if not session_id:
            return []
        return list(self.sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.save()

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.sessions, handle, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as handle:
            self.sessions = json.load(handle)
