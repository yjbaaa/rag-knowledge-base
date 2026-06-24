"""
Conversation Memory Manager
Multi-turn dialogue context management with:
- Token-aware window trimming
- Multi-session support
- JSON file persistence
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from app.core.config import MAX_MEMORY_TOKENS, MEMORY_WINDOW_SIZE, DATA_DIR


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str       # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict) -> "ConversationTurn":
        return cls(role=d["role"], content=d["content"], timestamp=d.get("timestamp", 0))


class ConversationMemory:
    """
    Token-aware conversation memory for one session.

    Features:
    - Sliding window: keeps last N turns
    - Token budget: trims when total exceeds MAX_MEMORY_TOKENS
    - Format conversion for LLM prompt injection
    """

    def __init__(
        self,
        session_id: str = "default",
        max_tokens: int = MAX_MEMORY_TOKENS,
        window_size: int = MEMORY_WINDOW_SIZE,
    ):
        self.session_id = session_id
        self.max_tokens = max_tokens
        self.window_size = window_size
        self._turns: List[ConversationTurn] = []

    # ---- History Management ----

    def add_user_message(self, content: str):
        self._turns.append(ConversationTurn(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str):
        self._turns.append(ConversationTurn(role="assistant", content=content))
        self._trim()

    def add_turn(self, user_msg: str, assistant_msg: str):
        """Add a complete Q&A turn."""
        self.add_user_message(user_msg)
        self.add_assistant_message(assistant_msg)

    def clear(self):
        """Clear all history."""
        self._turns.clear()

    # ---- Retrieval ----

    def get_history(self, last_n: Optional[int] = None) -> List[ConversationTurn]:
        """Get recent conversation turns."""
        n = last_n or self.window_size * 2  # *2 because each turn = user + assistant
        return self._turns[-n:]

    def get_formatted_history(self, last_n: Optional[int] = None) -> str:
        """
        Format conversation history as a string for LLM context injection.

        Format:
            User: <message>
            Assistant: <message>
        """
        turns = self.get_history(last_n)
        if not turns:
            return ""
        lines = []
        for turn in turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")
        return "\n".join(lines)

    def get_langchain_messages(self, last_n: Optional[int] = None):
        """Return history as LangChain message list (lazy import)."""
        from langchain_core.messages import HumanMessage, AIMessage
        turns = self.get_history(last_n)
        messages = []
        for turn in turns:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))
        return messages

    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn.content
        return None

    # ---- Properties ----

    @property
    def turn_count(self) -> int:
        """Number of complete Q&A pairs."""
        return len(self._turns) // 2

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    @property
    def total_chars(self) -> int:
        return sum(len(t.content) for t in self._turns)

    # ---- Internal ----

    def _trim(self):
        """Trim history to fit within token budget and window size."""
        # 1. Window size trim (keep last N turns)
        max_messages = self.window_size * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

        # 2. Token budget trim (approx: 1 token ≈ 2 Chinese chars ≈ 4 English chars)
        estimated_tokens = self.total_chars // 2
        while estimated_tokens > self.max_tokens and len(self._turns) > 2:
            # Remove oldest complete Q&A pair
            if self._turns[0].role == "user":
                self._turns = self._turns[2:]  # Remove one complete turn
            else:
                self._turns.pop(0)
            estimated_tokens = self.total_chars // 2


# =====================================================================
#  Session Manager (multi-session support)
# =====================================================================

class SessionManager:
    """
    Manages multiple conversation sessions.

    Usage:
        mgr = SessionManager()
        mem = mgr.get_session("user_123")
        mem.add_turn("Hello", "Hi there!")
        mgr.save_all()
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self._storage_dir = storage_dir or (DATA_DIR / "memory")
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, ConversationMemory] = {}

    def get_session(self, session_id: str) -> ConversationMemory:
        """Get or create a session."""
        if session_id not in self._sessions:
            mem = ConversationMemory(session_id=session_id)
            # Try to load from disk
            self._load_session(mem)
            self._sessions[session_id] = mem
        return self._sessions[session_id]

    def delete_session(self, session_id: str):
        """Delete a session from memory and disk."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        filepath = self._get_filepath(session_id)
        if filepath.exists():
            filepath.unlink()

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    def save_all(self):
        """Persist all sessions to disk."""
        for session_id, mem in self._sessions.items():
            self._save_session(mem)

    def save_session(self, session_id: str):
        """Persist a single session."""
        if session_id in self._sessions:
            self._save_session(self._sessions[session_id])

    # ---- Internal ----

    def _get_filepath(self, session_id: str) -> Path:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "_-")
        return self._storage_dir / f"{safe_id}.json"

    def _save_session(self, mem: ConversationMemory):
        filepath = self._get_filepath(mem.session_id)
        data = {
            "session_id": mem.session_id,
            "turns": [t.to_dict() for t in mem._turns],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_session(self, mem: ConversationMemory):
        filepath = self._get_filepath(mem.session_id)
        if not filepath.exists():
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            mem._turns = [ConversationTurn.from_dict(t) for t in data.get("turns", [])]
        except Exception:
            pass  # Corrupted file, start fresh


# ========== Global singleton ==========

_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
