"""Persistent topic filter store.

Telegram Forum (supergroup with topics enabled) delivers each message with a
``message_thread_id`` that identifies which topic thread it belongs to.

Default behaviour per chat:
  • Regular group / channel — no thread filtering, all messages accepted.
  • Forum supergroup with NO topics configured — all topic threads accepted.
  • Forum supergroup WITH at least one /set_topic — only those threads accepted.

Persistence: simple JSON file mounted as a Docker volume so state survives
container restarts.

Schema (topics.json):
  {
    "-1001234567890": {
      "is_forum": true,
      "allowed_threads": [123, 456]   // empty list = accept all
    }
  }
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set


class TopicStore:
    """Thread-safe, file-backed topic filter."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = self._load()

    # ── persistence ───────────────────────────────────────────────────

    def _load(self) -> Dict[str, dict]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))

    # ── chat/forum registration ───────────────────────────────────────

    def set_forum(self, chat_id: int, is_forum: bool) -> None:
        key = str(chat_id)
        with self._lock:
            entry = self._data.setdefault(key, {"is_forum": False, "allowed_threads": []})
            entry["is_forum"] = is_forum
            self._save()

    # ── topic management ──────────────────────────────────────────────

    def enable_thread(self, chat_id: int, thread_id: int) -> None:
        """Add *thread_id* to the allowed list for *chat_id*."""
        key = str(chat_id)
        with self._lock:
            entry = self._data.setdefault(key, {"is_forum": True, "allowed_threads": []})
            if thread_id not in entry["allowed_threads"]:
                entry["allowed_threads"].append(thread_id)
            self._save()

    def disable_thread(self, chat_id: int, thread_id: int) -> None:
        """Remove *thread_id* from the allowed list for *chat_id*."""
        key = str(chat_id)
        with self._lock:
            entry = self._data.get(key)
            if entry and thread_id in entry["allowed_threads"]:
                entry["allowed_threads"].remove(thread_id)
                self._save()

    def get_allowed_threads(self, chat_id: int) -> List[int]:
        with self._lock:
            return list(self._data.get(str(chat_id), {}).get("allowed_threads", []))

    def is_forum(self, chat_id: int) -> bool:
        with self._lock:
            return bool(self._data.get(str(chat_id), {}).get("is_forum", False))

    # ── main filter ───────────────────────────────────────────────────

    def is_message_allowed(
        self,
        chat_id: int,
        thread_id: Optional[int],
    ) -> bool:
        """Return True if this message should be forwarded to the queue."""
        key = str(chat_id)
        with self._lock:
            entry = self._data.get(key)
        if entry is None:
            # Chat not in store → accept everything (will update is_forum on first message)
            return True
        allowed = entry.get("allowed_threads", [])
        if not allowed:
            # No topics configured → accept all threads
            return True
        # Topics configured → only accept listed thread IDs
        return thread_id in allowed

    def all_configured_chats(self) -> List[int]:
        with self._lock:
            return [int(k) for k in self._data]
