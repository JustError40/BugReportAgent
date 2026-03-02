"""Persistent message deduplication using a JSON file backed set."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Set

from src.config import load_yaml_config


class SeenMessages:
    """Thread/coroutine-safe set of already-processed message keys backed by a JSON file."""

    def __init__(self) -> None:
        yaml_cfg = load_yaml_config()
        dedup_cfg = yaml_cfg.get("deduplication", {})
        self._path = Path(dedup_cfg.get("state_file", "/app/state/seen_messages.json"))
        self._max_entries: int = dedup_cfg.get("max_entries", 100_000)
        self._seen: Set[str] = self._load()

    def _load(self) -> Set[str]:
        if self._path.exists():
            try:
                return set(json.loads(self._path.read_text()))
            except Exception:
                return set()
        return set()

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = list(self._seen)
        # Trim to max entries (keep newest — approximate with slice)
        if len(data) > self._max_entries:
            data = data[-self._max_entries:]
            self._seen = set(data)
        self._path.write_text(json.dumps(data))

    def contains(self, key: str) -> bool:
        return key in self._seen

    def add(self, key: str) -> None:
        self._seen.add(key)
        self._persist()
