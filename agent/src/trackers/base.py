from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class TrackerDraft:
    title: str
    description: str
    tags: List[str]
    project: str
    status: str = "New"
    priority: str = "Normal"
    item_type: str = "Bug"


class TaskTracker(Protocol):
    async def list_candidates(self, draft: TrackerDraft, limit: int = 200) -> List[Dict[str, Any]]: ...

    async def create_item(self, draft: TrackerDraft, placement: str = "kanban") -> Dict[str, Any]: ...

    async def attach_from_url(
        self,
        *,
        item_id: int,
        draft: TrackerDraft,
        file_url: str,
        filename: str,
    ) -> Dict[str, Any]: ...

    async def append_error_note(
        self,
        *,
        item_id: int,
        draft: TrackerDraft,
        note: str,
    ) -> Dict[str, Any]: ...
