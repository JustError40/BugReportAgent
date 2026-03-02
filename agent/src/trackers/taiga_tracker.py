from __future__ import annotations

from typing import Any, Dict, List

from src.mcp.taiga_mcp import TaigaMCPClient
from src.trackers.base import TaskTracker, TrackerDraft


class TaigaTracker(TaskTracker):
    def __init__(self, work_item_type: str = "issue") -> None:
        self._work_item_type = work_item_type

    async def list_candidates(self, draft: TrackerDraft, limit: int = 200) -> List[Dict[str, Any]]:
        async with TaigaMCPClient() as client:
            if self._work_item_type == "user_story":
                return await client.list_user_story_candidates(project_slug=draft.project, limit=limit)
            return await client.list_issue_candidates(project_slug=draft.project, limit=limit)

    async def create_item(self, draft: TrackerDraft, placement: str = "kanban") -> Dict[str, Any]:
        async with TaigaMCPClient() as client:
            if self._work_item_type == "user_story":
                return await client.create_user_story(
                    project_slug=draft.project,
                    subject=draft.title,
                    description=draft.description,
                    tags=draft.tags,
                    status=draft.status,
                    placement=placement,
                )
            return await client.create_issue(
                project_slug=draft.project,
                title=draft.title,
                description=draft.description,
                tags=draft.tags,
                issue_type=draft.item_type,
                status=draft.status,
                priority=draft.priority,
            )

    async def attach_from_url(
        self,
        *,
        item_id: int,
        draft: TrackerDraft,
        file_url: str,
        filename: str,
    ) -> Dict[str, Any]:
        async with TaigaMCPClient() as client:
            if self._work_item_type == "user_story":
                return await client.attach_user_story_from_url(
                    project_slug=draft.project,
                    user_story_id=item_id,
                    file_url=file_url,
                    filename=filename,
                )
            return await client.attach_issue_from_url(
                project_slug=draft.project,
                issue_id=item_id,
                file_url=file_url,
                filename=filename,
            )

    async def append_error_note(
        self,
        *,
        item_id: int,
        draft: TrackerDraft,
        note: str,
    ) -> Dict[str, Any]:
        async with TaigaMCPClient() as client:
            if self._work_item_type == "user_story":
                return await client.append_user_story_note(user_story_id=item_id, note=note)
            return await client.append_issue_note(issue_id=item_id, note=note)
