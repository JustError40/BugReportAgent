from __future__ import annotations

from src.config import get_settings, load_yaml_config
from src.trackers.base import TaskTracker
from src.trackers.taiga_tracker import TaigaTracker


def get_task_tracker() -> TaskTracker:
    cfg = get_settings()
    y = load_yaml_config()
    tracker_cfg = (y or {}).get("task_tracker", {})

    provider = str(tracker_cfg.get("provider", cfg.task_tracker_provider)).strip().lower()
    work_item_type = str(tracker_cfg.get("work_item_type", "issue")).strip().lower()

    if provider == "taiga":
        return TaigaTracker(work_item_type=work_item_type)

    raise ValueError(f"Unsupported task tracker provider: {provider}")
