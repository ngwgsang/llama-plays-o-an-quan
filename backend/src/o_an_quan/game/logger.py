"""Write one game as a pretty-printed JSON log file."""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import LOGS_DIR

LOG_SCHEMA = "o-an-quan/game-log@1"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text.split("/")[-1]).strip("-") or "agent"


def write_game_log(
    record: dict,
    directory: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Save `record` (from play_game) to <directory>/game-<timestamp>-<A>-vs-<B>.json
    and return the path. `config` describes how the game was set up (agents,
    seed, round limit) and is stored next to the record.
    """
    folder = Path(directory or LOGS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"game-{stamp}-{_slug(record['agent_a'])}-vs-{_slug(record['agent_b'])}.json"

    log = {"schema": LOG_SCHEMA, "config": config or {}, **record}
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
