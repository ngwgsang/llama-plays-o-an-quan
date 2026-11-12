"""
Read / write game records stored as JSONL files in RESULTS_DIR.

A game id is "<file stem>-<line index>", e.g. "personas-12".
Files may also contain reasoning-type rows (written by `experiments classify`).
"""
import json
import threading
from pathlib import Path
from typing import Optional

from ..config import RESULTS_DIR

DASHBOARD_FILE = "games"  # games started from the dashboard
_write_lock = threading.Lock()


def results_dir() -> Path:
    return Path(RESULTS_DIR).resolve()


def _read_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all() -> tuple[list[dict], list[dict]]:
    """Return (game records, reasoning-type rows) from every results file."""
    games, reasons = [], []
    directory = results_dir()
    if not directory.exists():
        return games, reasons
    for path in sorted(directory.glob("*.jsonl")):
        for idx, row in enumerate(_read_file(path)):
            if "turns" in row:
                games.append({**row, "id": f"{path.stem}-{idx}", "source": path.stem})
            elif "label" in row:
                reasons.append(row)
    return games, reasons


def summarize(record: dict) -> dict:
    """Game record without the per-turn details."""
    llm_turns = [t for t in record["turns"] if t["model"] != "random"]
    return {
        "id": record["id"],
        "source": record["source"],
        "experiment": record.get("experiment", "dashboard"),
        "persona": record.get("persona"),
        "agent_a": record["agent_a"],
        "agent_b": record["agent_b"],
        "model_a": record["model_a"],
        "model_b": record["model_b"],
        "rounds_played": record["rounds_played"],
        "moves": len(record["turns"]),
        "end_reason": record["end_reason"],
        "final_score": record["final_score"],
        "winner": record["winner"],
        "invalid_attempts": sum(t["invalid_attempts"] for t in llm_turns),
        "fallbacks": sum(t["fallback"] for t in llm_turns),
        "created_at": record.get("created_at"),
    }


def get_game(game_id: str) -> Optional[dict]:
    stem, _, idx = game_id.rpartition("-")
    path = results_dir() / f"{stem}.jsonl"
    if not stem or not idx.isdigit() or not path.exists():
        return None
    rows = _read_file(path)
    if int(idx) >= len(rows) or "turns" not in rows[int(idx)]:
        return None
    return {**rows[int(idx)], "id": game_id, "source": stem}


def save_game(record: dict, stem: str = DASHBOARD_FILE) -> str:
    """Append a record and return its id."""
    directory = results_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.jsonl"
    with _write_lock:
        idx = len(_read_file(path)) if path.exists() else 0
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return f"{stem}-{idx}"
