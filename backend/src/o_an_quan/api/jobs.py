"""Games started from the dashboard run in background threads; clients poll the job."""
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..agents import Agent, RandomAgent
from ..config import DEFAULT_MODEL
from ..game import Environment, play_game
from ..prompts import PERSONAS
from . import store
from .schemas import AgentConfig, NewGameRequest


@dataclass
class Job:
    id: str
    request: NewGameRequest
    status: str = "running"            # running | finished | failed
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_a: str = ""
    agent_b: str = ""
    initial_board: dict = field(default_factory=dict)
    turns: list = field(default_factory=list)
    game_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "request": self.request.model_dump(),
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "max_rounds": self.request.max_rounds,
            "initial_board": self.initial_board,
            "turns": list(self.turns),
            "game_id": self.game_id,
            "error": self.error,
        }


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def build_agent(team: str, config: AgentConfig, seed: Optional[int]):
    if config.type == "random":
        return RandomAgent(team, seed=seed)
    if config.persona and config.persona not in PERSONAS:
        raise ValueError(f"Unknown persona: {config.persona}")
    model = config.model or DEFAULT_MODEL
    return Agent(
        team,
        PERSONAS.get(config.persona or "", ""),
        model,
        name=config.persona or model,
        seed=seed,
    )


def _run(job: Job, agent_a, agent_b):
    try:
        record = play_game(
            agent_a, agent_b,
            max_rounds=job.request.max_rounds,
            verbose=False,
            on_turn=job.turns.append,
        )
        record.update(experiment="dashboard")
        job.game_id = store.save_game(record)
        job.status = "finished"
    except Exception as e:
        job.error = f"{type(e).__name__}: {e}"
        job.status = "failed"
        traceback.print_exc()


def start_job(request: NewGameRequest) -> Job:
    # Build agents up front so configuration errors are returned immediately
    agent_a = build_agent("A", request.agent_a, request.seed)
    agent_b = build_agent("B", request.agent_b, request.seed)
    job = Job(
        id=uuid.uuid4().hex[:12],
        request=request,
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        initial_board=Environment(verbose=False).board_counts(),
    )
    with _lock:
        _jobs[job.id] = job
    threading.Thread(target=_run, args=(job, agent_a, agent_b), daemon=True).start()
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def list_jobs() -> list[Job]:
    return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
