"""
HTTP API for the dashboard.

    uvicorn o_an_quan.api.app:app --reload        (or: python -m o_an_quan.api)
"""
from collections import Counter
from statistics import mean

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..config import (
    CLASSIFIER_MODEL, DEFAULT_MODEL, GEMMA_MODEL, LLAMA_MODELS, MAX_ROUNDS, OPENROUTER_API_KEY,
)
from ..experiments.reports import (
    depth_rows, llm_turns, model_rows, persona_rows, reason_rows,
)
from ..prompts import PERSONAS, RULES
from ..utils import set_verbose
from . import jobs, store
from .schemas import NewGameRequest

set_verbose(False)  # tool calls run in background threads; keep the server log clean

app = FastAPI(title="Ô Ăn Quan API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "has_api_key": bool(OPENROUTER_API_KEY),
        "results_dir": str(store.results_dir()),
    }


@app.get("/api/config")
def config():
    return {
        "models": list(dict.fromkeys([DEFAULT_MODEL, GEMMA_MODEL, *LLAMA_MODELS])),
        "default_model": DEFAULT_MODEL,
        "classifier_model": CLASSIFIER_MODEL,
        "max_rounds": MAX_ROUNDS,
        "personas": list(PERSONAS),
    }


@app.get("/api/personas")
def personas():
    return [{"name": name, "content": content} for name, content in PERSONAS.items()]


@app.get("/api/rules")
def rules():
    return {"content": RULES}


@app.get("/api/games")
def list_games():
    games, _ = store.load_all()
    # Newest first; records without a timestamp keep their file order at the end
    games = sorted(reversed(games), key=lambda g: g.get("created_at") or "", reverse=True)
    return [store.summarize(g) for g in games]


@app.get("/api/games/{game_id}")
def get_game(game_id: str):
    game = store.get_game(game_id)
    if game is None:
        raise HTTPException(404, f"Game {game_id} not found")
    return game


@app.post("/api/games", status_code=202)
def create_game(request: NewGameRequest):
    if "llm" in (request.agent_a.type, request.agent_b.type) and not OPENROUTER_API_KEY:
        raise HTTPException(400, "OPENROUTER_API_KEY is not set; only random agents are available")
    try:
        job = jobs.start_job(request)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return [
        {k: v for k, v in job.to_dict().items() if k not in ("turns", "initial_board")}
        | {"moves": len(job.turns)}
        for job in jobs.list_jobs()
    ]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return job.to_dict()


@app.get("/api/stats")
def stats():
    games, reasons = store.load_all()
    turns = llm_turns(games)
    return {
        "totals": {
            "games": len(games),
            "llm_moves": len(turns),
            "invalid_per_move": round(mean(t["invalid_attempts"] for t in turns), 3) if turns else None,
            "fallback_rate": round(100 * mean(t["fallback"] for t in turns), 1) if turns else None,
            "avg_rounds": round(mean(g["rounds_played"] for g in games), 1) if games else None,
            "classified_moves": len(reasons),
        },
        "winners": dict(Counter(g["winner"] for g in games)),
        "end_reasons": dict(Counter(g["end_reason"] for g in games)),
        "experiments": dict(Counter(g.get("experiment", "dashboard") for g in games)),
        "personas": persona_rows(games),
        "models": model_rows(games),
        "depth": depth_rows(games),
        "reasoning": reason_rows(reasons),
    }
