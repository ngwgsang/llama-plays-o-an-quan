"""Game-playing experiments (paper, Sections V-A and V-B)."""
from ..agents import Agent, RandomAgent
from ..config import GAMES_PER_MATCHUP
from ..game import play_game
from ..prompts import PERSONAS
from ..utils import console
from .io import append_jsonl


def run_personas(model: str, personas: list[str], out: str,
                 games: int = GAMES_PER_MATCHUP, verbose: bool = False):
    """Experiment A: each persona agent (team A) vs RandomAgent (team B)."""
    for persona_name in personas:
        for game_idx in range(games):
            agent_a = Agent("A", PERSONAS[persona_name], model, name=persona_name, seed=game_idx)
            agent_b = RandomAgent("B", seed=game_idx)
            record = play_game(agent_a, agent_b, verbose=verbose)
            record.update(experiment="personas", persona=persona_name, game=game_idx)
            append_jsonl(out, record)
            console.print(
                f"[personas] {persona_name} game {game_idx + 1}/{games}: "
                f"{record['final_score']} winner={record['winner']}"
            )


def run_models(models: list[str], opponent: str, persona_name: str, out: str,
               games: int = GAMES_PER_MATCHUP, opponent_first: bool = False, verbose: bool = False):
    """Experiment B: each model vs the opponent model, same persona on both sides."""
    persona = PERSONAS[persona_name]
    for model in models:
        for game_idx in range(games):
            model_team, opponent_team = ("B", "A") if opponent_first else ("A", "B")
            agent = Agent(model_team, persona, model, name=model, seed=game_idx)
            rival = Agent(opponent_team, persona, opponent, name=opponent, seed=game_idx)
            record = play_game(rival, agent, verbose=verbose) if opponent_first \
                else play_game(agent, rival, verbose=verbose)
            record.update(experiment="models", model=model, opponent=opponent,
                          model_team=model_team, game=game_idx)
            append_jsonl(out, record)
            console.print(
                f"[models] {model} game {game_idx + 1}/{games}: "
                f"{record['final_score']} winner={record['winner']}"
            )
