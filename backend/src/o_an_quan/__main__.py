"""
Play one game and save it as a JSON log.

    python -m o_an_quan                                  # LLM demo, 5 rounds
    python -m o_an_quan --a random --b random --rounds 25 --seed 0
    python -m o_an_quan --a llm --persona-a DefensiveAgent --b random
"""
import argparse

from .agents import Agent, RandomAgent
from .config import DEFAULT_MODEL, LOGS_DIR
from .game import play_game, write_game_log
from .prompts import PERSONAS
from .utils import console, set_verbose


def build_agent(team: str, kind: str, model: str, persona: str, seed):
    if kind == "random":
        return RandomAgent(team, seed=seed)
    return Agent(team, PERSONAS[persona], model, name=persona, seed=seed)


def main():
    parser = argparse.ArgumentParser(prog="python -m o_an_quan", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", choices=["llm", "random"], default="llm", help="team A agent")
    parser.add_argument("--b", choices=["llm", "random"], default="llm", help="team B agent")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model for LLM agents")
    parser.add_argument("--persona-a", default="StrategicControlAgent", choices=list(PERSONAS))
    parser.add_argument("--persona-b", default="RiskTakerAgent", choices=list(PERSONAS))
    parser.add_argument("--rounds", type=int, default=5, help="round limit")
    parser.add_argument("--seed", type=int, default=None, help="seed for random choices")
    parser.add_argument("--log-dir", default=LOGS_DIR, help="where to write the JSON log")
    parser.add_argument("--quiet", action="store_true", help="do not print moves to the console")
    args = parser.parse_args()

    set_verbose(not args.quiet)
    seed_b = None if args.seed is None else args.seed + 1
    agent_a = build_agent("A", args.a, args.model, args.persona_a, args.seed)
    agent_b = build_agent("B", args.b, args.model, args.persona_b, seed_b)

    record = play_game(agent_a, agent_b, max_rounds=args.rounds, verbose=not args.quiet, record_events=True)
    config = {
        "max_rounds": args.rounds,
        "seed": args.seed,
        "agents": {
            team: {"type": kind, "model": agent.model, "persona": persona if kind == "llm" else None}
            for team, kind, agent, persona in [
                ("A", args.a, agent_a, args.persona_a),
                ("B", args.b, agent_b, args.persona_b),
            ]
        },
    }
    path = write_game_log(record, args.log_dir, config)

    console.print(f"\nFinal score: {record['final_score']}  Winner: {record['winner']} ({record['end_reason']})")
    console.print(f"Log written to {path}")


if __name__ == "__main__":
    main()
