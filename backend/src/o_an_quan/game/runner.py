from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from .environment import Environment
from ..config import MAX_ROUNDS
from ..utils import console


def build_history(my_last: Optional[dict], opponent_last: Optional[dict]) -> str:
    """
    H in the paper: a textual summary of the previous turn, including the
    opponent's move and rationale.
    """
    lines = []
    if my_last:
        lines.append(f"My previous move: {my_last['pos']} {my_last['way']}.")
        if my_last.get("reason"):
            lines.append(f"My reasoning: {my_last['reason']}")
        lines.append(
            f"Result: I captured {my_last['captured']} points; score after my move "
            f"A = {my_last['score']['A']}, B = {my_last['score']['B']}."
        )
    if opponent_last:
        lines.append(f"Opponent's last move: {opponent_last['pos']} {opponent_last['way']}.")
        if opponent_last.get("reason"):
            lines.append(f"Opponent's reasoning: {opponent_last['reason']}")
        lines.append(f"The opponent captured {opponent_last['captured']} points.")
    return "\n".join(lines)


def compact_events(events: list[dict]) -> list[dict]:
    """Environment events with token lists replaced by counts (for JSON logs)."""
    compact = []
    for e in events:
        kind = e["type"]
        if kind == "pickup":
            compact.append({"type": kind, "pos": e["pos"], "peasants": len(e["pieces"])})
        elif kind == "drop":
            compact.append({"type": kind, "from": e["from_pos"], "to": e["to_pos"]})
        elif kind == "capture":
            compact.append({
                "type": kind,
                "pos": e["pos"],
                "peasants": sum(t.startswith("peasant") for t in e["pieces"]),
                "mandarin": any(t.startswith("mandarin") for t in e["pieces"]),
            })
        elif kind == "forbidden_capture":
            compact.append({"type": kind, "pos": e["pos"], "reason": e["reason"]})
        else:  # score_update, end_turn, end_game
            compact.append(e)
    return compact


def play_game(
    agent_a,
    agent_b,
    max_rounds: int = MAX_ROUNDS,
    verbose: bool = True,
    on_turn: Optional[Callable[[dict], None]] = None,
    record_events: bool = False,
) -> Dict[str, Any]:
    """
    Play one game between agent_a (team A, moves first) and agent_b (team B).

    Returns a JSON-serialisable record with every turn (including the board
    after the move), per-round score snapshots, the end reason and the winner.
    `on_turn` is called with each turn record as soon as it is played.
    `record_events` adds the pickup / drop / capture events of every move.
    """
    env = Environment(verbose=verbose)
    agents = {"A": agent_a, "B": agent_b}
    last_turn: Dict[str, Optional[dict]] = {"A": None, "B": None}
    turns = []
    round_scores = []
    end_reason = ""
    initial_board = env.board_counts()

    for _ in range(max_rounds):
        round_idx = env.start_round()
        if verbose:
            console.print(f"\n--- Start of Round {round_idx} ---\n")

        for team in ("A", "B"):
            opponent = "B" if team == "A" else "A"

            # E2: restore peasants at the start of the turn if needed
            can_continue, message = env.restore_peasants(team)
            if message and verbose:
                console.print(message)
            if not can_continue:
                end_reason = f"END_BY_NO_RESTORE_{team}"
                break
            restored = bool(message)

            available = env.get_available_pos(team)
            history = build_history(last_turn[team], last_turn[opponent])
            result = agents[team].play_turn(env.render_state(), available, history)

            score_before = env.get_game_state()["score"][team]
            steps, events, is_end = env.commit_action(
                {"pos": result.pos, "way": result.way}, team
            )
            score = env.get_game_state()["score"].copy()

            turn = {
                "round": round_idx,
                "team": team,
                "agent": agents[team].name,
                "model": agents[team].model,
                **result.to_dict(),
                "steps": steps,
                "captured": score[team] - score_before,
                "score": score,
                "board": env.board_counts(),
                "restored": restored,
            }
            if record_events:
                turn["events"] = compact_events(events)
            turns.append(turn)
            last_turn[team] = turn
            if on_turn:
                on_turn(turn)

            if is_end:
                end_reason = env.is_end()[1]
                break

        round_scores.append(env.get_game_state()["score"].copy())
        if end_reason:
            break

    if not end_reason:
        end_reason = "END_BY_MAX_ROUNDS"

    final_score = env.finalize(end_reason)
    winner = env.get_winner(end_reason)
    if verbose:
        console.print(f"\n[GAME END] {end_reason} | score {final_score} | winner {winner}")

    return {
        "agent_a": agent_a.name,
        "agent_b": agent_b.name,
        "model_a": agent_a.model,
        "model_b": agent_b.model,
        "rounds_played": env.get_game_state()["round"],
        "end_reason": end_reason,
        "final_score": final_score,
        "winner": winner,
        "round_scores": round_scores,
        "initial_board": initial_board,
        "final_board": env.board_counts(),
        "turns": turns,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
