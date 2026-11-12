"""
Result tables computed from saved game records.

`*_rows` functions return plain dicts (used by the API); `print_*` functions
render the same rows as Rich tables (used by the CLI).
"""
from collections import Counter, defaultdict
from statistics import mean
from typing import Optional

from rich.table import Table

from ..utils import console

# Table I: score snapshots at the end of each phase (round number)
SCORE_PHASES = {"EGE": 10, "MGE": 20, "LGE": 25}
# Section V-C: rounds used for the planning-depth breakdown
DEPTH_PHASES = {"Early (1-5)": (1, 5), "Mid (6-15)": (6, 15), "Late (16-25)": (16, 25)}
REASON_LABELS = ["SHORT_TERM_GAIN", "LONG_TERM_STRATEGY", "AMBIGUOUS"]
REASON_SHORT = {"SHORT_TERM_GAIN": "STG", "LONG_TERM_STRATEGY": "LTS", "AMBIGUOUS": "AMB"}


def rate(n: int, total: int) -> Optional[float]:
    return round(100 * n / total, 1) if total else None


def fmt(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def llm_turns(records: list[dict]) -> list[dict]:
    return [t for r in records for t in r["turns"] if t["model"] != "random"]


# ----------------------------------------------------------------------------
# Table I: persona agents vs RandomAgent
# ----------------------------------------------------------------------------
def persona_rows(records: list[dict]) -> list[dict]:
    by_persona = defaultdict(list)
    for r in records:
        if r.get("experiment") == "personas":
            by_persona[r["persona"]].append(r)

    rows = []
    for persona_name, games in by_persona.items():
        row = {"agent": persona_name, "games": len(games)}
        for phase, round_end in SCORE_PHASES.items():
            row[phase] = round(mean(
                (g["round_scores"][round_end - 1] if len(g["round_scores"]) >= round_end
                 else g["final_score"])["A"]
                for g in games
            ), 1)
        row["avg_point"] = round(mean(g["final_score"]["A"] for g in games), 1)
        row["win_rate"] = rate(sum(g["winner"] == "A" for g in games), len(games))
        row["draw_rate"] = rate(sum(g["winner"] == "DRAW" for g in games), len(games))
        rows.append(row)
    return rows


def print_persona_table(records: list[dict]):
    rows = persona_rows(records)
    if not rows:
        return
    table = Table(title="Table I: Average scores across game phases (persona agent = team A vs RandomAgent)")
    for col in ["Agent", "Games", *SCORE_PHASES, "Avg Point", "Win %", "Draw %"]:
        table.add_column(col)
    for row in rows:
        table.add_row(
            row["agent"], str(row["games"]),
            *[fmt(row[p]) for p in SCORE_PHASES],
            fmt(row["avg_point"]), fmt(row["win_rate"]), fmt(row["draw_rate"]),
        )
    console.print(table)


# ----------------------------------------------------------------------------
# Table II: LLM architectures vs the opponent model
# ----------------------------------------------------------------------------
def model_rows(records: list[dict]) -> list[dict]:
    by_model = defaultdict(list)
    for r in records:
        if r.get("experiment") == "models":
            by_model[(r["model"], r["model_team"])].append(r)

    rows = []
    for (model, team), games in by_model.items():
        own_turns = [t for g in games for t in g["turns"] if t["team"] == team]
        rows.append({
            "model": model,
            "team": team,
            "plays_first": team == "A",
            "opponent": games[0].get("opponent"),
            "games": len(games),
            "win_rate": rate(sum(g["winner"] == team for g in games), len(games)),
            "draw_rate": rate(sum(g["winner"] == "DRAW" for g in games), len(games)),
            "invalid_per_move": round(mean(t["invalid_attempts"] for t in own_turns), 2)
            if own_turns else None,
        })
    return rows


def print_model_table(records: list[dict]):
    rows = model_rows(records)
    if not rows:
        return
    table = Table(title="Table II: Win / draw rates against the opponent model")
    for col in ["Model", "Team", "Games", "Win %", "Draw %", "Illegal outputs / move"]:
        table.add_column(col)
    for row in rows:
        table.add_row(
            row["model"] + (" *" if row["plays_first"] else ""), row["team"], str(row["games"]),
            fmt(row["win_rate"]), fmt(row["draw_rate"]), fmt(row["invalid_per_move"], 2),
        )
    console.print(table)
    console.print("* = plays first (team A)")


# ----------------------------------------------------------------------------
# Planning depth (Section V-C, Figure 3)
# ----------------------------------------------------------------------------
def depth_rows(records: list[dict]) -> list[dict]:
    by_model = defaultdict(list)
    for t in llm_turns(records):
        by_model[t["model"]].append(t)

    rows = []
    for model, model_turns in by_model.items():
        row = {"model": model, "moves": len(model_turns), "phases": {}}
        for phase, (lo, hi) in DEPTH_PHASES.items():
            depth = [t["output_tokens"] for t in model_turns if lo <= t["round"] <= hi]
            row["phases"][phase] = {"mean": round(mean(depth), 1), "max": max(depth)} if depth else None
        depth = [t["output_tokens"] for t in model_turns]
        row["overall"] = {"mean": round(mean(depth), 1), "max": max(depth)}
        by_round = defaultdict(list)
        for t in model_turns:
            by_round[t["round"]].append(t["output_tokens"])
        row["per_round"] = [
            {"round": r, "mean": round(mean(v), 1), "max": max(v)} for r, v in sorted(by_round.items())
        ]
        rows.append(row)
    return rows


def print_depth_table(records: list[dict]):
    rows = depth_rows(records)
    if not rows:
        return
    table = Table(title="Planning depth: generated tokens per move (mean / max)")
    for col in ["Model", "Moves", *DEPTH_PHASES, "Overall"]:
        table.add_column(col)
    for row in rows:
        cells = [
            f"{p['mean']:.0f} / {p['max']}" if p else "-" for p in row["phases"].values()
        ]
        overall = row["overall"]
        table.add_row(row["model"], str(row["moves"]), *cells,
                      f"{overall['mean']:.0f} / {overall['max']}")
    console.print(table)


# ----------------------------------------------------------------------------
# Reasoning types (Section V-D, Table III, Figure 4)
# ----------------------------------------------------------------------------
def reason_rows(rows: list[dict]) -> list[dict]:
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    result = []
    for model, model_rows_ in by_model.items():
        counts = Counter(r["label"] for r in model_rows_)
        by_round = defaultdict(list)
        for r in model_rows_:
            by_round[r["round"]].append(r["label"])
        result.append({
            "model": model,
            "moves": len(model_rows_),
            **{REASON_SHORT[l]: rate(counts[l], len(model_rows_)) for l in REASON_LABELS},
            "per_round": [
                {
                    "round": round_idx,
                    "moves": len(labels),
                    **{REASON_SHORT[l]: rate(Counter(labels)[l], len(labels)) for l in REASON_LABELS},
                }
                for round_idx, labels in sorted(by_round.items())
            ],
        })
    return result


def print_reason_tables(rows: list[dict]):
    result = reason_rows(rows)
    table = Table(title="Table III: Distribution of reasoning types (%)")
    for col in ["Model", "Moves", "STG", "LTS", "AMB"]:
        table.add_column(col)
    for row in result:
        table.add_row(row["model"], str(row["moves"]), fmt(row["STG"]), fmt(row["LTS"]), fmt(row["AMB"]))
    console.print(table)

    for row in result:
        table = Table(title=f"Reasoning types per round (%): {row['model']}")
        for col in ["Round", "Moves", "STG", "LTS", "AMB"]:
            table.add_column(col)
        for r in row["per_round"]:
            table.add_row(str(r["round"]), str(r["moves"]), fmt(r["STG"]), fmt(r["LTS"]), fmt(r["AMB"]))
        console.print(table)
