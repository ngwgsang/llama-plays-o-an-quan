"""
Reproduce the experiments of the paper (Section V).

  python -m o_an_quan.experiments personas --model meta-llama/llama-3.3-70b-instruct   # Table I
  python -m o_an_quan.experiments models                                               # Table II
  python -m o_an_quan.experiments classify --input results/models.jsonl                # Table III
  python -m o_an_quan.experiments report --input results/personas.jsonl results/models.jsonl

Every finished game is appended to a JSONL file, so `report` can be re-run
on partial results.
"""
import argparse
import os

from ..config import (
    CLASSIFIER_MODEL, GAMES_PER_MATCHUP, LLAMA_MODELS, RESULTS_DIR, STRONGEST_MODEL,
)
from ..prompts import PERSONAS
from ..utils import set_verbose
from .classify import classify_reasoning
from .io import load_jsonl
from .reports import print_depth_table, print_model_table, print_persona_table, print_reason_tables
from .runners import run_models, run_personas


def cmd_personas(args):
    run_personas(args.model, args.personas, args.out, args.games, args.verbose)
    print_persona_table(load_jsonl([args.out]))


def cmd_models(args):
    run_models(args.models, args.opponent, args.persona, args.out,
               args.games, args.opponent_first, args.verbose)
    print_model_table(load_jsonl([args.out]))


def cmd_classify(args):
    classify_reasoning(load_jsonl(args.input), args.out, args.classifier, args.concurrency)
    print_reason_tables(load_jsonl([args.out]))


def cmd_report(args):
    records = load_jsonl(args.input)
    print_persona_table(records)
    print_model_table(records)
    print_depth_table(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m o_an_quan.experiments",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", help="print every tool call and move")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("personas", help="persona agents vs RandomAgent (Table I)")
    p.add_argument("--model", default=STRONGEST_MODEL)
    p.add_argument("--personas", nargs="+", default=list(PERSONAS), choices=list(PERSONAS))
    p.add_argument("--games", type=int, default=GAMES_PER_MATCHUP)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "personas.jsonl"))
    p.set_defaults(func=cmd_personas)

    p = sub.add_parser("models", help="LLMs vs the strongest model (Table II)")
    p.add_argument("--models", nargs="+", default=LLAMA_MODELS)
    p.add_argument("--opponent", default=STRONGEST_MODEL)
    p.add_argument("--persona", default="BalancedAgent", choices=list(PERSONAS))
    p.add_argument("--opponent-first", action="store_true", help="let the opponent play as team A")
    p.add_argument("--games", type=int, default=GAMES_PER_MATCHUP)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "models.jsonl"))
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("classify", help="classify reasoning types (Table III)")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--classifier", default=CLASSIFIER_MODEL)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "reasoning_types.jsonl"))
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("report", help="print Table I, Table II and planning depth from saved games")
    p.add_argument("--input", nargs="+", required=True)
    p.set_defaults(func=cmd_report)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_verbose(args.verbose)
    args.func(args)
