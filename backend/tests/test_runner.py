import json
import os
import tempfile
import unittest

from o_an_quan.agents import RandomAgent
from o_an_quan.experiments.io import append_jsonl, load_jsonl
from o_an_quan.experiments.reports import llm_turns
from o_an_quan.game import build_history, play_game


class TestPlayGame(unittest.TestCase):
    def test_record_is_complete_and_serialisable(self):
        record = play_game(RandomAgent("A", seed=1), RandomAgent("B", seed=2), verbose=False)
        json.dumps(record)
        self.assertIn(record["winner"], {"A", "B", "DRAW"})
        self.assertLessEqual(record["rounds_played"], 25)
        self.assertEqual(len(record["round_scores"]), record["rounds_played"])
        self.assertEqual(record["turns"][0]["team"], "A")
        for turn in record["turns"]:
            self.assertTrue(turn["pos"].startswith(turn["team"]))

    def test_max_rounds(self):
        record = play_game(RandomAgent("A", seed=3), RandomAgent("B", seed=4),
                           max_rounds=1, verbose=False)
        self.assertLessEqual(record["rounds_played"], 1)

    def test_history_contains_both_moves_and_reasons(self):
        mine = {"pos": "A2", "way": "clockwise", "reason": "set up", "captured": 0,
                "score": {"A": 0, "B": 0}}
        theirs = {"pos": "B3", "way": "counter_clockwise", "reason": "grab", "captured": 2}
        history = build_history(mine, theirs)
        for text in ["A2 clockwise", "set up", "B3 counter_clockwise", "grab"]:
            self.assertIn(text, history)
        self.assertEqual(build_history(None, None), "")


class TestResultsIO(unittest.TestCase):
    def test_jsonl_roundtrip_and_llm_turn_filter(self):
        record = play_game(RandomAgent("A", seed=5), RandomAgent("B", seed=6), verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out", "games.jsonl")
            append_jsonl(path, record)
            append_jsonl(path, record)
            loaded = load_jsonl([path])
        self.assertEqual(len(loaded), 2)
        self.assertEqual(llm_turns(loaded), [])  # random agents are excluded


if __name__ == "__main__":
    unittest.main()
