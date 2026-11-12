import json
import tempfile
import unittest
from pathlib import Path

from o_an_quan.agents import RandomAgent
from o_an_quan.game import play_game, write_game_log
from o_an_quan.game.logger import LOG_SCHEMA


class TestGameLog(unittest.TestCase):
    def test_log_is_valid_json_with_events(self):
        record = play_game(RandomAgent("A", seed=0), RandomAgent("B", seed=1), verbose=False, record_events=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_game_log(record, tmp, {"seed": 0})
            self.assertEqual(path.parent, Path(tmp))
            self.assertEqual(path.suffix, ".json")
            log = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(log["schema"], LOG_SCHEMA)
        self.assertEqual(log["config"], {"seed": 0})
        self.assertEqual(log["final_score"], record["final_score"])
        first = log["turns"][0]
        self.assertEqual(first["events"][0], {"type": "pickup", "pos": first["pos"], "peasants": 5})
        drops = [e for e in first["events"] if e["type"] == "drop"]
        self.assertGreaterEqual(len(drops), 5)
        self.assertIn("restored", first)

    def test_events_are_off_by_default(self):
        record = play_game(RandomAgent("A", seed=0), RandomAgent("B", seed=1), verbose=False)
        self.assertNotIn("events", record["turns"][0])


if __name__ == "__main__":
    unittest.main()
