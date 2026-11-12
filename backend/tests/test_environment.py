import random
import unittest

from o_an_quan.game import Environment


def empty_board(env: Environment, round_idx: int = 5):
    env.game_state["round"] = round_idx
    board = env.game_state["board"]
    for pos in board:
        board[pos] = []
    board["QA"] = ["mandarin_a"]
    board["QB"] = ["mandarin_b"]
    return board


def peasants_on_board(env: Environment) -> int:
    return sum(t.startswith("peasant") for cell in env.game_state["board"].values() for t in cell)


def mandarins_on_board(env: Environment) -> int:
    return sum(t.startswith("mandarin") for cell in env.game_state["board"].values() for t in cell)


class TestMoves(unittest.TestCase):
    def setUp(self):
        self.env = Environment(verbose=False)
        self.env.start_round()

    def test_invalid_moves_are_rejected(self):
        for action, team in [
            ({"pos": "B1", "way": "clockwise"}, "A"),   # opponent's cell
            ({"pos": "QA", "way": "clockwise"}, "A"),   # mandarin cell
            ({"pos": "A1", "way": "up"}, "A"),          # unknown direction
        ]:
            steps, _, _ = self.env.commit_action(action, team)
            self.assertTrue(steps[0].startswith("[error]"), action)

    def test_capture_chain_is_credited_to_mover(self):
        board = empty_board(self.env)
        board["QB"] += ["peasant_b"] * 5
        board["A3"] = ["peasant_a"]
        board["B4"] = ["peasant_b"] * 2
        board["B2"] = ["peasant_a"]
        # A3 -> A4; A5 empty: capture QB (10 + 5); B5 empty: B4 (2); B3 empty: B2 (1)
        self.env.commit_action({"pos": "A3", "way": "clockwise"}, "A")
        self.assertEqual(self.env.game_state["score"], {"A": 18, "B": 0})

    def test_turn_ends_on_mandarin_cell(self):
        board = empty_board(self.env)
        board["QB"] += ["peasant_b"] * 5
        board["A4"] = ["peasant_a"]
        _, events, _ = self.env.commit_action({"pos": "A4", "way": "clockwise"}, "A")
        self.assertIn({"type": "end_turn", "reason": "HIT_MANDARIN_CELL"}, events)
        self.assertEqual(self.env.game_state["board"]["QB"], ["mandarin_b"] + ["peasant_b"] * 5)

    def test_chain_stops_on_non_empty_cell_after_capture(self):
        board = empty_board(self.env)
        board["A1"] = ["peasant_a"]
        board["A4"] = ["peasant_a"] * 3
        board["A5"] = ["peasant_a"] * 4
        # A1 -> A2; A3 empty: capture A4; next A5 non-empty -> turn ends (E5)
        _, events, _ = self.env.commit_action({"pos": "A1", "way": "clockwise"}, "A")
        self.assertIn({"type": "end_turn", "reason": "CHAIN_BROKEN"}, events)
        self.assertEqual(self.env.game_state["score"]["A"], 3)
        self.assertEqual(len(self.env.game_state["board"]["A5"]), 4)

    def test_e1_immature_mandarin(self):
        board = empty_board(self.env)
        board["QB"] += ["peasant_b"] * 4
        board["A3"] = ["peasant_a"]
        _, events, _ = self.env.commit_action({"pos": "A3", "way": "clockwise"}, "A")
        self.assertTrue(any(e.get("reason") == "IMMATURE_MANDARIN" for e in events))
        self.assertEqual(self.env.game_state["score"]["A"], 0)

    def test_e1_does_not_apply_once_mandarin_is_gone(self):
        board = empty_board(self.env)
        board["QB"] = ["peasant_b"] * 2
        board["A3"] = ["peasant_a"]
        self.env.commit_action({"pos": "A3", "way": "clockwise"}, "A")
        self.assertEqual(self.env.game_state["score"]["A"], 2)

    def test_e3_early_game_restriction(self):
        board = empty_board(self.env, round_idx=2)
        board["QB"] += ["peasant_b"] * 9
        board["A3"] = ["peasant_a"]
        _, events, _ = self.env.commit_action({"pos": "A3", "way": "clockwise"}, "A")
        self.assertTrue(any(e.get("reason") == "EARLY_GAME_RESTRICTION" for e in events))

    def test_e4_two_empty_cells_end_turn(self):
        board = empty_board(self.env)
        board["A1"] = ["peasant_a"]
        board["B5"] = ["peasant_b"]
        _, events, _ = self.env.commit_action({"pos": "A1", "way": "clockwise"}, "A")
        self.assertIn({"type": "end_turn", "reason": "TWO_EMPTY"}, events)


class TestRestoreAndEnd(unittest.TestCase):
    def setUp(self):
        self.env = Environment(verbose=False)

    def test_e2_restore_costs_five(self):
        board = empty_board(self.env)
        board["B1"] = ["peasant_b"]
        self.env.game_state["score"]["A"] = 7
        can_continue, _ = self.env.restore_peasants("A")
        self.assertTrue(can_continue)
        self.assertEqual(self.env.game_state["score"]["A"], 2)
        self.assertEqual(self.env.get_available_pos("A"), ["A1", "A2", "A3", "A4", "A5"])

    def test_e2_cannot_restore_loses(self):
        board = empty_board(self.env)
        board["B1"] = ["peasant_b"]
        self.env.game_state["score"]["A"] = 4
        can_continue, _ = self.env.restore_peasants("A")
        self.assertFalse(can_continue)
        self.assertEqual(self.env.is_end(), (True, "END_BY_NO_RESTORE_A"))
        self.assertEqual(self.env.get_winner("END_BY_NO_RESTORE_A"), "B")

    def test_finalize_collects_remaining_peasants(self):
        board = empty_board(self.env)
        board["QA"], board["QB"] = [], []
        board["A1"] = ["peasant_b"] * 3
        board["B2"] = ["peasant_a"]
        self.assertEqual(self.env.is_end(), (True, "END_BY_NO_MANDARINS"))
        self.assertEqual(self.env.finalize("END_BY_NO_MANDARINS"), {"A": 3, "B": 1})


class TestRandomPlay(unittest.TestCase):
    def test_rules_and_conservation_hold_in_random_games(self):
        """Peasants on board + points - 10 * captured mandarins is always 50."""
        for seed in range(300):
            rng = random.Random(seed)
            env = Environment(verbose=False)
            for _ in range(25):
                round_idx = env.start_round()
                ended = False
                for team in "AB":
                    if not env.restore_peasants(team)[0]:
                        ended = True
                        break
                    pos = rng.choice(env.get_available_pos(team))
                    way = rng.choice(["clockwise", "counter_clockwise"])
                    steps, events, is_end = env.commit_action({"pos": pos, "way": way}, team)
                    self.assertFalse(steps[0].startswith("[error]"))

                    for e in events:
                        if e["type"] == "capture":
                            self.assertEqual(e["team"], team)
                            if any(t.startswith("mandarin") for t in e["pieces"]):
                                self.assertGreater(round_idx, 2)  # E3
                                self.assertGreaterEqual(
                                    sum(t.startswith("peasant") for t in e["pieces"]), 5
                                )  # E1

                    score = env.game_state["score"]
                    captured_mandarins = 2 - mandarins_on_board(env)
                    self.assertEqual(
                        peasants_on_board(env) + score["A"] + score["B"] - 10 * captured_mandarins,
                        50,
                    )
                    if is_end:
                        ended = True
                        break
                if ended:
                    break


if __name__ == "__main__":
    unittest.main()
