import random
from typing import Any, List, Optional

from .base import TurnResult


class RandomAgent:
    """Baseline agent that selects legal moves and directions randomly (paper, Section V-A)."""

    def __init__(self, team: str, name: str = "RandomAgent", seed: Optional[int] = None):
        self.team = team
        self.name = name
        self.model = "random"
        self.persona = ""
        self.rng = random.Random(seed)

    def play_turn(self, game_state: Any, available_positions: List[str], history: str = "") -> TurnResult:
        return TurnResult(
            pos=self.rng.choice(available_positions),
            way=self.rng.choice(["clockwise", "counter_clockwise"]),
            reason="Random legal move.",
        )
