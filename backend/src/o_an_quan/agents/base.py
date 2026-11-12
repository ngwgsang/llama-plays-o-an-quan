from dataclasses import dataclass, asdict
from typing import Any, List, Protocol


@dataclass
class TurnResult:
    """(reason, action) produced by an agent for one move, plus bookkeeping."""
    pos: str
    way: str
    reason: str
    observation: str = ""
    invalid_attempts: int = 0   # illegal or unparseable outputs before a legal one
    fallback: bool = False      # True if a random legal move was used instead
    output_tokens: int = 0      # tokens generated for this move (planning depth)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Player(Protocol):
    """Interface shared by every agent the game runner can drive."""
    team: str
    name: str
    model: str

    def play_turn(self, game_state: Any, available_positions: List[str], history: str = "") -> TurnResult:
        ...
