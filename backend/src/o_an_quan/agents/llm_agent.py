import random
from typing import Any, List, Optional

from langchain.agents import create_agent

from ..config import DEFAULT_MODEL
from ..prompts import SYSTEM_PROMPT, RULES, DECISION_PROMPT
from .base import TurnResult
from .schemas import PlayerAgentOutput, PlayerState
from .tools import build_tools, make_llm


class Agent:
    """
    LLM-based player: LLM_theta(G, H, R, P) -> (reason, action).
    """

    def __init__(
        self,
        team: str,
        persona: str = "",
        model_endpoint: str = DEFAULT_MODEL,
        name: Optional[str] = None,
        max_retries: int = 2,
        seed: Optional[int] = None,
    ):
        self.team = team
        self.persona = persona
        self.model = model_endpoint
        self.name = name or model_endpoint
        self.max_retries = max_retries
        self.rng = random.Random(seed)

        llm = make_llm(model_endpoint)
        self.agent = create_agent(
            llm,
            tools=build_tools(llm),
            state_schema=PlayerState,
            system_prompt=SYSTEM_PROMPT,
            response_format=PlayerAgentOutput,
        )

    def play_turn(self, game_state: Any, available_positions: List[str], history: str = "") -> TurnResult:
        """
        Ask the LLM for a move. Illegal moves are retried up to `max_retries`
        times; after that a random legal move is played and flagged as fallback.
        """
        prompt = DECISION_PROMPT.format(
            team=self.team,
            persona=self.persona,
            game_state=game_state,
            available_positions=", ".join(available_positions),
            history=history or "This is my first move.",
            rules=RULES,
        )

        invalid_attempts = 0
        output_tokens = 0
        error = ""
        content = prompt
        for _ in range(self.max_retries + 1):
            try:
                response = self.agent.invoke({
                    "messages": [{"role": "user", "content": content}],
                    "player_team": self.team,
                    "current_game_state": str(game_state),
                    "current_available_positions": ", ".join(available_positions),
                })
            except Exception as e:  # provider / parsing errors count as invalid outputs
                invalid_attempts += 1
                error = f"{type(e).__name__}: {e}"
                continue

            output_tokens += sum(
                (getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0)
                for m in response["messages"]
            )
            output: PlayerAgentOutput = response["structured_response"]
            pos, way = output.action.pos.value, output.action.way.value
            if pos in available_positions:
                return TurnResult(
                    pos=pos,
                    way=way,
                    reason=output.reason,
                    observation=output.observation,
                    invalid_attempts=invalid_attempts,
                    output_tokens=output_tokens,
                    error=error,
                )

            invalid_attempts += 1
            error = f"Illegal position {pos}"
            content = (
                f"{prompt}\n---\n**Warning**\nYour previous choice {pos} is illegal. "
                f"You MUST choose one of: {', '.join(available_positions)}."
            )

        return TurnResult(
            pos=self.rng.choice(available_positions),
            way=self.rng.choice(["clockwise", "counter_clockwise"]),
            reason="",
            invalid_attempts=invalid_attempts,
            fallback=True,
            output_tokens=output_tokens,
            error=error,
        )
