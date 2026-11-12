"""Structured outputs expected from the LLMs."""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from langchain.agents import AgentState


class DirectionOutput(str, Enum):
    CLOCKWISE = "clockwise"
    COUNTER_CLOCKWISE = "counter_clockwise"

class PositionOutput(str, Enum):
    A1 = "A1"; A2 = "A2"; A3 = "A3"; A4 = "A4"; A5 = "A5"
    B1 = "B1"; B2 = "B2"; B3 = "B3"; B4 = "B4"; B5 = "B5"

class ActionOutput(BaseModel):
    pos: PositionOutput = Field(description="The position chosen as the starting pit")
    way: DirectionOutput = Field(description="The direction selected for sowing")

class PlayerAgentOutput(BaseModel):
    observation: str = Field(description="A brief summary of the environment perceived before reasoning")
    reason: str = Field(description="The agent's reasoning after observing the environment and deciding on the final action")
    action: ActionOutput = Field(description="The action selected by the agent")

class PlayerState(AgentState):
    player_team: str
    current_game_state: str
    current_available_positions: str


class ReasonTypeOutput(BaseModel):
    label: Literal["SHORT_TERM_GAIN", "LONG_TERM_STRATEGY", "AMBIGUOUS"] = Field(
        description="The reasoning type"
    )
    transcription: str = Field(
        description="The reasoning with the parts that led to the label highlighted in **bold**"
    )
