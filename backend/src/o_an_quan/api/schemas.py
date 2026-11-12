from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..config import MAX_ROUNDS


class AgentConfig(BaseModel):
    type: Literal["random", "llm"] = "random"
    model: Optional[str] = Field(None, description="OpenRouter model id (LLM agents)")
    persona: Optional[str] = Field(None, description="Persona name from prompts/personas")


class NewGameRequest(BaseModel):
    agent_a: AgentConfig
    agent_b: AgentConfig
    max_rounds: int = Field(MAX_ROUNDS, ge=1, le=100)
    seed: Optional[int] = None
