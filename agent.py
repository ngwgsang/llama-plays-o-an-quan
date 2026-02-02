import json

from typing import Dict, List, Any, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv
from os import getenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig


from utils import log_tool
from environment import Enviroment


load_dotenv()

llm = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini-2024-07-18"
    # model="meta-llama/llama-3.1-8b-instruct"
)

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
    current_game_state: dict
    current_available_positions: list[str]

enviroment = Enviroment()

@tool
@log_tool
def see_gameboard(current_game_state: str, current_available_positions: str, player_team: str) -> str:
    """
    [Observe | FIRST]
    Observe the current game board and determine all legal positions
    available for the specified player team.

    This tool MUST be called before any planning or action.
    Use it to understand the board state only. Do not decide a move here.
    """
    return f"{current_game_state} \n Available Positions: {current_available_positions}"

@tool
@log_tool
def plan_the_strategy(player_team: str) -> str:
    """    
    [Reason | AFTER OBSERVE]
    Analyze the observed game state and explain the strategic intention
    for the NEXT move.

    This tool SHOULD be called only AFTER observing the board.
    Do NOT execute any action here. Provide reasoning only.
    """
    response = llm.invoke(f"""
    ** Current Game State **
    You are team {player_team}.
            
    Task:
    Briefly explain the strategic idea for the NEXT move ( position and scatter direction ).

    Rules:
    - Max 2 sentences
    - No listing
    - No repetition
    - No emojis
    - Do NOT restate the board
    - Focus on intent (capture, deny opponent, setup)

    Return explanation only.
    """)
    return response.text

@tool
@log_tool
def scat_and_capture(pos: str, way: str, player_team: str) -> str:
    """
    [Action | FINAL]
    Execute the chosen move by scattering and capturing from the given position
    in the given direction.

    This action FINALIZES the current turn.
    After calling this tool, the agent MUST stop reasoning and produce no further actions.
    """
    return f"Scattered and captured from position {pos} in direction {way} by team {player_team}"

class Agent:
    def __init__(self, team: str, persona: str = "", model_endpoint: str = "openai/gpt-4o-mini-2024-07-18"):
        self.team = team
        
        llm = ChatOpenAI(
            api_key=getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_endpoint
            # model="meta-llama/llama-3.1-8b-instruct"
        )
        self.persona = persona
        self.agent = create_agent(
            llm, 
            tools=[
                see_gameboard,
                plan_the_strategy,
                scat_and_capture
            ],
            state_schema=PlayerState,
            system_prompt="You are an intelligent agent playing the traditional Vietnamese game \"Ô Ăn Quan\".",
            response_format=PlayerAgentOutput,
        )
    
    def play_turn(self, game_state: Dict[str, Any], available_positions: List[str]) -> PlayerAgentOutput:
        response = self.agent.invoke({
            "messages": [
                {
                    "role": "user", 
                    "content": f"""
                        You are Player Team {self.team} in the game "Ô Ăn Quan".
                    
                        **Persona**
                        {self.persona}
                        
                        **Game Rules**
                        - Do not start from mandarin cells (Q1, Q2)
                        - Only pick from your own side (A-side for team A, B-side for team B)
                        - You must pick a position that has at least one piece
                        - The game ends when a player cannot restore peasants
                        
                        **Task**
                        Based on the above rules and current game state, think about:
                        - Which position should you pick to scatter from?
                        - Which direction to scatter?
                        
                        ** Current Game State **
                        {game_state}
                    """
                }
            ],        
            "player_team": self.team,
            "current_game_state": str(game_state),
            "current_available_positions": str(available_positions)
        })
        return response["structured_response"]