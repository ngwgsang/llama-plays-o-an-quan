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

from typing import Any

from functools import wraps

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


from enviroment import Enviroment


console = Console()

def log_tool(tool_func):
    @wraps(tool_func)
    def wrapper(*args, **kwargs):
        # --- format input ---
        args_str = json.dumps(args, ensure_ascii=False)
        kwargs_str = json.dumps(kwargs, ensure_ascii=False)

        # gọi tool
        result = tool_func(*args, **kwargs)

        result_str = str(result)

        # --- nội dung gộp ---
        body = Text()
        body.append("ARGS:\n", style="bold cyan")
        body.append(f"{args_str}\n\n")

        body.append("KWARGS:\n", style="bold cyan")
        body.append(f"{kwargs_str}\n\n")

        body.append("RESULT:\n", style="bold green")
        body.append(result_str)

        console.print(
            Panel(
                body,
                title=f"Agent :: {tool_func.__name__}",
                border_style="green",
                width=80
            )
        )

        return result

    return wrapper


load_dotenv()

llm = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini-2024-07-18"
    # model="meta-llama/llama-3.1-8b-instruct"
)

# --- System Prompt & Models (Giữ nguyên) ---
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
    # current_game_state: Dict[str, Any]
    # current_available_positions: List[str]
    current_game_state: dict
    current_available_positions: list[str]


enviroment = Enviroment()

@tool
@log_tool
def see_gameboard(current_game_state, current_available_positions) -> str:
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
def plan_the_strategy(player_team: Literal["A", "B"] = "A") -> str:
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
def scat_and_capture(pos: str, way: str) -> str:
    """
    [Action | FINAL]
    Execute the chosen move by scattering and capturing from the given position
    in the given direction.

    This action FINALIZES the current turn.
    After calling this tool, the agent MUST stop reasoning and produce no further actions.
    """
    # enviroment.commit_action(action={"pos": pos, "way": way})
    return f"Scattered and captured from position {pos} in direction {way}"


# --- 4. TẠO AGENT ---

# @before_model
# def trim_messages(state: AgentState, runtime: Runtime):
#     messages = state["messages"]
#     if len(messages) <= 4:
#         return None

#     system = messages[0]
#     recent = messages[-4:]

#     return {
#         "messages": [
#             RemoveMessage(id=REMOVE_ALL_MESSAGES),
#             system,
#             *recent
#         ]
#     }

    
# agent = create_agent(
#     llm, 
#     tools=[
#         see_gameboard,
#         plan_the_strategy,
#         scat_and_capture
#     ],
#     state_schema=PlayerState,
#     system_prompt="You are an intelligent agent playing the traditional Vietnamese game \"Ô Ăn Quan\".",
#     response_format=PlayerAgentOutput,
# )

# # --- 5. CHẠY ---
# response = agent.invoke({
#     "messages": [
#         {
#             "role": "user", 
#             "content": """
#                 **Persona**
#                 You are a strategic and thoughtful player who enjoys planning several moves ahead.
                
#                 ** Flow **
#                 Start → Observe → Reason → Action → End
                
#                 **Task**
#                 Based on the above rules and current game state, think about:
#                 - Which position should you pick to scatter from?
#                 - Which direction to scatter?
#             """
#         }
#     ],        
#     "player_team": "A",
# })


class Agent:
    def __init__(self, team: Literal["A", "B"] = "A", persona: str = "", model_endpoint: str = "openai/gpt-4o-mini-2024-07-18"):
        self.team = team
        
        llm = ChatOpenAI(
            api_key=getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_endpoint
            # model="meta-llama/llama-3.1-8b-instruct"
        )

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
                        **Persona**
                        You are a strategic and thoughtful player who enjoys planning several moves ahead.
                        
                        ** Flow **
                        Start → Observe → Reason → Action → End
                        
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
            "current_game_state": game_state,
            "current_available_positions": available_positions
        })
        return response