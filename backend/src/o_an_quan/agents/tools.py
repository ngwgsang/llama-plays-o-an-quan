from langchain_openai import ChatOpenAI
from langchain.tools import tool

from ..config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from ..utils import log_tool


def make_llm(model_endpoint: str) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=model_endpoint,
    )


def build_tools(llm: ChatOpenAI) -> list:
    """Create the agent's tools bound to its own LLM."""

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
    def plan_the_strategy(current_game_state: str, player_team: str) -> str:
        """
        [Reason | AFTER OBSERVE]
        Analyze the observed game state and explain the strategic intention
        for the NEXT move.

        This tool SHOULD be called only AFTER observing the board.
        Do NOT execute any action here. Provide reasoning only.
        """
        response = llm.invoke(f"""
        ** Current Game State **
        {current_game_state}

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

    return [see_gameboard, plan_the_strategy, scat_and_capture]
