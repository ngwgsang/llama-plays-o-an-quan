from functools import wraps
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import json

# Shared Rich console instance for structured logging
console = Console()


def log_tool(tool_func):
    """
    Decorator for logging tool invocations in a structured and readable format.

    This decorator:
    - Captures positional and keyword arguments passed to the tool
    - Executes the wrapped tool function
    - Displays inputs and outputs in a Rich Panel for clear observability

    Intended use cases:
    - Debugging agent tool calls
    - Inspecting LLM-to-tool interactions
    - Visual tracing during development and demos

    Parameters:
        tool_func (callable): The tool function to be wrapped.

    Returns:
        callable: Wrapped function with logging side effects.
    """

    @wraps(tool_func)
    def wrapper(*args, **kwargs):
        # --- Serialize inputs for display ---
        # ensure_ascii=False preserves Unicode content (e.g., Vietnamese text)
        args_str = json.dumps(args, ensure_ascii=False)
        kwargs_str = json.dumps(kwargs, ensure_ascii=False)
        
        player_team = kwargs.get("player_team", None)
        border_style = "red" if player_team == "B" else "green"
        
        # --- Execute the tool ---
        result = tool_func(*args, **kwargs)
        result_str = str(result)

        # --- Compose formatted log body ---
        body = Text()
        body.append("ARGS:\n", style="bold cyan")
        body.append(f"{args_str}\n\n")

        body.append("KWARGS:\n", style="bold cyan")
        body.append(f"{kwargs_str}\n\n")

        body.append("RESULT:\n", style="bold green")
        body.append(result_str)

        # --- Render log panel ---
        console.print(
            Panel(
                body,
                title=f"Agent :: {tool_func.__name__}",
                border_style=border_style,
                width=80,
            )
        )

        return result

    return wrapper

def log_action_events(
    title: str,
    animation_events: list[dict],
    border_style: str = "orange1",
):
    """
    Render a single Rich panel that visualizes all animation events
    of one action execution.

    One action = one panel.
    Events are rendered in execution order.
    """
    body = Text()
    body.append("ACTION EVENTS:\n", style="bold orange1")

    for idx, event in enumerate(animation_events, start=1):
        etype = event.get("type")

        # --- PICKUP ---
        if etype == "pickup":
            body.append(
                f"{idx}. PICKUP from {event['pos']} "
                f"({len(event['pieces'])} pieces)\n",
                style="orange1",
            )

        # --- DROP ---
        elif etype == "drop":
            body.append(
                f"{idx}. DROP {event['piece']} "
                f"{event['from_pos']} -> {event['to_pos']}\n",
                style="orange1",
            )

        # --- CAPTURE ---
        elif etype == "capture":
            body.append(
                f"{idx}. CAPTURE at {event['pos']} "
                f"({len(event['pieces'])} pieces, team {event['team']})\n",
                style="bold orange1",
            )

        # --- SCORE UPDATE ---
        elif etype == "score_update":
            score = event["score"]
            body.append(
                f"{idx}. SCORE UPDATE -> A: {score['A']} | B: {score['B']}\n",
                style="bold orange1",
            )

        else:
            body.append(
                f"{idx}. {etype.upper()} {event}\n",
                style="orange1",
            )

    console.print(
        Panel(
            body,
            title=title,
            border_style=border_style,
            width=90,
        )
    )