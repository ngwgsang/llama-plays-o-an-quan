"""
Prompt components used by the agents (see paper, Section IV and Appendix VII-B).

LLM_theta(G, H, R, P) -> (reason, action)
    G: current board state      -> Environment.render_state()
    H: previous-turn history    -> game.runner.build_history()
    R: rule specification       -> RULES             (templates/rules.md)
    P: persona                  -> PERSONAS          (personas/*.md)

All prompts live in markdown files next to this module; templates use
str.format placeholders such as {team} or {think}.
"""
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
TEMPLATES_DIR = PROMPTS_DIR / "templates"
PERSONAS_DIR = PROMPTS_DIR / "personas"


def load_template(name: str) -> str:
    return (TEMPLATES_DIR / f"{name}.md").read_text(encoding="utf-8").strip() + "\n"


# The paper names five personas but does not publish their prompts; the files
# in personas/ are written to match each name (file name = persona name).
def load_personas(directory: Path = PERSONAS_DIR) -> dict[str, str]:
    return {
        path.stem: path.read_text(encoding="utf-8").strip()
        for path in sorted(directory.glob("*.md"))
    }


SYSTEM_PROMPT = load_template("system").strip()
RULES = load_template("rules")
# Appendix VII-B: Zero-shot Decision-Making Prompt
DECISION_PROMPT = load_template("decision")
# Appendix VII-B: Zero-shot Reasoning-Type Classification Prompt
CLASSIFICATION_PROMPT = load_template("classification")
PERSONAS = load_personas()

__all__ = [
    "SYSTEM_PROMPT",
    "RULES",
    "DECISION_PROMPT",
    "CLASSIFICATION_PROMPT",
    "PERSONAS",
    "load_template",
    "load_personas",
]
