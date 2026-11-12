"""Zero-shot reasoning-type classification (paper, Section V-D)."""
from ..agents import make_llm
from ..agents.schemas import ReasonTypeOutput
from ..config import CLASSIFIER_MODEL
from ..prompts import CLASSIFICATION_PROMPT
from .io import append_jsonl
from .reports import llm_turns


def classify_reasoning(records: list[dict], out: str,
                       classifier_model: str = CLASSIFIER_MODEL, concurrency: int = 8):
    """Label every LLM move's reasoning and append the labels to `out`."""
    turns = [t for t in llm_turns(records) if t["reason"]]
    classifier = make_llm(classifier_model).with_structured_output(ReasonTypeOutput)
    prompts = [CLASSIFICATION_PROMPT.format(think=t["reason"]) for t in turns]
    outputs = classifier.batch(prompts, config={"max_concurrency": concurrency},
                               return_exceptions=True)

    for turn, output in zip(turns, outputs):
        ok = isinstance(output, ReasonTypeOutput)
        append_jsonl(out, {
            "model": turn["model"],
            "round": turn["round"],
            "reason": turn["reason"],
            "label": output.label if ok else "AMBIGUOUS",
            "transcription": output.transcription if ok else "",
            "error": "" if ok else str(output),
        })
