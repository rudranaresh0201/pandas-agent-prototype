import logging
from typing import Any, Optional

from core.llm_client import LLMClient
from core.query_plan import QueryPlan

logger = logging.getLogger(__name__)

_PERSONA_STYLES = {
    "General": (
        "Explain the answer clearly and concisely for a general audience. "
        "Avoid jargon; focus on what the number means in plain English."
    ),
    "Risk Analyst": (
        "Frame the answer in terms of risk exposure and statistical confidence. "
        "Highlight outliers, thresholds breached, and portfolio-level implications."
    ),
    "Student": (
        "Explain in the simplest terms possible, as if teaching a beginner. "
        "Use a relatable analogy and break down the finding step by step."
    ),
    "Compliance Officer": (
        "Emphasize regulatory compliance, reporting obligations, and any thresholds "
        "that trigger required action. Be precise and formal in tone."
    ),
}

_PROMPT = """\
You answered a data question. Generate a 2-3 sentence explanation tailored for the specified persona.

IMPORTANT: Always start your response with the actual computed number or result from Raw Answer. Never explain without stating the value first.

Question: {question}
Raw Answer: {result}

Persona: {persona}
Style guide: {style}

Respond with ONLY the explanation — no preamble, no labels.
"""


class InsightAgent:
    def __init__(self):
        self.llm = LLMClient()

    def generate(
        self,
        result: Any,
        query_plan: Optional[QueryPlan],
        persona: str = "General",
    ) -> str:
        question = query_plan.question if query_plan else "the data question"
        style = _PERSONA_STYLES.get(persona, _PERSONA_STYLES["General"])

        prompt = _PROMPT.format(
            question=question,
            result=str(result)[:500],
            persona=persona,
            style=style,
        )

        try:
            return self.llm.call(prompt, temperature=0.3, max_tokens=300)
        except Exception as exc:
            logger.error(f"Insight generation failed: {exc}")
            return f"Answer: {result}"
