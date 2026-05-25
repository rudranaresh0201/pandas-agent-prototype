import json
import logging
import re
from typing import Any, Dict

from core.llm_client import LLMClient
from core.query_plan import QueryPlan
from core.schema_profile import SchemaProfile

logger = logging.getLogger(__name__)

_FALLBACK_PLAN: Dict[str, Any] = {
    "target_columns": [],
    "operation": "aggregate",
    "expected_output_type": "scalar",
    "validation_rules": {},
    "suggested_code": "result = df.describe().to_string()",
}

_PROMPT_TEMPLATE = """\
You are a pandas query planner. Given a DataFrame schema and a question, produce an execution plan.

Schema:
{schema_json}

Question: {question}

Respond with ONLY a valid JSON object — no markdown fences, no explanation:
{{
  "target_columns": ["col1", "col2"],
  "operation": "filter|groupby|aggregate|sort|count|rank",
  "expected_output_type": "scalar|series|dataframe|string",
  "validation_rules": {{}},
  "suggested_code": "result = df[...]"
}}

Rules for suggested_code:
- DataFrame already loaded as `df`; do NOT import anything
- Assign the final answer to `result`
- Keep code concise and correct for pandas 2.x
"""


class QueryPlannerAgent:
    def __init__(self):
        self.llm = LLMClient()

    def plan(self, question: str, schema_profile: SchemaProfile) -> QueryPlan:
        schema_dict = schema_profile.to_dict() if schema_profile else {}
        schema_json = json.dumps(schema_dict, indent=2, default=str)

        prompt = _PROMPT_TEMPLATE.format(
            schema_json=schema_json[:3000],  # truncate to avoid token overflow
            question=question,
        )

        try:
            raw = self.llm.call(prompt, temperature=0, max_tokens=600)
            plan_dict = self._parse_json(raw)
        except Exception as exc:
            logger.warning(f"QueryPlanner LLM failed: {exc} — using fallback plan")
            plan_dict = _FALLBACK_PLAN.copy()
            plan_dict["suggested_code"] = self._fallback_code(question, schema_profile)

        return QueryPlan(
            question=question,
            target_columns=plan_dict.get("target_columns", []),
            operation=plan_dict.get("operation", "aggregate"),
            expected_output_type=plan_dict.get("expected_output_type", "scalar"),
            validation_rules=plan_dict.get("validation_rules", {}),
            suggested_code=self._clean_code(plan_dict.get("suggested_code", "")),
        )

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\n?", "", raw)
        cleaned = re.sub(r"```", "", cleaned).strip()

        # Find the first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response")

        return json.loads(match.group())

    def _clean_code(self, code: str) -> str:
        code = re.sub(r"```(?:python)?\n?", "", code)
        code = re.sub(r"```", "", code)
        return code.strip()

    def _fallback_code(self, question: str, schema: SchemaProfile) -> str:
        cols = list(schema.columns.keys()) if schema else []
        first_num = next(
            (c for c, p in schema.columns.items() if p.semantic_type in ("numeric", "currency", "percentage")),
            cols[0] if cols else "value",
        ) if schema else "value"
        return f"result = df['{first_num}'].describe().to_string()"
