import json
import logging
import re
from typing import Any, Dict, Optional

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


class QueryPlannerAgent:
    def __init__(self):
        self.llm = LLMClient()

    def plan(self, question: str, schema_profile: Optional[SchemaProfile]) -> QueryPlan:
        prompt = self._build_planner_prompt(question, schema_profile)

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

    def _build_planner_prompt(self, question: str, schema: Optional[SchemaProfile]) -> str:
        if schema is None:
            return (
                "You are a pandas query planner. DataFrame is loaded as `df`.\n"
                f"Question: {question}\n"
                'Return ONLY valid JSON: {{"target_columns": [], "operation": "aggregate", '
                '"expected_output_type": "scalar", "validation_rules": {{}}, '
                '"suggested_code": "result = df.describe().to_string()"}}'
            )

        col_catalog = []
        for col_name, col_profile in schema.columns.items():
            hint = f"{col_name} ({col_profile.semantic_type}"
            vr = col_profile.value_range
            if vr and "min" in vr and "max" in vr:
                hint += f", range: {vr['min']} to {vr['max']}"
            if col_profile.unique_values <= 10:
                hint += f", values: {col_profile.sample_values[:5]}"
            hint += ")"
            col_catalog.append(hint)

        catalog_str = "\n".join(f"  - {c}" for c in col_catalog)

        return f"""You are a pandas query planner with STRICT column grounding rules.

AVAILABLE COLUMNS (use ONLY these exact names):
{catalog_str}

STRICT RULES:
1. ONLY use column names from the list above — no variations
2. Match the user's business metric to the MOST SEMANTICALLY RELEVANT column — not just any numeric column
3. For grouping: match the user's grouping entity to the correct categorical column
4. If user says "VaR" → use VaR_1Day not Face_Value
5. If user says "mark to market" → use Mark_to_Market not Trade_Price
6. If user says "PnL" → use PnL_Daily or Mark_to_Market
7. If user says "by portfolio type" → group by Portfolio column
8. If user says "by desk" → group by Desk column
9. If user says "by branch" → group by Branch column

Question: {question}

Return ONLY valid JSON — no markdown fences, no explanation:
{{
  "target_columns": ["exact_column_name"],
  "group_by_column": "exact_column_name_or_null",
  "operation": "mean|sum|count|max|min|filter",
  "expected_output_type": "scalar|series|dataframe",
  "suggested_code": "result = df...",
  "validation_rules": {{}}
}}

VERIFY before returning:
- Every column in target_columns exists in the AVAILABLE COLUMNS list
- suggested_code uses only exact column names from the list
- group_by_column matches the user's grouping intent exactly
"""

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
