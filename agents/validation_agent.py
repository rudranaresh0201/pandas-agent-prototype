import logging
from typing import Any, Optional, Tuple

import pandas as pd

from core.query_plan import QueryPlan

logger = logging.getLogger(__name__)


class ValidationAgent:
    def validate(
        self, result: Any, query_plan: QueryPlan
    ) -> Tuple[bool, str]:
        if result is None:
            return False, "Result is None"

        expected = query_plan.expected_output_type
        rules = query_plan.validation_rules

        ok, reason = self._check_type(result, expected)
        if not ok:
            return False, reason

        ok, reason = self._check_semantic_columns(query_plan.suggested_code, query_plan)
        if not ok:
            return False, reason

        ok, reason = self._check_rules(result, rules, query_plan)
        if not ok:
            return False, reason

        return True, "OK"

    def _check_type(self, result: Any, expected: str) -> Tuple[bool, str]:
        if expected == "scalar":
            if isinstance(result, (int, float, str, bool)):
                return True, "OK"
            if isinstance(result, pd.Series) and len(result) == 1:
                return True, "OK"
            return False, f"Expected scalar, got {type(result).__name__}"

        if expected == "series":
            if isinstance(result, (pd.Series, list)):
                return True, "OK"
            return False, f"Expected series, got {type(result).__name__}"

        if expected == "dataframe":
            if isinstance(result, pd.DataFrame):
                return True, "OK"
            return False, f"Expected DataFrame, got {type(result).__name__}"

        if expected == "string":
            return True, "OK"  # anything can be stringified

        return True, "OK"

    def _check_semantic_columns(
        self, code: str, plan: QueryPlan
    ) -> Tuple[bool, str]:
        """Verify generated code uses the planned target columns."""
        if not plan.target_columns:
            return True, "OK"

        for col in plan.target_columns:
            if col and col not in code:
                return False, (
                    f"Generated code does not use planned column "
                    f"'{col}'. Possible semantic mismatch."
                )
        return True, "OK"

    def _check_rules(
        self, result: Any, rules: dict, plan: QueryPlan
    ) -> Tuple[bool, str]:
        if not rules:
            return True, "OK"

        # Count / numeric range check
        vmin: Optional[float] = rules.get("min")
        vmax: Optional[float] = rules.get("max")

        try:
            numeric_val: Optional[float] = None
            if isinstance(result, (int, float)):
                numeric_val = float(result)
            elif isinstance(result, pd.Series) and len(result) == 1:
                numeric_val = float(result.iloc[0])

            if numeric_val is not None:
                if vmin is not None and numeric_val < vmin:
                    return False, f"Result {numeric_val} below min {vmin}"
                if vmax is not None and numeric_val > vmax:
                    return False, f"Result {numeric_val} above max {vmax}"
        except (TypeError, ValueError):
            pass

        # Categorical membership check
        allowed_values = rules.get("allowed_values")
        if allowed_values and isinstance(result, str):
            if result not in allowed_values:
                logger.warning(f"Result '{result}' not in allowed values")

        return True, "OK"
