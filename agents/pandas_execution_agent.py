import logging
import re
from typing import Any, Optional, Tuple

from core.datastore import DataStore
from core.llm_client import LLMClient
from core.query_plan import QueryPlan
from core.safe_executor import safe_exec, validate_ast

logger = logging.getLogger(__name__)

_FIX_PROMPT = """\
Fix this pandas code that failed. Return ONLY the corrected code — no explanation, no markdown.

Question: {question}
Available columns: {columns}
Operation: {operation}

Broken code:
{code}

Error:
{error}

Rules: no imports, assign final answer to `result`, use pandas 2.x syntax.
"""


class PandasExecutionAgent:
    MAX_ATTEMPTS = 3

    def __init__(self):
        self.llm = LLMClient()

    def execute(
        self, query_plan: QueryPlan, datastore: DataStore
    ) -> Tuple[Any, str, Optional[str]]:
        df = datastore.sheets[datastore.primary_sheet]
        code = query_plan.suggested_code
        last_error: Optional[str] = None

        for attempt in range(self.MAX_ATTEMPTS):
            logger.info(f"Execution attempt {attempt + 1}/{self.MAX_ATTEMPTS}")

            valid, reason = validate_ast(code)
            if not valid:
                last_error = f"AST validation: {reason}"
                logger.warning(f"Attempt {attempt + 1} — {last_error}")
                if attempt < self.MAX_ATTEMPTS - 1:
                    code = self._fix_with_llm(code, last_error, query_plan, df.columns.tolist())
                continue

            result, error = safe_exec(code, df)

            if error is None:
                logger.info(f"Execution succeeded on attempt {attempt + 1}")
                return result, code, None

            last_error = error
            logger.warning(f"Attempt {attempt + 1} failed: {error}")
            if attempt < self.MAX_ATTEMPTS - 1:
                code = self._fix_with_llm(code, error, query_plan, df.columns.tolist())

        return None, code, last_error

    def _fix_with_llm(
        self,
        code: str,
        error: str,
        plan: QueryPlan,
        columns: list,
    ) -> str:
        prompt = _FIX_PROMPT.format(
            question=plan.question,
            columns=columns,
            operation=plan.operation,
            code=code,
            error=error,
        )
        try:
            raw = self.llm.call(prompt, temperature=0, max_tokens=400)
            return self._strip_fences(raw)
        except Exception as exc:
            logger.error(f"LLM code-fix failed: {exc}")
            return code

    @staticmethod
    def _strip_fences(raw: str) -> str:
        raw = re.sub(r"```(?:python)?\n?", "", raw)
        raw = re.sub(r"```", "", raw)
        return raw.strip()
