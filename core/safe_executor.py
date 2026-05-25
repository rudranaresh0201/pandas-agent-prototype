import ast
import logging
from typing import Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

BLOCKED_NAMES = frozenset({
    "os", "sys", "subprocess", "open", "eval", "exec",
    "__import__", "compile", "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
})

SAFE_BUILTINS: dict = {
    "len": len, "range": range, "list": list, "dict": dict,
    "str": str, "int": int, "float": float, "bool": bool,
    "round": round, "sum": sum, "min": min, "max": max,
    "sorted": sorted, "enumerate": enumerate, "zip": zip,
    "print": print, "abs": abs, "any": any, "all": all,
    "isinstance": isinstance, "tuple": tuple, "set": set,
    "None": None, "True": True, "False": False,
}


def validate_ast(code: str) -> Tuple[bool, str]:
    """Return (ok, reason). Blocks imports, dunders, and dangerous builtins."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Import statements are not allowed"

        if isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and node.attr.startswith("__"):
                return False, f"Dunder attribute access not allowed: '{node.attr}'"

        if isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES:
                return False, f"Blocked identifier: '{node.id}'"

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                return False, f"Blocked function call: '{node.func.id}'"

    return True, "OK"


def safe_exec(code: str, df: pd.DataFrame) -> Tuple[Any, str]:
    """Execute pandas code in a restricted namespace. Returns (result, error_msg)."""
    valid, reason = validate_ast(code)
    if not valid:
        return None, f"AST validation failed: {reason}"

    namespace: dict = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(),
        "pd": pd,
    }
    try:
        exec(code, namespace)  # noqa: S102
        result = namespace.get("result")
        if result is None:
            return None, "Code did not assign a value to 'result'"
        return result, None
    except Exception as exc:
        logger.warning(f"safe_exec error: {exc}")
        return None, str(exc)
