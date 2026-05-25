from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class QueryPlan:
    question: str
    target_columns: List[str]
    operation: str
    expected_output_type: str
    validation_rules: Dict[str, Any]
    suggested_code: str

    def __repr__(self) -> str:
        return (
            f"QueryPlan(op='{self.operation}', "
            f"output='{self.expected_output_type}', "
            f"cols={self.target_columns})"
        )
