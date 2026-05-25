from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str
    unique_values: int
    value_range: Dict[str, Any]
    null_percent: float
    sample_values: List[Any]
    needs_cleaning: bool
    cleaning_hint: str

    def __repr__(self) -> str:
        return (
            f"ColumnProfile(name='{self.name}', semantic='{self.semantic_type}', "
            f"nulls={self.null_percent:.1f}%, unique={self.unique_values})"
        )


@dataclass
class SchemaProfile:
    columns: Dict[str, ColumnProfile]
    null_summary: Dict[str, float]
    primary_key: Optional[str]
    row_count: int
    suggested_cleaning: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SchemaProfile(columns={list(self.columns.keys())}, "
            f"rows={self.row_count}, pk='{self.primary_key}')"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": {
                name: {
                    "dtype": col.dtype,
                    "semantic_type": col.semantic_type,
                    "unique_values": col.unique_values,
                    "value_range": col.value_range,
                    "null_percent": col.null_percent,
                    "sample_values": [str(v) for v in col.sample_values],
                    "needs_cleaning": col.needs_cleaning,
                    "cleaning_hint": col.cleaning_hint,
                }
                for name, col in self.columns.items()
            },
            "null_summary": self.null_summary,
            "primary_key": self.primary_key,
            "row_count": self.row_count,
            "suggested_cleaning": self.suggested_cleaning,
        }
