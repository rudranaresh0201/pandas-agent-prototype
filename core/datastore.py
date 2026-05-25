from dataclasses import dataclass, field
from typing import Dict, List
import json
import pandas as pd


@dataclass
class DataStore:
    sheets: Dict[str, pd.DataFrame]
    primary_sheet: str
    file_type: str
    row_count: int
    warnings: List[str] = field(default_factory=list)
    load_time_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"DataStore(sheets={list(self.sheets.keys())}, "
            f"primary='{self.primary_sheet}', rows={self.row_count}, "
            f"type='{self.file_type}')"
        )

    def to_json(self) -> str:
        payload = {
            "primary_sheet": self.primary_sheet,
            "file_type": self.file_type,
            "row_count": self.row_count,
            "warnings": self.warnings,
            "load_time_ms": self.load_time_ms,
            "sheets": {
                name: df.to_dict(orient="records")
                for name, df in self.sheets.items()
            },
        }
        return json.dumps(payload, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "DataStore":
        data = json.loads(json_str)
        sheets = {
            name: pd.DataFrame(rows) for name, rows in data["sheets"].items()
        }
        return cls(
            sheets=sheets,
            primary_sheet=data["primary_sheet"],
            file_type=data["file_type"],
            row_count=data["row_count"],
            warnings=data.get("warnings", []),
            load_time_ms=data.get("load_time_ms", 0.0),
        )
