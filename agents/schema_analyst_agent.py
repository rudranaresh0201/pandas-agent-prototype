import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.datastore import DataStore
from core.schema_profile import ColumnProfile, SchemaProfile

logger = logging.getLogger(__name__)

CATEGORY_THRESHOLD = 0.05  # unique / total <= 5% → category
CURRENCY_PATTERN = re.compile(r"[₹$€£¥]")


class SchemaAnalystAgent:
    def analyze(self, datastore: DataStore) -> SchemaProfile:
        df = datastore.sheets[datastore.primary_sheet].copy()
        row_count = len(df)

        columns: Dict[str, ColumnProfile] = {}
        null_summary: Dict[str, float] = {}
        suggested_cleaning: List[str] = []
        primary_key: Optional[str] = None

        for col in df.columns:
            series = df[col]
            null_pct = round(series.isna().mean() * 100, 2)
            null_summary[col] = null_pct

            semantic, needs_clean, hint = self._detect_semantic(series, row_count)
            vrange = self._value_range(series, semantic)
            sample = [v for v in series.dropna().head(5).tolist()]

            if needs_clean:
                suggested_cleaning.append(f"{col}: {hint}")

            if primary_key is None and self._looks_like_pk(series, row_count):
                primary_key = col

            columns[str(col)] = ColumnProfile(
                name=str(col),
                dtype=str(series.dtype),
                semantic_type=semantic,
                unique_values=int(series.nunique()),
                value_range=vrange,
                null_percent=null_pct,
                sample_values=sample,
                needs_cleaning=needs_clean,
                cleaning_hint=hint,
            )

        profile = SchemaProfile(
            columns=columns,
            null_summary=null_summary,
            primary_key=primary_key,
            row_count=row_count,
            suggested_cleaning=suggested_cleaning,
        )
        logger.info(f"Schema analyzed: {profile}")
        return profile

    def _detect_semantic(
        self, series: pd.Series, row_count: int
    ) -> Tuple[str, bool, str]:
        needs_clean = False
        hint = ""

        if pd.api.types.is_datetime64_any_dtype(series):
            return "date", False, ""

        str_sample = series.dropna().astype(str).head(50)

        if str_sample.str.match(r"^\d{4}[-/]\d{2}[-/]\d{2}").any():
            return "date", True, "Parse as datetime"

        has_currency = str_sample.apply(lambda v: bool(CURRENCY_PATTERN.search(v))).any()
        if has_currency:
            return "currency", True, "Strip currency symbols and convert to float"

        if pd.api.types.is_numeric_dtype(series):
            mn, mx = series.min(), series.max()
            if 0 <= mn and mx <= 100 and series.dtype == float:
                return "percentage", False, ""
            return "numeric", False, ""

        uniq_ratio = series.nunique() / max(row_count, 1)
        if uniq_ratio <= CATEGORY_THRESHOLD:
            has_whitespace = str_sample.str.startswith(" ").any() or str_sample.str.endswith(" ").any()
            if has_whitespace:
                needs_clean = True
                hint = "Strip whitespace and title-case values"
            return "category", needs_clean, hint

        col_lower = series.name.lower() if hasattr(series, "name") else ""
        if any(kw in col_lower for kw in ("id", "code", "key", "ref", "num")):
            return "id", False, ""

        return "text", False, ""

    def _value_range(self, series: pd.Series, semantic: str) -> Dict[str, Any]:
        if semantic in ("numeric", "percentage", "currency"):
            clean = pd.to_numeric(
                series.astype(str).str.replace(r"[₹$€£¥,]", "", regex=True),
                errors="coerce",
            )
            return {
                "min": float(clean.min()) if not clean.isna().all() else None,
                "max": float(clean.max()) if not clean.isna().all() else None,
            }
        if semantic == "category":
            return {"values": series.dropna().unique().tolist()[:20]}
        return {}

    def _looks_like_pk(self, series: pd.Series, row_count: int) -> bool:
        return (
            series.nunique() == row_count
            and series.isna().sum() == 0
            and row_count > 1
        )
