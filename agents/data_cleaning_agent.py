import logging
import re
from typing import List, Tuple

import pandas as pd

from core.datastore import DataStore
from core.schema_profile import SchemaProfile

logger = logging.getLogger(__name__)

_CURRENCY_RE = re.compile(r"[₹$€£¥,\s]")


class DataCleaningAgent:
    def clean(
        self, datastore: DataStore, schema_profile: SchemaProfile
    ) -> Tuple[DataStore, List[str]]:
        ops: List[str] = []
        cleaned_sheets = {}

        for sheet_name, df in datastore.sheets.items():
            df = df.copy()
            for col_name, col_profile in schema_profile.columns.items():
                if col_name not in df.columns:
                    continue
                if not col_profile.needs_cleaning and col_profile.null_percent == 0:
                    continue

                series = df[col_name]

                if col_profile.semantic_type == "currency":
                    df[col_name], op = self._clean_currency(series, col_name)
                    if op:
                        ops.append(op)

                elif col_profile.semantic_type == "category":
                    df[col_name], op = self._clean_category(series, col_name)
                    if op:
                        ops.append(op)

                elif col_profile.semantic_type == "date":
                    df[col_name], op = self._clean_date(series, col_name)
                    if op:
                        ops.append(op)

                # Handle nulls
                if df[col_name].isna().any():
                    df[col_name], op = self._fill_nulls(
                        df[col_name], col_name, col_profile.semantic_type
                    )
                    if op:
                        ops.append(op)

            cleaned_sheets[sheet_name] = df

        new_ds = DataStore(
            sheets=cleaned_sheets,
            primary_sheet=datastore.primary_sheet,
            file_type=datastore.file_type,
            row_count=datastore.row_count,
            warnings=datastore.warnings[:],
            load_time_ms=datastore.load_time_ms,
        )
        logger.info(f"Cleaning applied {len(ops)} operations")
        return new_ds, ops

    def _clean_currency(self, series: pd.Series, col: str) -> Tuple[pd.Series, str]:
        cleaned = pd.to_numeric(
            series.astype(str).str.replace(_CURRENCY_RE, "", regex=True),
            errors="coerce",
        )
        return cleaned, f"'{col}': stripped currency symbols, converted to float"

    def _clean_category(self, series: pd.Series, col: str) -> Tuple[pd.Series, str]:
        cleaned = series.astype(str).str.strip().str.title()
        cleaned = cleaned.where(series.notna(), other=pd.NA)
        return cleaned, f"'{col}': stripped whitespace and applied title case"

    def _clean_date(self, series: pd.Series, col: str) -> Tuple[pd.Series, str]:
        try:
            cleaned = pd.to_datetime(series, errors="coerce")
            return cleaned, f"'{col}': parsed as datetime"
        except Exception:
            return series, ""

    def _fill_nulls(
        self, series: pd.Series, col: str, semantic: str
    ) -> Tuple[pd.Series, str]:
        null_count = series.isna().sum()
        if null_count == 0:
            return series, ""

        if semantic in ("numeric", "currency", "percentage"):
            fill_val = series.median()
            return series.fillna(fill_val), (
                f"'{col}': filled {null_count} nulls with median ({fill_val:.2f})"
            )

        if semantic == "category":
            mode_vals = series.mode()
            if len(mode_vals) > 0:
                fill_val = mode_vals[0]
                return series.fillna(fill_val), (
                    f"'{col}': filled {null_count} nulls with mode ('{fill_val}')"
                )

        return series, ""
