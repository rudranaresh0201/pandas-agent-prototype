import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.datastore import DataStore

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252", "utf-8-sig"]
DELIMITERS_TO_TRY = [",", ";", "\t", "|"]


class FileIngestionAgent:
    def ingest(self, file_path: str, file_type: str) -> DataStore:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f} MB (max 50 MB)"
            )

        warnings: List[str] = []
        start = time.time()
        ft = file_type.lower().strip()

        try:
            if ft == "csv":
                sheets, ws = self._ingest_csv(path)
            elif ft in ("xlsx", "xls", "excel"):
                sheets, ws = self._ingest_excel(path)
            else:
                raise ValueError(f"Unsupported file type: '{file_type}'")
            warnings.extend(ws)
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            logger.error(f"Ingestion failed for {file_path}: {exc}")
            raise RuntimeError(f"Failed to ingest '{file_path}': {exc}") from exc

        primary = list(sheets.keys())[0]
        elapsed_ms = (time.time() - start) * 1000

        ds = DataStore(
            sheets=sheets,
            primary_sheet=primary,
            file_type=ft,
            row_count=len(sheets[primary]),
            warnings=warnings,
            load_time_ms=round(elapsed_ms, 2),
        )
        logger.info(f"Ingested: {ds}")
        return ds

    def _ingest_csv(self, path: Path) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        warnings: List[str] = []
        df: Optional[pd.DataFrame] = None
        used_encoding = "utf-8"

        for encoding in ENCODINGS_TO_TRY:
            for delim in DELIMITERS_TO_TRY:
                try:
                    candidate = pd.read_csv(
                        path, delimiter=delim, encoding=encoding, engine="python"
                    )
                    if candidate.shape[1] > 1 and not candidate.empty:
                        df = candidate
                        used_encoding = encoding
                        break
                except Exception:
                    continue
            if df is not None:
                break

        if df is None or df.empty:
            raise ValueError("Could not parse CSV with any standard delimiter/encoding")

        if df.shape[1] == 1:
            warnings.append("Single-column CSV — check that the delimiter is correct")

        if used_encoding != "utf-8":
            warnings.append(f"Non-UTF-8 encoding detected: {used_encoding}")

        return {path.stem: df}, warnings

    def _ingest_excel(self, path: Path) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        warnings: List[str] = []
        sheets: Dict[str, pd.DataFrame] = {}

        try:
            xl = pd.ExcelFile(path, engine="openpyxl")
        except Exception as exc:
            raise RuntimeError(f"Cannot open Excel file: {exc}") from exc

        for sheet_name in xl.sheet_names:
            try:
                df = pd.read_excel(xl, sheet_name=sheet_name)
                if df.empty:
                    warnings.append(f"Sheet '{sheet_name}' is empty — skipped")
                    continue
                sheets[str(sheet_name)] = df
            except Exception as exc:
                warnings.append(f"Sheet '{sheet_name}' failed to load: {exc}")

        if not sheets:
            raise ValueError("No readable sheets found in Excel file")

        return sheets, warnings
