"""Tests for FileIngestionAgent."""
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from agents.file_ingestion_agent import FileIngestionAgent


@pytest.fixture()
def agent():
    return FileIngestionAgent()


@pytest.fixture()
def tmp_csv(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("Name,Age,Score\nAlice,25,85.5\nBob,30,92.0\nCarol,22,78.3\n", encoding="utf-8")
    return str(path)


@pytest.fixture()
def tmp_xlsx(tmp_path):
    path = tmp_path / "sample.xlsx"
    df = pd.DataFrame({"Product": ["A", "B"], "Sales": [100, 200], "Region": ["North", "South"]})
    df.to_excel(path, index=False)
    return str(path)


def test_csv_ingestion(agent, tmp_csv):
    ds = agent.ingest(tmp_csv, "csv")
    assert ds.file_type == "csv"
    assert ds.row_count == 3
    assert "Name" in ds.sheets[ds.primary_sheet].columns


def test_xlsx_ingestion(agent, tmp_xlsx):
    ds = agent.ingest(tmp_xlsx, "xlsx")
    assert ds.file_type == "xlsx"
    assert ds.row_count == 2
    assert "Product" in ds.sheets[ds.primary_sheet].columns


def test_file_not_found(agent):
    with pytest.raises(FileNotFoundError):
        agent.ingest("/nonexistent/path/file.csv", "csv")


def test_unsupported_format(agent, tmp_csv):
    with pytest.raises(ValueError, match="Unsupported"):
        agent.ingest(tmp_csv, "json")


def test_datastore_json_roundtrip(agent, tmp_csv):
    ds = agent.ingest(tmp_csv, "csv")
    from core.datastore import DataStore
    restored = DataStore.from_json(ds.to_json())
    assert restored.row_count == ds.row_count
    assert restored.primary_sheet == ds.primary_sheet
    assert list(restored.sheets[restored.primary_sheet].columns) == list(
        ds.sheets[ds.primary_sheet].columns
    )


def test_semicolon_delimited_csv(agent, tmp_path):
    path = tmp_path / "semi.csv"
    path.write_text("col1;col2;col3\n1;2;3\n4;5;6\n", encoding="utf-8")
    ds = agent.ingest(str(path), "csv")
    assert ds.sheets[ds.primary_sheet].shape == (2, 3)


def test_warnings_populated_for_empty_sheet(agent, tmp_path):
    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame({"x": [1, 2]}).to_excel(w, sheet_name="Data", index=False)
        pd.DataFrame().to_excel(w, sheet_name="Empty", index=False)
    ds = agent.ingest(str(path), "xlsx")
    assert "Data" in ds.sheets
    assert any("Empty" in w for w in ds.warnings)
