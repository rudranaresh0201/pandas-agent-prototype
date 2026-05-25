"""Tests for safe_executor and PandasExecutionAgent."""
import pandas as pd
import pytest

from core.safe_executor import safe_exec, validate_ast


# ── AST validation ─────────────────────────────────────────────────────────────

def test_valid_ast_passes():
    ok, _ = validate_ast("result = df['col'].sum()")
    assert ok


def test_import_blocked():
    ok, msg = validate_ast("import os\nresult = os.getcwd()")
    assert not ok
    assert "Import" in msg


def test_from_import_blocked():
    ok, msg = validate_ast("from subprocess import run\nresult = run(['ls'])")
    assert not ok
    assert "Import" in msg


def test_dunder_attribute_blocked():
    ok, msg = validate_ast("result = df.__class__.__name__")
    assert not ok
    assert "Dunder" in msg or "dunder" in msg.lower()


def test_eval_blocked():
    ok, msg = validate_ast("result = eval('1+1')")
    assert not ok


def test_exec_blocked():
    ok, msg = validate_ast("exec('x=1'); result = x")
    assert not ok


def test_os_name_blocked():
    ok, msg = validate_ast("result = os.path.join('a','b')")
    assert not ok
    assert "os" in msg


def test_syntax_error_caught():
    ok, msg = validate_ast("result = df[")
    assert not ok
    assert "Syntax" in msg


# ── safe_exec ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "branch": ["A", "B", "C"],
        "npa_ratio": [3.5, 6.2, 1.8],
        "amount": [100, 200, 150],
    })


def test_safe_exec_basic_sum(sample_df):
    result, err = safe_exec("result = df['npa_ratio'].sum()", sample_df)
    assert err is None
    assert abs(result - 11.5) < 1e-6


def test_safe_exec_filter(sample_df):
    result, err = safe_exec("result = df[df['npa_ratio'] > 5]['branch'].tolist()", sample_df)
    assert err is None
    assert result == ["B"]


def test_safe_exec_no_result_var(sample_df):
    result, err = safe_exec("x = df['amount'].mean()", sample_df)
    assert result is None
    assert err is not None


def test_safe_exec_blocks_import(sample_df):
    result, err = safe_exec("import os; result = os.getcwd()", sample_df)
    assert result is None
    assert err is not None


def test_safe_exec_runtime_error(sample_df):
    result, err = safe_exec("result = df['nonexistent_col'].sum()", sample_df)
    assert result is None
    assert err is not None


def test_safe_exec_df_is_a_copy(sample_df):
    original_len = len(sample_df)
    safe_exec("df.drop(df.index, inplace=True); result = 1", sample_df)
    assert len(sample_df) == original_len
