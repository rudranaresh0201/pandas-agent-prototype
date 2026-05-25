"""
CrewAI tool wrappers. Each tool is also exposed as a plain Python function
for direct use in the pipeline (main.py auto-scenarios).

Module-level _state stores the DataStore and SchemaProfile between tool calls
so that the full DataFrame is never serialized through CrewAI's text context.
"""
import json
import logging
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Shared in-memory state across tool calls within a run
_state: Dict[str, Any] = {"datastore": None, "schema": None, "cleaning_ops": []}


# ── Plain Python functions (also called by BaseTool._run) ─────────────────────

def ingest_file_tool(file_path: str, file_type: str) -> str:
    """Ingest a CSV/XLSX file. Stores DataStore in module state. Returns summary JSON."""
    from agents.file_ingestion_agent import FileIngestionAgent
    try:
        ds = FileIngestionAgent().ingest(file_path, file_type)
        _state["datastore"] = ds
        summary = {
            "status": "ok",
            "primary_sheet": ds.primary_sheet,
            "row_count": ds.row_count,
            "file_type": ds.file_type,
            "sheets": list(ds.sheets.keys()),
            "warnings": ds.warnings,
            "load_time_ms": ds.load_time_ms,
        }
        return json.dumps(summary)
    except Exception as exc:
        logger.error(f"ingest_file_tool error: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def analyze_schema_tool(datastore_json: str = "") -> str:
    """Analyze schema & clean the loaded DataStore. Returns SchemaProfile JSON."""
    from agents.schema_analyst_agent import SchemaAnalystAgent
    from agents.data_cleaning_agent import DataCleaningAgent
    from core.datastore import DataStore

    try:
        ds = _state.get("datastore")
        if ds is None and datastore_json:
            ds = DataStore.from_json(datastore_json)

        if ds is None:
            return json.dumps({"status": "error", "message": "No datastore loaded"})

        schema_agent = SchemaAnalystAgent()
        schema = schema_agent.analyze(ds)

        cleaning_agent = DataCleaningAgent()
        ds_clean, ops = cleaning_agent.clean(ds, schema)
        schema = schema_agent.analyze(ds_clean)  # re-profile after cleaning

        _state["datastore"] = ds_clean
        _state["schema"] = schema
        _state["cleaning_ops"] = ops

        result = schema.to_dict()
        result["cleaning_ops"] = ops
        result["status"] = "ok"
        return json.dumps(result, default=str)
    except Exception as exc:
        logger.error(f"analyze_schema_tool error: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


def execute_query_tool(question: str, datastore_json: str = "") -> str:
    """Plan, execute, and validate a pandas query. Returns result JSON."""
    from agents.query_planner_agent import QueryPlannerAgent
    from agents.pandas_execution_agent import PandasExecutionAgent
    from agents.validation_agent import ValidationAgent
    from core.datastore import DataStore

    try:
        ds = _state.get("datastore")
        if ds is None and datastore_json:
            ds = DataStore.from_json(datastore_json)
        if ds is None:
            return json.dumps({"status": "error", "message": "No datastore available"})

        schema = _state.get("schema")

        plan = QueryPlannerAgent().plan(question, schema)
        result, code, error = PandasExecutionAgent().execute(plan, ds)

        if error:
            return json.dumps({"status": "error", "question": question, "error": error, "code": code})

        valid, reason = ValidationAgent().validate(result, plan)

        return json.dumps({
            "status": "ok",
            "question": question,
            "result": str(result),
            "code": code,
            "valid": valid,
            "validation_reason": reason,
            "operation": plan.operation,
            "output_type": plan.expected_output_type,
        })
    except Exception as exc:
        logger.error(f"execute_query_tool error: {exc}")
        return json.dumps({"status": "error", "message": str(exc)})


# ── CrewAI BaseTool wrappers ───────────────────────────────────────────────────

try:
    from crewai.tools import BaseTool

    class _IngestInput(BaseModel):
        file_path: str = Field(description="Absolute path to the CSV or XLSX file")
        file_type: str = Field(description="File type: 'csv' or 'xlsx'")

    class IngestFileTool(BaseTool):
        name: str = "ingest_file"
        description: str = (
            "Ingest a CSV or XLSX file and load it into the analysis session. "
            "Call this first before any other tool."
        )
        args_schema: Type[BaseModel] = _IngestInput

        def _run(self, file_path: str, file_type: str) -> str:
            return ingest_file_tool(file_path, file_type)

    class _SchemaInput(BaseModel):
        datastore_json: str = Field(default="", description="Optional DataStore JSON (leave empty to use session state)")

    class AnalyzeSchemaTool(BaseTool):
        name: str = "analyze_schema"
        description: str = (
            "Analyze the schema of the loaded dataset, detect semantic types, "
            "and apply cleaning. Call after ingest_file."
        )
        args_schema: Type[BaseModel] = _SchemaInput

        def _run(self, datastore_json: str = "") -> str:
            return analyze_schema_tool(datastore_json)

    class _QueryInput(BaseModel):
        question: str = Field(description="The business question to answer with data")
        datastore_json: str = Field(default="", description="Optional DataStore JSON (leave empty to use session state)")

    class ExecuteQueryTool(BaseTool):
        name: str = "execute_query"
        description: str = (
            "Execute a pandas query to answer a business question. "
            "Returns the result, generated code, and validation status."
        )
        args_schema: Type[BaseModel] = _QueryInput

        def _run(self, question: str, datastore_json: str = "") -> str:
            return execute_query_tool(question, datastore_json)

except ImportError:
    logger.warning("crewai not installed — BaseTool wrappers unavailable")

    class IngestFileTool:  # type: ignore[no-redef]
        def _run(self, file_path, file_type):
            return ingest_file_tool(file_path, file_type)

    class AnalyzeSchemaTool:  # type: ignore[no-redef]
        def _run(self, datastore_json=""):
            return analyze_schema_tool(datastore_json)

    class ExecuteQueryTool:  # type: ignore[no-redef]
        def _run(self, question, datastore_json=""):
            return execute_query_tool(question, datastore_json)
