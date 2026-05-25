---
name: crewai-state-propagation-bug
description: Root cause and fix for DataStore not reaching execute_query in CrewAI flow — LLM tool call ordering cannot be trusted for required setup
metadata:
  type: project
---

DataStore was None when execute_query_tool ran in the CrewAI path because the orchestrator LLM (llama-3.1-8b-instruct) did not reliably call ingest_file before task2 started.

**Why:** CrewAI inter-task context is text-only (the LLM's written output). Python module-level `_state` in `crew/tools.py` only gets populated if the tool function is actually invoked — if the LLM describes what it would do instead of calling the tool, `_state["datastore"]` stays None. DataFrames cannot cross an LLM prompt boundary.

**Fix applied in `crew/analytics_crew.py`:** Pre-seed `_state` deterministically by calling `ingest_file_tool()` and `analyze_schema_tool()` directly before `crew.kickoff()`. The orchestrator task1 now receives the schema summary as pre-built text (not from tool calls), and task2's LLM only needs to call `execute_query` with a question string.

**How to apply:** Any time an agentic workflow requires shared Python state between agents, pre-populate that state before handing control to LLMs. Never rely on LLM tool-call ordering for required setup. LLM-driven tool calls are for business logic, not initialization.

**Related pattern:** [[direct-pipeline-works]] — `_run_direct_pipeline` works correctly because it calls the functions directly, bypassing LLM indirection entirely.
