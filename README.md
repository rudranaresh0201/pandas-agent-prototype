# CrewAI CSV Analytics Agent

A production-grade multi-agent system for CSV/XLSX analytics built on CrewAI, pandas, and OpenRouter.

## Architecture

```
crewai_csv_agent/
├── agents/               # 7 specialist agents
│   ├── file_ingestion_agent.py   — CSV/XLSX loading, encoding detection
│   ├── schema_analyst_agent.py   — semantic type detection, profiling
│   ├── data_cleaning_agent.py    — currency strips, null fills, title-case
│   ├── query_planner_agent.py    — LLM-based pandas query planning
│   ├── pandas_execution_agent.py — safe AST-validated execution + LLM retry
│   ├── validation_agent.py       — result type & range validation
│   └── insight_agent.py          — persona-aware natural language answers
├── core/                 # shared utilities
│   ├── datastore.py      — DataStore dataclass with JSON round-trip
│   ├── schema_profile.py — ColumnProfile & SchemaProfile dataclasses
│   ├── query_plan.py     — QueryPlan dataclass
│   ├── safe_executor.py  — AST validation + restricted exec sandbox
│   └── llm_client.py     — OpenRouter client with 3-attempt retry
├── crew/                 # CrewAI integration
│   ├── tools.py          — BaseTool wrappers + plain Python functions
│   └── analytics_crew.py — Two-agent crew: orchestrator + data_analyst
├── tests/                — pytest suite
├── data/                 — place your CSV/XLSX files here
├── logs/                 — run results written here
└── main.py               — CLI entry point + auto scenarios
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add your OPENROUTER_API_KEY
python main.py                # run both demo scenarios
```

## CLI usage

```bash
# Single question
python main.py --file data/sample_data.xlsx --question "Which branch has the highest NPA?"

# With persona
python main.py --file data/loans.csv --question "Total loan amount?" --persona "Compliance Officer"
```

## Personas

| Persona            | Style                                      |
|--------------------|--------------------------------------------|
| General            | Plain English, concise                     |
| Risk Analyst       | Risk-framed, statistical precision         |
| Student            | Simple language, analogies, step-by-step  |
| Compliance Officer | Formal, regulatory, action-oriented        |

## Safety

`core/safe_executor.py` validates every LLM-generated code snippet via Python's `ast` module **before** execution:
- Blocks all `import` / `from … import` statements
- Blocks dunder (`__`) attribute access
- Blocks dangerous builtins: `os`, `sys`, `subprocess`, `open`, `eval`, `exec`
- Runs code in a restricted `__builtins__` namespace

## Environment

```
OPENROUTER_API_KEY=sk-or-v1-...
```

## Run tests

```bash
pytest tests/ -v
```
