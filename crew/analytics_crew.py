import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "openrouter/meta-llama/llama-3.1-8b-instruct"


class AnalyticsCrew:
    """
    Two-agent CrewAI crew: orchestrator (ingest + schema) → data_analyst (query).
    Falls back to direct pipeline execution if crewai is unavailable.
    """

    def __init__(self):
        self._crew_available = self._try_init_crew()

    def _try_init_crew(self) -> bool:
        try:
            from crewai import Agent, LLM
            from crew.tools import IngestFileTool, AnalyzeSchemaTool, ExecuteQueryTool

            api_key = os.getenv("OPENROUTER_API_KEY", "")
            llm = LLM(
                model=_MODEL,
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                temperature=0,
            )

            self.orchestrator = Agent(
                role="Data Orchestrator",
                goal="Ingest data files and profile their schemas for downstream analysis",
                backstory=(
                    "You are a meticulous data engineer who loads files, detects "
                    "schema issues, and summarizes dataset structure."
                ),
                tools=[IngestFileTool(), AnalyzeSchemaTool()],
                llm=llm,
                verbose=True,
                allow_delegation=False,
            )

            self.data_analyst = Agent(
                role="Senior Data Analyst",
                goal="Answer business questions accurately using pandas queries",
                backstory=(
                    "You are an expert data analyst who translates business questions "
                    "into precise pandas code and presents clear findings."
                ),
                tools=[ExecuteQueryTool()],
                llm=llm,
                verbose=True,
                allow_delegation=False,
            )

            logger.info("CrewAI agents initialized successfully")
            return True
        except Exception as exc:
            logger.warning(f"CrewAI init failed ({exc}) — will use direct pipeline")
            return False

    def run(
        self,
        question: str,
        file_path: str,
        file_type: str,
        persona: str = "General",
    ) -> str:
        if self._crew_available:
            return self._run_with_crew(question, file_path, file_type, persona)
        return self._run_direct_pipeline(question, file_path, file_type, persona)

    def _run_with_crew(self, question, file_path, file_type, persona) -> str:
        from crewai import Crew, Process, Task
        from crew.tools import ingest_file_tool, analyze_schema_tool
        import json

        # Pre-populate module _state deterministically before the crew starts.
        # CrewAI inter-task context is text-only — DataFrames cannot survive an
        # LLM prompt boundary, so we must not rely on the orchestrator LLM to
        # actually invoke ingest_file. We call the tool functions directly here,
        # which sets crew.tools._state["datastore"] and crew.tools._state["schema"].
        # The orchestrator's task1 then receives the schema summary as text context
        # for task2, which is all CrewAI context can legitimately provide.
        logger.info("Pre-seeding module state before crew kickoff")
        ingest_result_raw = ingest_file_tool(file_path, file_type)
        ingest_result = json.loads(ingest_result_raw)
        if ingest_result.get("status") != "ok":
            logger.error(f"Ingestion failed: {ingest_result}")
            return self._generate_insight(
                f"[Ingestion error] {ingest_result.get('message', ingest_result_raw)}",
                question,
                persona,
            )

        schema_result_raw = analyze_schema_tool()
        schema_result = json.loads(schema_result_raw)
        if schema_result.get("status") != "ok":
            logger.warning(f"Schema analysis issue: {schema_result.get('message')}")

        # Build a plain-text schema summary to inject into task1's output so the
        # data_analyst has column names and types as prompt context.
        schema_summary = (
            f"Dataset: {ingest_result.get('primary_sheet')} | "
            f"Rows: {ingest_result.get('row_count')} | "
            f"Columns: {', '.join(str(c) for c in schema_result.get('columns', {}).keys())}"
        )

        task1 = Task(
            description=(
                f"The dataset has already been loaded and profiled. "
                f"Here is the schema summary:\n{schema_summary}\n\n"
                f"Review this schema and confirm the dataset is ready for analysis. "
                f"Note any potential data quality concerns relevant to: \"{question}\""
            ),
            expected_output=(
                "A brief confirmation that the dataset is loaded, with column names, "
                "row count, and any data quality notes relevant to the question."
            ),
            agent=self.orchestrator,
        )

        task2 = Task(
            description=(
                f"Answer this question using the execute_query tool: \"{question}\"\n"
                f"The dataset is already loaded in the session — do not re-load it. "
                f"Call execute_query with only the 'question' argument."
            ),
            expected_output="A clear, concise answer to the business question with the key metric or finding.",
            agent=self.data_analyst,
            context=[task1],
        )

        crew = Crew(
            agents=[self.orchestrator, self.data_analyst],
            tasks=[task1, task2],
            process=Process.sequential,
            verbose=True,
            memory=False,
        )

        try:
            result = crew.kickoff()
            raw_answer = str(result)
        except Exception as exc:
            logger.error(f"Crew execution failed: {exc}")
            raw_answer = f"[Crew error] {exc}"

        return self._generate_insight(raw_answer, question, persona)

    def _run_direct_pipeline(self, question, file_path, file_type, persona) -> str:
        from crew.tools import (
            ingest_file_tool,
            analyze_schema_tool,
            execute_query_tool,
        )
        from agents.insight_agent import InsightAgent
        import json

        logger.info("Running direct pipeline (CrewAI unavailable)")
        ingest_file_tool(file_path, file_type)
        analyze_schema_tool()
        result_json = execute_query_tool(question)
        result_data = json.loads(result_json)

        raw_answer = result_data.get("result", result_data.get("error", "No answer"))
        return self._generate_insight(raw_answer, question, persona)

    def _generate_insight(self, raw_answer: str, question: str, persona: str) -> str:
        from agents.insight_agent import InsightAgent
        from core.query_plan import QueryPlan

        dummy_plan = QueryPlan(
            question=question,
            target_columns=[],
            operation="",
            expected_output_type="string",
            validation_rules={},
            suggested_code="",
        )
        try:
            return InsightAgent().generate(raw_answer, dummy_plan, persona)
        except Exception as exc:
            logger.warning(f"Insight generation failed: {exc}")
            return raw_answer
