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

        task1 = Task(
            description=(
                f"Use the ingest_file tool to load '{file_path}' (type: {file_type}). "
                f"Then use analyze_schema to profile the data. "
                f"Return a concise summary of the dataset structure."
            ),
            expected_output=(
                "A brief description of the dataset: sheet names, row count, "
                "column types, and any data quality warnings."
            ),
            agent=self.orchestrator,
        )

        task2 = Task(
            description=(
                f"Answer this question: \"{question}\"\n"
                f"Use the execute_query tool with the question above. "
                f"The data has already been loaded by the orchestrator in task 1."
            ),
            expected_output="A clear, concise answer to the business question.",
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
