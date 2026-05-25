"""
CrewAI CSV Analytics Agent - entry point.

Usage:
  python main.py                              # run both auto scenarios
  python main.py --file data/sample_data.xlsx --question "Which branch has the highest NPA?"
  python main.py --file data/sample_data.xlsx --question "..." --persona "Risk Analyst"
"""
import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

BASE_DIR = Path(__file__).parent

FINTECH_QUERIES = [
    "Which branch has the highest NPA ratio?",
    "What is the average DSCR across all branches?",
    "How many loans are in defaulted status?",
    "Which region has the most active loans?",
    "Show all branches where NPA ratio exceeds 5%",
]

EDTECH_QUERIES = [
    "What is the average marks per subject?",
    "How many students have attendance below 75%?",
    "Which semester has the highest average marks?",
    "Who are the top 5 students by marks?",
    "What is the distribution of grades?",
]


# ---- Data generation ---------------------------------------------------------

def generate_student_grades(out_path: Path) -> None:
    random.seed(42)
    subjects = ["Math", "Science", "English", "History", "Computer Science"]
    semesters = ["Semester 1", "Semester 2"]
    rows = []
    for i in range(1, 81):
        marks = random.randint(35, 100)
        grade = (
            "A" if marks >= 90 else
            "B" if marks >= 75 else
            "C" if marks >= 60 else
            "D" if marks >= 50 else "F"
        )
        rows.append({
            "Student_Name": f"Student_{i:03d}",
            "Subject": random.choice(subjects),
            "Marks": marks,
            "Grade": grade,
            "Attendance": round(random.uniform(50.0, 100.0), 1),
            "Semester": random.choice(semesters),
        })
    pd.DataFrame(rows).to_excel(out_path, index=False)
    logger.info(f"Generated {len(rows)}-row student_grades.xlsx -> {out_path}")


# ---- Pipeline ----------------------------------------------------------------

def run_pipeline(
    file_path: str,
    file_type: str,
    queries: List[str],
    persona: str = "General",
) -> List[Dict[str, Any]]:
    """Run all 7 agents sequentially for a batch of queries on one file."""
    from agents.file_ingestion_agent import FileIngestionAgent
    from agents.schema_analyst_agent import SchemaAnalystAgent
    from agents.data_cleaning_agent import DataCleaningAgent
    from agents.query_planner_agent import QueryPlannerAgent
    from agents.pandas_execution_agent import PandasExecutionAgent
    from agents.validation_agent import ValidationAgent
    from agents.insight_agent import InsightAgent

    # Pre-processing (shared across all queries in this scenario)
    print(f"\n  [FileIngestionAgent] Loading '{file_path}' ...")
    ds = FileIngestionAgent().ingest(file_path, file_type)
    print(f"  => {ds}")

    print("\n  [SchemaAnalystAgent] Profiling schema ...")
    schema_agent = SchemaAnalystAgent()
    schema = schema_agent.analyze(ds)
    print(f"  => {schema}")

    print("\n  [DataCleaningAgent] Cleaning data ...")
    ds, ops = DataCleaningAgent().clean(ds, schema)
    schema = schema_agent.analyze(ds)  # re-profile after cleaning
    print(f"  => {len(ops)} operation(s) applied")
    for op in ops:
        print(f"       * {op}")

    # Per-query pipeline
    planner = QueryPlannerAgent()
    executor = PandasExecutionAgent()
    validator = ValidationAgent()
    insight_agent = InsightAgent()

    results = []
    for i, question in enumerate(queries, 1):
        print(f"\n  -- Query {i}: {question}")

        try:
            print("  [QueryPlannerAgent] Planning ...")
            plan = planner.plan(question, schema)
            print(f"  => {plan}")

            print("  [PandasExecutionAgent] Executing ...")
            result, code, error = executor.execute(plan, ds)

            if error:
                valid, reason = False, error
                insight = f"[Execution error] {error}"
                print(f"  [x] Error: {error}")
            else:
                print("  [ValidationAgent] Validating ...")
                valid, reason = validator.validate(result, plan)
                print(f"  => valid={valid} | {reason}")

                print(f"  [InsightAgent] Generating insight ({persona}) ...")
                insight = insight_agent.generate(result, plan, persona)

            print(f"\n  Answer: {insight[:300]}")
            results.append({
                "query_num": i,
                "question": question,
                "answer": insight,
                "raw_result": str(result) if result is not None else "N/A",
                "code": code,
                "valid": valid,
                "validation_reason": reason,
                "error": error,
            })
        except Exception as exc:
            logger.exception(f"Query {i} failed: {exc}")
            results.append({
                "query_num": i,
                "question": question,
                "answer": f"[FAILED] {exc}",
                "raw_result": "N/A",
                "code": "",
                "valid": False,
                "validation_reason": str(exc),
                "error": str(exc),
            })

    return results


# ---- Report writer -----------------------------------------------------------

def write_report(scenario_results: List[Dict[str, Any]], out_path: Path) -> None:
    lines = [
        "=" * 70,
        "  CrewAI CSV Analytics Agent - Results",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
    ]
    for scenario in scenario_results:
        lines += [
            "",
            f"SCENARIO: {scenario['name']}",
            f"File    : {scenario['file']}",
            f"Persona : {scenario['persona']}",
            "-" * 70,
        ]
        for r in scenario["results"]:
            lines += [
                "",
                f"Query {r['query_num']}: {r['question']}",
                f"Answer : {r['answer']}",
                f"Valid  : {r['valid']}  ({r['validation_reason']})",
                "",
                "Generated code:",
                r["code"] or "(no code)",
                "-" * 70,
            ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Results saved -> {out_path}")


# ---- Auto scenarios ----------------------------------------------------------

def run_auto_scenarios() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = BASE_DIR / "logs" / f"results_{ts}.txt"
    all_results = []

    # Scenario 1: Fintech
    fintech_file = BASE_DIR / "data" / "sample_data.xlsx"
    if not fintech_file.exists():
        print(f"  [WARN] {fintech_file} not found - skipping Scenario 1")
    else:
        print("\n" + "=" * 70)
        print("  SCENARIO 1 - Fintech (sample_data.xlsx)  |  Persona: Risk Analyst")
        print("=" * 70)
        res = run_pipeline(str(fintech_file), "xlsx", FINTECH_QUERIES, persona="Risk Analyst")
        all_results.append({
            "name": "Scenario 1 - Fintech",
            "file": str(fintech_file),
            "persona": "Risk Analyst",
            "results": res,
        })

    # Scenario 2: EdTech
    edtech_file = BASE_DIR / "data" / "student_grades.xlsx"
    generate_student_grades(edtech_file)

    print("\n" + "=" * 70)
    print("  SCENARIO 2 - EdTech (student_grades.xlsx)  |  Persona: Student")
    print("=" * 70)
    res = run_pipeline(str(edtech_file), "xlsx", EDTECH_QUERIES, persona="Student")
    all_results.append({
        "name": "Scenario 2 - EdTech",
        "file": str(edtech_file),
        "persona": "Student",
        "results": res,
    })

    write_report(all_results, log_file)


# ---- Entry point -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CrewAI CSV Analytics Agent")
    parser.add_argument("--file", help="Path to CSV or XLSX file")
    parser.add_argument("--question", help="Question to answer")
    parser.add_argument(
        "--persona",
        default="General",
        choices=["General", "Risk Analyst", "Student", "Compliance Officer"],
        help="Explanation persona (default: General)",
    )
    args = parser.parse_args()

    if args.file and args.question:
        ft = "xlsx" if args.file.lower().endswith((".xlsx", ".xls")) else "csv"
        print(f"\nSingle-query mode | file={args.file} | persona={args.persona}")
        from crew.analytics_crew import AnalyticsCrew
        answer = AnalyticsCrew().run(args.question, args.file, ft, args.persona)
        print(f"\nAnswer:\n{answer}")
    elif args.file or args.question:
        parser.error("Provide both --file and --question together, or neither.")
    else:
        run_auto_scenarios()


if __name__ == "__main__":
    main()
