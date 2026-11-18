from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.database import Database
from app.ingestion import ingest
from app.llm import LLMUnavailable, OpenAIAnswerModel
from app.models import Employee
from app.retrieval import (
    Retriever,
    build_context,
    reference_date_from_question,
)


def load_cases() -> list[dict[str, object]]:
    path = ROOT / "evaluation" / "cases.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def employee_for(database: Database, region: str, role: str) -> Employee:
    with database.connect() as connection:
        row = connection.execute(
            """
            SELECT employee_id FROM employees
            WHERE region = ? AND role = ?
            ORDER BY employee_id LIMIT 1
            """,
            (region, role),
        ).fetchone()
    if row is None:
        raise ValueError(f"No {role} record exists for evaluation region {region}")
    employee = database.get_employee(row["employee_id"])
    if employee is None:
        raise ValueError("Evaluation employee record disappeared")
    return employee


def run(live: bool, mock_ocr: bool) -> int:
    settings = Settings.from_env()
    database = Database(settings.db_path)
    database.initialize()
    if mock_ocr:
        with patch(
            "app.ingestion.pytesseract.image_to_string",
            return_value="OCR skipped by explicit offline-evaluation option.",
        ):
            documents = ingest(settings.corpus_dir, settings.manifest_path)
        print("OCR stub enabled; scanned-document content is not being evaluated.")
    else:
        documents = ingest(settings.corpus_dir, settings.manifest_path)
    passages = tuple(passage for doc in documents for passage in doc.passages)
    retriever = Retriever(passages)
    model = OpenAIAnswerModel(settings.openai_api_key, settings.openai_model)
    cases = load_cases()
    outcomes: list[dict[str, object]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    priced_calls = 0

    for case in cases:
        employee = employee_for(
            database, str(case["region"]), str(case["role"])
        )
        question = str(case["question"])
        started = time.perf_counter()
        reference_date = reference_date_from_question(question, settings.today)
        results = retriever.retrieve(
            question,
            employee,
            reference_date,
            settings.retrieval_top_k,
        )
        context, selected, corpus_tokens = build_context(
            results,
            settings.max_context_tokens,
            database.redact_personal_data,
        )
        selected_ids = {item.passage.doc_id for item in selected}
        expected_ids = set(case["expected_doc_ids"])
        expected_upcoming_ids = set(case.get("expected_upcoming_doc_ids", []))
        excluded_ids = set(case["excluded_doc_ids"])
        actual_upcoming_ids = {
            item.passage.doc_id for item in selected if item.upcoming
        }
        evidence = "\n".join(item.passage.text for item in selected).casefold()
        evidence_terms_ok = all(
            str(term).casefold() in evidence for term in case["expected_terms"]
        )
        source_recall_ok = expected_ids.issubset(selected_ids)
        upcoming_ok = expected_upcoming_ids == actual_upcoming_ids
        access_ok = not (excluded_ids & selected_ids)
        citation_ok = True
        answer_terms_ok = True
        evidence_support_ok = True
        latency_ms = 0.0
        input_tokens = 0
        output_tokens = 0
        answer = ""
        latency_ms = (time.perf_counter() - started) * 1000
        if live and selected:
            if not settings.openai_api_key:
                raise LLMUnavailable("Set OPENAI_API_KEY before running --live")
            answer_started = time.perf_counter()
            generated = model.answer(
                database.redact_personal_data(question),
                context,
                corpus_tokens,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            input_tokens = generated.input_tokens
            output_tokens = generated.output_tokens
            answer = generated.answer
            allowed = {
                (item.passage.doc_id, item.passage.locator) for item in selected
            }
            citation_ids = {citation.doc_id for citation in generated.citations}
            citation_ok = all(
                (citation.doc_id, citation.locator) in allowed
                for citation in generated.citations
            ) and (expected_ids | expected_upcoming_ids).issubset(citation_ids)
            answer_terms_ok = all(
                str(term).casefold() in answer.casefold()
                for term in case["expected_terms"]
            )
            cited_text = "\n".join(
                item.passage.text
                for item in selected
                if any(
                    (c.doc_id, c.locator)
                    == (item.passage.doc_id, item.passage.locator)
                    for c in generated.citations
                )
            ).casefold()
            evidence_support_ok = all(
                str(term).casefold() in cited_text for term in case["expected_terms"]
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            if (
                settings.input_cost_per_million is not None
                and settings.output_cost_per_million is not None
            ):
                total_cost += (
                    input_tokens * settings.input_cost_per_million
                    + output_tokens * settings.output_cost_per_million
                ) / 1_000_000
                priced_calls += 1
        elif not live:
            latency_ms = (time.perf_counter() - started) * 1000

        passed = (
            source_recall_ok
            and upcoming_ok
            and evidence_terms_ok
            and access_ok
            and corpus_tokens <= settings.max_context_tokens
            and citation_ok
            and answer_terms_ok
            and evidence_support_ok
        )
        outcomes.append(
            {
                "id": case["id"],
                "pass": passed,
                "retrieved": ",".join(sorted(selected_ids)) or "-",
                "expected": ",".join(sorted(expected_ids)) or "-",
                "access_ok": access_ok,
                "corpus_tokens": corpus_tokens,
                "latency_ms": round(latency_ms, 1),
            }
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"{case['id']:<27} {status:<5} retrieved={','.join(sorted(selected_ids)) or '-':<45} "
            f"upcoming={','.join(sorted(actual_upcoming_ids)) or '-':<38} "
            f"checks=src:{source_recall_ok},future:{upcoming_ok},evidence:{evidence_terms_ok},"
            f"acl:{access_ok} corpus_tokens={corpus_tokens:<4} latency_ms={latency_ms:.1f}"
        )

    passed_count = sum(bool(outcome["pass"]) for outcome in outcomes)
    print(f"\nEvidence cases: {passed_count}/{len(outcomes)} passed")
    print(f"Access-control leaks: {sum(not bool(o['access_ok']) for o in outcomes)}")
    print(f"Corpus token violations: {sum(int(o['corpus_tokens']) > 3000 for o in outcomes)}")
    if live:
        print(f"LLM input tokens: {total_input_tokens}")
        print(f"LLM output tokens: {total_output_tokens}")
        print(
            f"Estimated cost USD: {total_cost:.8f}"
            if priced_calls
            else "Estimated cost USD: not priced (configure both token-rate variables)"
        )
    return 0 if passed_count == len(outcomes) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured OpenAI model and measure answer/citation quality.",
    )
    parser.add_argument(
        "--mock-ocr",
        action="store_true",
        help="Stub scanned-page OCR for environments without Tesseract.",
    )
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.live, arguments.mock_ocr))
