from __future__ import annotations

from datetime import date

from app.models import Employee
from app.retrieval import (
    Retriever,
    build_context,
    can_read_document,
    precedence_key,
    reference_date_from_question,
)

from conftest import make_document


def test_document_access_by_role_and_region() -> None:
    employee = Employee("E1", "A", "a@example.test", "1", "IN", date(2020, 1, 1), "L1", "employee", None)
    manager = Employee("E2", "B", "b@example.test", "2", "IN", date(2020, 1, 1), "L2", "manager", None)
    hr = Employee("E3", "C", "c@example.test", "3", "IN", date(2020, 1, 1), "L3", "hr", None)
    docs = (
        make_document("ALL", "leave policy access token", access_level="all"),
        make_document("MANAGER", "leave policy access token", access_level="managers"),
        make_document("HR", "leave policy access token", access_level="hr"),
        make_document("DE", "leave policy access token", regions=("DE",)),
    )
    for user, expected in (
        (employee, {"ALL"}),
        (manager, {"ALL", "MANAGER"}),
        (hr, {"ALL", "MANAGER", "HR"}),
    ):
        retrieved = Retriever(tuple(p for doc in docs for p in doc.passages)).retrieve(
            "leave policy access token", user, date(2026, 1, 1), 8
        )
        assert {item.passage.doc_id for item in retrieved} == expected

    assert can_read_document(employee, "all")
    assert not can_read_document(employee, "managers")
    assert can_read_document(manager, "managers")
    assert not can_read_document(manager, "hr")
    assert can_read_document(hr, "hr")


def test_effective_date_and_precedence_metadata() -> None:
    employee = Employee("E1", "A", "a@example.test", "1", "IN", date(2020, 1, 1), "L1", "employee", None)
    docs = (
        make_document(
            "V1",
            "sick leave allowance 10 working days",
            effective_date=date(2025, 1, 1),
        ),
        make_document(
            "V2",
            "sick leave allowance 12 working days",
            effective_date=date(2026, 4, 1),
        ),
        make_document(
            "IN-ADDENDUM",
            "sick leave allowance 15 working days",
            doc_type="addendum",
            regions=("IN",),
            effective_date=date(2025, 1, 1),
        ),
        make_document(
            "FAQ",
            "sick leave allowance 9 working days",
            doc_type="faq",
            effective_date=date(2026, 5, 1),
        ),
        make_document(
            "V3",
            "sick leave allowance 20 working days",
            effective_date=date(2027, 1, 1),
        ),
        make_document(
            "V4",
            "two paid wellbeing days start in 2027",
            effective_date=date(2027, 1, 1),
        ),
    )
    retriever = Retriever(tuple(p for doc in docs for p in doc.passages))
    today = date(2026, 9, 15)
    results = retriever.retrieve(
        "sick leave allowance working days", employee, today, 10
    )
    ids = {result.passage.doc_id for result in results}
    assert {"V1", "V2", "IN-ADDENDUM", "FAQ"} == ids
    assert all(not result.upcoming for result in results)
    specific_addendum = next(
        item.passage for item in results if item.passage.doc_id == "IN-ADDENDUM"
    )
    current_global = next(item.passage for item in results if item.passage.doc_id == "V2")
    faq = next(item.passage for item in results if item.passage.doc_id == "FAQ")
    old_global = next(item.passage for item in results if item.passage.doc_id == "V1")
    assert precedence_key(specific_addendum, "IN") > precedence_key(current_global, "IN")
    assert precedence_key(current_global, "IN") > precedence_key(faq, "IN")
    assert precedence_key(current_global, "IN") > precedence_key(old_global, "IN")
    assert reference_date_from_question("What was sick leave in 2025?", today) == date(
        2025, 12, 31
    )

    historical = retriever.retrieve(
        "sick leave allowance in 2025", employee, date(2025, 12, 31), 10
    )
    assert {item.passage.doc_id for item in historical} == {"V1", "IN-ADDENDUM"}

    upcoming = retriever.retrieve(
        "wellbeing days", employee, today, 10
    )
    assert [item.passage.doc_id for item in upcoming] == ["V4"]
    assert upcoming[0].upcoming


def test_context_builder_enforces_exact_corpus_token_limit() -> None:
    employee = Employee("E1", "A", "a@example.test", "1", "IN", date(2020, 1, 1), "L1", "employee", None)
    text = "leave policy " + ("supported information " * 200)
    doc = make_document("LONG", text)
    selected = Retriever(doc.passages).retrieve(
        "leave policy supported information", employee, date(2026, 1, 1), 8
    )
    context, used, tokens = build_context(selected, 120)
    assert used
    assert tokens <= 120
