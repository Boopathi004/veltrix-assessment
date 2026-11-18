from __future__ import annotations

import logging
import json
import re
from datetime import date, datetime, timedelta
from uuid import uuid4

from app.config import Settings
from app.database import Database
from app.llm import AnswerModel, LLMError, LLMUnavailable, ModelAnswer
from app.models import (
    ChatResponse,
    Citation,
    Employee,
    ToolCall,
)
from app.retrieval import (
    Retriever,
    build_context,
    reference_date_from_question,
)
from app.tools import LeaveTools, ToolError

logger = logging.getLogger(__name__)
HR_TERMS = re.compile(
    r"\b(leave|holiday|vacation|benefit|policy|employee|hr|salary|compensation|"
    r"pay|expense|travel|manager|hybrid|remote|working hours|work time|"
    r"accrual|carry forward|parental|sick|annual|probation|notice|conduct|"
    r"security|learning|development|public holiday|wellbeing)\b",
    re.IGNORECASE,
)
PROMPT_ATTACK = re.compile(
    r"(ignore|disregard|override|bypass).{0,80}(instructions|rules|prompt|security)|"
    r"(reveal|show|print|repeat|provide).{0,80}(system prompt|hidden prompt|"
    r"developer message|secret instruction)|"
    r"\b(system prompt|developer message|hidden instructions)\b",
    re.IGNORECASE,
)
APPLY_INTENT = re.compile(
    r"\b(?:i\s+(?:want|need|would like)\s+to\s+(?:take|request)|"
    r"(?:request|book|schedule|apply for))\b.{0,80}\b(?:leave|time off|vacation)\b|"
    r"\b(?:take|request|book|schedule)\b.{0,50}\b(?:annual|sick)\s+leave\b",
    re.IGNORECASE,
)
CANCEL_INTENT = re.compile(r"\b(cancel|cancellation)\b.{0,50}\b(leave|request)\b", re.I)
EMPLOYEE_ID = re.compile(r"\bE\d{3,}\b", re.IGNORECASE)
REQUEST_ID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
ISO_DATE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)


class ChatOrchestrator:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        documents,
        model: AnswerModel,
    ) -> None:
        self.settings = settings
        self.database = database
        self.documents = documents
        self.model = model
        self.retriever = Retriever(
            tuple(passage for document in documents for passage in document.passages)
        )
        self.leave_tools = LeaveTools(database, documents, settings.today)

    def handle(
        self, employee: Employee, session_id: str, message: str
    ) -> ChatResponse:
        normalized = message.strip()
        if not normalized:
            return self._response(session_id, "Please send a question or request.", "clarify")
        if PROMPT_ATTACK.search(normalized):
            return self._response(
                session_id,
                "I can help with Veltrix HR policies, holidays, and permitted leave information.",
                "refuse",
            )
        if re.search(r"\bhalf[- ]day\b", normalized, re.IGNORECASE):
            return self._response(
                session_id,
                "Half-day leave is not supported; leave requests must cover whole days.",
                "refuse",
            )
        if self._is_bulk_request(normalized):
            return self._response(
                session_id,
                "I can’t provide leave data for many employees at once.",
                "refuse",
            )

        pending = self.database.get_proposal(session_id)
        if pending and self._is_confirmation(normalized):
            return self._confirm(employee, session_id, pending)
        if pending:
            adjusted = self._adjust_proposal(pending, normalized)
            self.database.clear_proposal(session_id)
            if adjusted:
                return self._propose(employee, session_id, *adjusted)

        if CANCEL_INTENT.search(normalized):
            return self._handle_cancel(employee, session_id, normalized)
        is_how_to_question = bool(
            re.match(r"^\s*(how|where|what|can i|am i)\b", normalized, re.IGNORECASE)
        )
        if APPLY_INTENT.search(normalized) and not is_how_to_question:
            return self._handle_apply(employee, session_id, normalized)
        if self._is_team_leave_query(normalized) and employee.role == "manager":
            dates = self._extract_dates(normalized, self.settings.today)
            if len(dates) < 2:
                return self._response(
                    session_id,
                    "Please specify the start and end dates for the team leave search.",
                    "clarify",
                )
        if self._is_read_query(normalized) and not self._is_team_leave_query(normalized):
            has_explicit_target = bool(
                EMPLOYEE_ID.search(normalized)
                or self.database.find_employee_id_by_name(normalized)
            )
            refers_to_self = bool(
                re.search(r"\b(my|me|i|mine|our|us)\b", normalized, re.IGNORECASE)
            )
            if not has_explicit_target and not refers_to_self:
                return self._response(
                    session_id,
                    "Please say whether you mean your own leave data or provide one employee ID.",
                    "clarify",
                )
        read_intent = self._read_intent(employee, normalized)
        if read_intent:
            tool_call, tool_result = read_intent
            if tool_result is None:
                return self._tool_error(
                    session_id, tool_call.name, tool_call.args, "Please provide the missing information."
                )
            needs_documents = bool(
                re.search(
                    r"\b(policy|carry forward|accrual|accrue|entitlement|how many days "
                    r"are allowed|holiday rule|will i have by|by december|"
                    r"by year[- ]end|future balance)\b",
                    normalized,
                    re.IGNORECASE,
                )
            )
            if needs_documents:
                return self._hybrid(session_id, tool_call, tool_result, normalized, employee)
            return self._response(
                session_id,
                tool_result.message,
                "tool",
                tool_calls=(tool_call,),
            )

        if not HR_TERMS.search(normalized):
            return self._response(
                session_id,
                "I can help with Veltrix HR policies, holidays, and permitted leave information.",
                "refuse",
            )
        return self._rag(session_id, normalized, employee)

    def _handle_apply(
        self, employee: Employee, session_id: str, message: str
    ) -> ChatResponse:
        leave_type_match = re.search(r"\b(annual|sick)\b", message, re.IGNORECASE)
        leave_type = leave_type_match.group(1).lower() if leave_type_match else ""
        dates = self._extract_dates(message, self.settings.today)
        if len(dates) >= 2:
            start_date, end_date = dates[0], dates[1]
        elif len(dates) == 1 and re.search(
            r"\b(on|for|next\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|next\s+weekday)\b",
            message,
            re.IGNORECASE,
        ):
            start_date = end_date = dates[0]
        else:
            start_date = end_date = None
        missing = []
        if not leave_type:
            missing.append("the leave type (annual or sick)")
        if start_date is None or end_date is None:
            missing.append("the start and end dates")
        if missing:
            return self._response(
                session_id,
                f"Please specify {', and '.join(missing)} so I can check the request.",
                "clarify",
            )
        if start_date < self.settings.today:
            return self._response(
                session_id,
                "That start date has already passed. Please provide a future date.",
                "clarify",
            )
        args = {
            "leave_type": leave_type,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        return self._propose(employee, session_id, "apply_leave", args)

    def _handle_cancel(
        self, employee: Employee, session_id: str, message: str
    ) -> ChatResponse:
        match = REQUEST_ID.search(message)
        if not match:
            return self._response(
                session_id,
                "Please provide the request ID you want to cancel.",
                "clarify",
            )
        args = {"request_id": match.group(0)}
        return self._propose(employee, session_id, "cancel_leave", args)

    def _propose(
        self,
        employee: Employee,
        session_id: str,
        name: str,
        args: dict[str, str],
    ) -> ChatResponse:
        try:
            if name == "apply_leave":
                start, end, days = self.leave_tools.validate_apply(employee, args)
                message = (
                    f"Proposed {args['leave_type']} leave from {start} through {end} "
                    f"({days} working days). Please confirm to submit this request."
                )
            else:
                preview = self.leave_tools.validate_cancel(employee, args["request_id"])
                message = f"{preview.message} Please confirm to cancel it."
            self.database.set_proposal(session_id, name, args)
        except ToolError as exc:
            return self._tool_error(session_id, name, args, str(exc))
        return self._response(
            session_id,
            message,
            "tool",
            tool_calls=(ToolCall(name, args, "pending_confirmation"),),
            pending_confirmation=True,
        )

    def _confirm(
        self,
        employee: Employee,
        session_id: str,
        proposal: tuple[str, dict[str, str]],
    ) -> ChatResponse:
        name, args = proposal
        self.database.clear_proposal(session_id)
        try:
            if name == "apply_leave":
                result = self.leave_tools.apply_leave(employee, **args)
            elif name == "cancel_leave":
                result = self.leave_tools.cancel_leave(employee, args["request_id"])
            else:
                return self._tool_error(
                    session_id, name, args, "The pending action is not supported."
                )
        except ToolError as exc:
            return self._tool_error(session_id, name, args, str(exc))
        return self._response(
            session_id,
            result.message,
            "tool",
            tool_calls=(ToolCall(name, args, "success"),),
        )

    def _read_intent(
        self, employee: Employee, message: str
    ) -> tuple[ToolCall, object | None] | None:
        lower = message.casefold()
        target_ids = EMPLOYEE_ID.findall(message)
        if not target_ids:
            named_employee_id = self.database.find_employee_id_by_name(message)
            if named_employee_id:
                target_ids = [named_employee_id]
            elif self._mentions_employee_target(message):
                target_ids = ["__unknown_employee__"]
        if re.search(r"\bteam leave\b", lower):
            if employee.role != "manager":
                return self._make_error_call(
                    "get_team_leave",
                    {"start_date": "", "end_date": ""},
                    "Only managers can view direct-report leave.",
                )
            dates = self._extract_dates(message, self.settings.today)
            if len(dates) < 2:
                return (
                    ToolCall("get_team_leave", {"start_date": "", "end_date": ""}, "error"),
                    None,
                )
            args = {
                "start_date": dates[0].isoformat(),
                "end_date": dates[1].isoformat(),
            }
            return self._execute_read(
                "get_team_leave",
                args,
                lambda: self.leave_tools.get_team_leave(employee, **args),
            )

        employee_id = target_ids[0].upper() if target_ids else employee.employee_id
        if re.search(
            r"\b(balance|how many .*leave days .*have|how much .*leave .*have|"
            r"leave days left)\b",
            lower,
        ):
            leave_type_match = re.search(r"\b(annual|sick)\b", lower)
            args = {"employee_id": employee_id}
            if leave_type_match:
                args["leave_type"] = leave_type_match.group(1)
            return self._execute_read(
                "get_leave_balance",
                args,
                lambda: self.leave_tools.get_leave_balance(
                    employee, employee_id, args.get("leave_type")
                ),
            )
        if re.search(
            r"\b(my|their|his|her)?\s*(leave requests|requests for leave|booked leave|"
            r"approved leave|pending leave)\b",
            lower,
        ):
            status_match = re.search(r"\b(pending|approved|cancelled)\b", lower)
            args = {"employee_id": employee_id}
            if status_match:
                args["status"] = status_match.group(1)
            return self._execute_read(
                "list_leave_requests",
                args,
                lambda: self.leave_tools.list_leave_requests(
                    employee, employee_id, args.get("status")
                ),
            )
        return None

    def _execute_read(
        self, name: str, args: dict[str, str], operation
    ) -> tuple[ToolCall, object | None]:
        try:
            result = operation()
            return ToolCall(name, args, "success"), result
        except ToolError as exc:
            return ToolCall(name, args, "error"), _ErrorResult(str(exc))

    def _make_error_call(
        self, name: str, args: dict[str, str], message: str
    ) -> tuple[ToolCall, object | None]:
        return ToolCall(name, args, "error"), _ErrorResult(message)

    def _tool_error(
        self, session_id: str, name: str, args: dict[str, str], message: str
    ) -> ChatResponse:
        return self._response(
            session_id,
            message,
            "tool",
            tool_calls=(ToolCall(name, args, "error"),),
        )

    def _hybrid(
        self,
        session_id: str,
        tool_call: ToolCall,
        tool_result,
        question: str,
        employee: Employee,
    ) -> ChatResponse:
        tool_data = ""
        if (
            tool_call.status == "success"
            and tool_call.name == "get_leave_balance"
            and re.search(
                r"\b(will i have|by december|by year[- ]end|future balance)\b",
                question,
                re.IGNORECASE,
            )
        ):
            target_id = tool_call.args["employee_id"]
            structured = {
                "current_balance": tool_result.data["balances"],
                "as_of_date": self.settings.today.isoformat(),
                "join_date": self.database.get_join_date(target_id).isoformat(),
            }
            tool_data = json.dumps(structured, separators=(",", ":"))
        rag = self._rag(session_id, question, employee, tool_data)
        if rag.route != "rag":
            return self._response(
                session_id,
                f"{tool_result.message} {rag.message}",
                "tool",
                tool_calls=(tool_call,),
            )
        return self._response(
            session_id,
            f"{tool_result.message}\n\n{rag.message}",
            "hybrid",
            citations=rag.citations,
            tool_calls=(tool_call,),
        )

    def _rag(
        self,
        session_id: str,
        question: str,
        employee: Employee,
        tool_data: str = "",
    ) -> ChatResponse:
        reference_date = reference_date_from_question(question, self.settings.today)
        results = self.retriever.retrieve(
            question,
            employee,
            reference_date,
            self.settings.retrieval_top_k,
        )
        if not results:
            if HR_TERMS.search(question):
                return self._response(
                    session_id,
                    "The available documents do not cover this question. Please contact the HR team.",
                    "no_answer",
                )
            return self._response(
                session_id,
                "I can help with Veltrix HR policies, holidays, and permitted leave information.",
                "refuse",
            )
        context, selected, corpus_tokens = build_context(
            results,
            self.settings.max_context_tokens,
            self.database.redact_personal_data,
        )
        if not selected:
            return self._response(
                session_id,
                "The available documents do not cover this question. Please contact the HR team.",
                "no_answer",
            )
        try:
            safe_question = self.database.redact_personal_data(question)
            answer: ModelAnswer = self.model.answer(
                safe_question, context, corpus_tokens, tool_data
            )
        except (LLMUnavailable, LLMError) as exc:
            logger.error("llm_failure error_type=%s", type(exc).__name__)
            return self._response(
                session_id,
                "I couldn’t retrieve a reliable answer right now. Please contact the HR team.",
                "no_answer",
            )
        allowed = {
            (result.passage.doc_id, result.passage.locator) for result in selected
        }
        citations = tuple(
            citation
            for citation in answer.citations
            if (citation.doc_id, citation.locator) in allowed
        )
        if not citations:
            return self._response(
                session_id,
                "The available documents do not support a reliable answer. Please contact the HR team.",
                "no_answer",
            )
        return self._response(
            session_id,
            self.database.redact_personal_data(answer.answer),
            "rag",
            citations=self._deduplicate(citations),
        )

    @staticmethod
    def _deduplicate(citations: tuple[Citation, ...]) -> tuple[Citation, ...]:
        seen: set[tuple[str, str]] = set()
        unique: list[Citation] = []
        for citation in citations:
            key = (citation.doc_id, citation.locator)
            if key not in seen:
                seen.add(key)
                unique.append(citation)
        return tuple(unique)

    def _response(
        self,
        session_id: str,
        message: str,
        route: str,
        citations: tuple[Citation, ...] = (),
        tool_calls: tuple[ToolCall, ...] = (),
        pending_confirmation: bool = False,
    ) -> ChatResponse:
        return ChatResponse(
            request_id=str(uuid4()),
            session_id=session_id,
            message=message,
            route=route,
            citations=citations,
            tool_calls=tool_calls,
            pending_confirmation=pending_confirmation,
        )

    def _is_bulk_request(self, message: str) -> bool:
        lower = message.casefold()
        if "team leave" in lower:
            return False
        return bool(
            re.search(
                r"\b(everyone|everybody|all employees|all staff|company[- ]wide)\b",
                lower,
            )
        ) and bool(re.search(r"\b(balance|leave|request|employee)\b", lower))

    @staticmethod
    def _is_read_query(message: str) -> bool:
        return bool(
            re.search(
                r"\b(balance|leave requests|requests for leave|booked leave|"
                r"approved leave|pending leave|leave days left|team leave)\b",
                message,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_team_leave_query(message: str) -> bool:
        return bool(re.search(r"\bteam leave\b", message, re.IGNORECASE))

    @staticmethod
    def _mentions_employee_target(message: str) -> bool:
        return bool(
            re.search(
                r"(?:\b(?:for|about|employee)\s+[A-Z][a-z'-]+|"
                r"\b[A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+)?(?:'s|’s)\b)",
                message,
            )
        )

    @staticmethod
    def _is_confirmation(message: str) -> bool:
        return bool(
            re.fullmatch(
                r"\s*(yes|yes please|confirm|confirmed|go ahead|proceed|do it)\s*[.!]?\s*",
                message,
                re.IGNORECASE,
            )
        )

    def _adjust_proposal(
        self, pending: tuple[str, dict[str, str]], message: str
    ) -> tuple[str, dict[str, str]] | None:
        name, old_args = pending
        if name != "apply_leave" or not re.search(
            r"\b(until|through|to|instead|change|make it)\b", message, re.I
        ):
            return None
        dates = self._extract_dates(message, self.settings.today)
        if len(dates) != 1:
            return None
        args = dict(old_args)
        if re.search(r"\b(start|from)\b", message, re.I):
            args["start_date"] = dates[0].isoformat()
        else:
            args["end_date"] = dates[0].isoformat()
        return name, args

    @staticmethod
    def _extract_dates(message: str, today: date) -> list[date]:
        dates: list[date] = []
        for value in ISO_DATE.findall(message):
            try:
                dates.append(date.fromisoformat(value))
            except ValueError:
                return []
        if dates:
            return dates

        lower = message.casefold()
        if "next week" in lower:
            monday = today - timedelta(days=today.weekday()) + timedelta(days=7)
            return [monday, monday + timedelta(days=4)]
        if "next friday" in lower:
            days_ahead = (4 - today.weekday()) % 7 or 7
            target = today + timedelta(days=days_ahead)
            return [target]
        if "tomorrow" in lower:
            return [today + timedelta(days=1)]
        if re.search(r"\btoday\b", lower):
            return [today]

        patterns = (
            rf"\b({MONTHS})\s+(\d{{1,2}})(?:,?\s+(20\d{{2}}))?\b",
            rf"\b(\d{{1,2}})\s+({MONTHS})(?:\s+(20\d{{2}}))?\b",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                groups = match.groups()
                if groups[0].isalpha():
                    month, day_value, year_value = groups
                else:
                    day_value, month, year_value = groups
                year = int(year_value or 2026)
                try:
                    parsed = datetime.strptime(
                        f"{month} {day_value} {year}", "%B %d %Y"
                    ).date()
                except ValueError:
                    return []
                dates.append(parsed)
        if dates:
            return dates

        numeric_dates = re.findall(r"\b(\d{1,2})/(\d{1,2})(?:/(20\d{2}))?\b", message)
        if numeric_dates:
            for first_text, second_text, year_text in numeric_dates:
                first, second = int(first_text), int(second_text)
                if first <= 12 and second <= 12:
                    return []
                month, day_value = (first, second) if first <= 12 else (second, first)
                try:
                    dates.append(date(int(year_text or 2026), month, day_value))
                except ValueError:
                    return []
        return dates


class _ErrorResult:
    def __init__(self, message: str) -> None:
        self.message = message
