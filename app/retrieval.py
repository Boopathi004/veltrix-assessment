from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime
from typing import Callable

import tiktoken

from app.models import Employee, Passage, RetrievedPassage

AUTHORITY_TYPES = {"policy", "addendum", "memo", "handbook", "calendar"}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
QUERY_STOPWORDS = {
    "leave",
    "days",
    "day",
    "working",
    "paid",
    "policy",
    "employee",
    "employees",
    "allowance",
    "company",
    "veltrix",
    "how",
    "what",
    "are",
    "the",
    "for",
    "in",
    "on",
    "of",
    "do",
    "i",
    "my",
    "is",
    "was",
    "will",
    "to",
    "by",
    "hr",
    "many",
    "does",
    "did",
    "provide",
    "global",
    "receive",
    "observed",
    "which",
    "introduced",
    "can",
    "have",
    "has",
    "had",
    "a",
    "an",
    "and",
    "with",
    "from",
    "when",
    "where",
    "much",
    "to",
    "will",
    "would",
    "be",
    "as",
    "at",
    "into",
    "their",
    "for",
    "did",
}
COMPARISON_STOPWORDS = QUERY_STOPWORDS | {
    "annual",
    "sick",
    "holiday",
    "holidays",
    "india",
    "germany",
    "united",
    "states",
    "bengaluru",
    "austin",
    "berlin",
}
COMPARISON_STOPWORDS.discard("leave")


def can_read_document(employee: Employee, access_level: str) -> bool:
    if access_level == "all":
        return True
    if access_level == "managers":
        return employee.role in {"manager", "hr"}
    if access_level == "hr":
        return employee.role == "hr"
    return False


def precedence_key(passage: Passage, employee_region: str) -> tuple[int, int, int]:
    return (
        int(passage.doc_type in AUTHORITY_TYPES),
        int("GLOBAL" not in passage.regions),
        passage.effective_date.toordinal(),
    )


def reference_date_from_question(question: str, today: date) -> date:
    explicit = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", question)
    if explicit:
        requested = date.fromisoformat(explicit.group(0))
        return requested if requested < today else today
    month = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:,?\s+(20\d{2}))?\b",
        question,
        re.IGNORECASE,
    )
    if month:
        year = int(month.group(3) or today.year)
        requested = datetime.strptime(
            f"{month.group(1)} {month.group(2)} {year}", "%B %d %Y"
        ).date()
        return requested if requested < today else today
    year = re.search(r"\b(20\d{2})\b", question)
    if year:
        requested = date(int(year.group(1)), 12, 31)
        return requested if requested < today else today
    return today


class Retriever:
    def __init__(self, passages: tuple[Passage, ...]) -> None:
        self._passages = passages
        self._tokens = tuple(
            self._tokenize(f"{passage.title} {passage.text}") for passage in passages
        )
        self._lengths = tuple(len(tokens) for tokens in self._tokens)
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        self._idf = {
            token: math.log(1 + (len(self._tokens) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def retrieve(
        self,
        question: str,
        employee: Employee,
        reference_date: date,
        top_k: int,
    ) -> tuple[RetrievedPassage, ...]:
        all_query_terms = self._tokenize(question)
        query_terms = [
            term
            for term in all_query_terms
            if term not in QUERY_STOPWORDS
        ]
        if not query_terms:
            return ()
        calendar_question = bool(
            set(query_terms).intersection({"holiday", "holidays", "calendar", "public"})
        )
        restricted_question = "restricted" in query_terms

        candidates: list[RetrievedPassage] = []
        current_candidates: list[RetrievedPassage] = []
        future_candidates: list[RetrievedPassage] = []
        for passage, tokens, length in zip(
            self._passages, self._tokens, self._lengths
        ):
            if not can_read_document(employee, passage.access_level):
                continue
            if "GLOBAL" not in passage.regions and employee.region not in passage.regions:
                continue
            upcoming = passage.effective_date > reference_date
            score = self._score(all_query_terms, tokens, length)
            if (
                calendar_question
                and passage.doc_type == "calendar"
                and (
                    restricted_question
                    or "restricted" not in passage.locator.casefold()
                )
                and len(
                    re.findall(
                        rf"\b\d{{1,2}}\s+(?:{MONTHS})\s+20\d{{2}}\b",
                        passage.text,
                        re.IGNORECASE,
                    )
                )
                >= 2
            ):
                score += 100
            if score > 0 and query_terms and set(tokens).intersection(query_terms):
                result = RetrievedPassage(passage, score, upcoming)
                (future_candidates if upcoming else current_candidates).append(result)

        comparison_terms = set(all_query_terms) - COMPARISON_STOPWORDS
        if not comparison_terms:
            comparison_terms = set(query_terms)
        required_overlap = min(2, len(comparison_terms))
        quantitative_question = bool(
            re.search(
                r"\b(how many|how much|number|days?|weeks?|months?|limit|rate)\b",
                question,
                re.IGNORECASE,
            )
        )
        for future in future_candidates:
            future_terms = set(
                self._tokenize(f"{future.passage.title} {future.passage.text}")
            ) & comparison_terms
            competing_current = [
                current
                for current in current_candidates
                if (
                    len(
                        future_terms
                        & set(
                            self._tokenize(
                                current.passage.title + " " + current.passage.text
                            )
                        )
                    )
                    >= required_overlap
                    and (
                        not quantitative_question
                        or (
                            re.search(r"\b\d+(?:\.\d+)?\b", future.passage.text)
                            and re.search(
                                r"\b\d+(?:\.\d+)?\b", current.passage.text
                            )
                        )
                    )
                )
            ]
            if not competing_current or all(
                precedence_key(future.passage, employee.region)
                > precedence_key(current.passage, employee.region)
                for current in competing_current
            ):
                candidates.append(future)
        candidates.extend(current_candidates)

        candidates.sort(
            key=lambda result: (
                -result.score,
                tuple(-value for value in precedence_key(result.passage, employee.region)),
                result.passage.doc_id,
                result.passage.locator,
            )
        )
        return tuple(candidates[:top_k])

    def _score(self, query_terms: list[str], document_terms: list[str], length: int) -> float:
        term_counts = Counter(document_terms)
        score = 0.0
        for term in set(query_terms):
            frequency = term_counts.get(term, 0)
            if not frequency:
                continue
            inverse_document_frequency = self._idf.get(term, 0.0)
            denominator = frequency + 1.2 * (
                1 - 0.75 + 0.75 * length / max(self._average_length, 1)
            )
            score += inverse_document_frequency * frequency * 2.2 / denominator
        return score

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        return [
            token
            for token in TOKEN_PATTERN.findall(value.casefold())
            if len(token) > 1 and not re.fullmatch(r"20\d{2}", token)
        ]


def build_context(
    results: tuple[RetrievedPassage, ...],
    token_limit: int,
    redact_text: Callable[[str], str] | None = None,
) -> tuple[str, tuple[RetrievedPassage, ...], int]:
    encoding = tiktoken.get_encoding("cl100k_base")
    selected: list[RetrievedPassage] = []
    blocks: list[str] = []
    total_tokens = 0

    for result in results:
        passage = result.passage
        status = "upcoming" if result.upcoming else "applicable"
        header = (
            f"[doc_id={passage.doc_id}; title={passage.title}; "
            f"type={passage.doc_type}; regions={','.join(passage.regions)}; "
            f"effective_date={passage.effective_date.isoformat()}; "
            f"status={status}; locator={passage.locator}]"
        )
        text = redact_text(passage.text) if redact_text else passage.text
        block = f"{header}\n{text}"
        candidate_context = "\n\n".join([*blocks, block])
        candidate_tokens = len(encoding.encode(candidate_context))
        if candidate_tokens > token_limit:
            if not blocks:
                header_tokens = encoding.encode(f"{header}\n")
                available_text_tokens = token_limit - len(header_tokens)
                if available_text_tokens <= 0:
                    break
                text_tokens = encoding.encode(text)[:available_text_tokens]
                block = f"{header}\n{encoding.decode(text_tokens)}"
                candidate_context = block
                candidate_tokens = len(encoding.encode(candidate_context))
                while candidate_tokens > token_limit and text_tokens:
                    text_tokens = text_tokens[:-1]
                    block = f"{header}\n{encoding.decode(text_tokens)}"
                    candidate_context = block
                    candidate_tokens = len(encoding.encode(candidate_context))
            else:
                continue
        if candidate_tokens > token_limit:
            continue
        blocks.append(block)
        selected.append(result)
        total_tokens = candidate_tokens

    context = "\n\n".join(blocks)
    actual_tokens = len(encoding.encode(context))
    if actual_tokens > token_limit or actual_tokens != total_tokens:
        raise RuntimeError("Corpus context token accounting failed")
    return context, tuple(selected), actual_tokens
