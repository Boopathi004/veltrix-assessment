from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from openai import OpenAI, OpenAIError

from app.models import Citation

logger = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    Path(__file__).parent / "prompts" / "hr_system.txt"
).read_text(encoding="utf-8")


class LLMUnavailable(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelAnswer:
    answer: str
    citations: tuple[Citation, ...]
    input_tokens: int = 0
    output_tokens: int = 0


class AnswerModel(Protocol):
    def answer(
        self,
        question: str,
        context: str,
        corpus_tokens: int,
        tool_data: str = "",
    ) -> ModelAnswer: ...


class OpenAIAnswerModel:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def answer(
        self,
        question: str,
        context: str,
        corpus_tokens: int,
        tool_data: str = "",
    ) -> ModelAnswer:
        if not self.api_key:
            raise LLMUnavailable("OPENAI_API_KEY is not configured")
        logger.info("llm_call model=%s corpus_tokens=%d", self.model, corpus_tokens)
        try:
            response = OpenAI(api_key=self.api_key, timeout=20).chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{question}\n\n"
                            f"Authorized structured tool data (not corpus text):\n"
                            f"{tool_data or 'None'}\n\nEvidence:\n{context}"
                        ),
                    },
                ],
            )
        except OpenAIError as exc:
            raise LLMError(type(exc).__name__) from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError("Model returned an empty response")
        try:
            decoded = json.loads(content)
            answer = decoded["answer"]
            raw_citations = decoded["citations"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("answer must be a non-empty string")
            if not isinstance(raw_citations, list):
                raise ValueError("citations must be a list")
            citations_list = []
            for item in raw_citations:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("doc_id"), str)
                    or not isinstance(item.get("locator"), str)
                ):
                    raise ValueError("citation must contain doc_id and locator strings")
                citations_list.append(
                    Citation(doc_id=item["doc_id"], locator=item["locator"])
                )
            citations = tuple(citations_list)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LLMError("Model returned an invalid answer shape") from exc
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        logger.info(
            "llm_usage model=%s input_tokens=%d output_tokens=%d corpus_tokens=%d",
            self.model,
            input_tokens,
            output_tokens,
            corpus_tokens,
        )
        return ModelAnswer(answer.strip(), citations, input_tokens, output_tokens)
