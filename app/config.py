from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    manifest_path: Path
    db_path: Path
    today: date
    llm_provider: str
    openai_api_key: str
    openai_model: str
    log_level: str
    max_context_tokens: int
    retrieval_top_k: int
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        today_value = os.getenv("APP_TODAY", "").strip()
        try:
            today = date.fromisoformat(today_value) if today_value else date.today()
        except ValueError as exc:
            raise ValueError("APP_TODAY must use YYYY-MM-DD format") from exc

        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        if provider != "openai":
            raise ValueError("LLM_PROVIDER must be openai")

        max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
        retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "16"))
        input_cost = os.getenv("INPUT_COST_PER_MILLION", "").strip()
        output_cost = os.getenv("OUTPUT_COST_PER_MILLION", "").strip()
        if not 1 <= max_context_tokens <= 3000:
            raise ValueError("MAX_CONTEXT_TOKENS must be between 1 and 3000")
        if retrieval_top_k < 1:
            raise ValueError("RETRIEVAL_TOP_K must be positive")
        try:
            input_cost_value = float(input_cost) if input_cost else None
            output_cost_value = float(output_cost) if output_cost else None
        except ValueError as exc:
            raise ValueError("Token cost rates must be numeric") from exc
        if any(
            value is not None and value < 0
            for value in (input_cost_value, output_cost_value)
        ):
            raise ValueError("Token cost rates cannot be negative")

        return cls(
            corpus_dir=Path(os.getenv("CORPUS_DIR", "./data/corpus")),
            manifest_path=Path(os.getenv("MANIFEST_PATH", "./data/manifest.json")),
            db_path=Path(os.getenv("DB_PATH", "./data/veltrix.db")),
            today=today,
            llm_provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            max_context_tokens=max_context_tokens,
            retrieval_top_k=retrieval_top_k,
            input_cost_per_million=input_cost_value,
            output_cost_per_million=output_cost_value,
        )
