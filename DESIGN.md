# Technical design

## Scope and assumptions

The assignment is a constrained PeopleOps assistant, not a general HR agent. It needs policy answers grounded in an eligible corpus and a small set of leave operations. The supplied manifest/database are treated as the source of truth for document metadata and employee/leave data. No model-specific requirement or formal quality thresholds were supplied, so the implementation defaults to OpenAI `gpt-4o-mini` and makes the evidence/token limits and evaluation checks explicit.

## Components and request flow

1. **FastAPI boundary** validates the request shape, employee header, and session ownership.
2. **Ingestion** parses the supported document formats and OCRs image-based PDF pages. It retains document ID and passage locator metadata.
3. **Retriever** first filters by access level, region, effective date, and query time intent. It then ranks passages lexically, applies version precedence, and packs evidence under the configured context cap (at most 3,000 tokens).
4. **Orchestrator** decides whether the request is a policy lookup, leave read, write proposal/confirmation, or refusal/clarification.
5. **Answer model** receives only redacted user text and eligible, bounded evidence. Its structured output is validated and citations must resolve to selected passage locators.
6. **Leave tools** receive typed/validated arguments and enforce authorization themselves. Mutations use SQLite transactions and recheck balance, overlap, and request status at commit time.

```text
Client -> API/Auth/Session -> Orchestrator -> Retrieval -> Answer model
                                 |                `-> eligible citations
                                 `-> Leave tools -> SQLite
```

No employee-controlled query is interpolated into SQL. HR/manager access is explicit in the tools; employee role and region come from the database, never from chat text. The model cannot access database connections or choose an employee identity.

## Prompt and failure strategy

The system prompt separates instruction from evidence and treats retrieved content as untrusted data. It requires abstention when evidence does not support a response and citations from the supplied source set. Policy context is not silently replaced with unrestricted results if no eligible evidence is found. Invalid inputs return client errors; unauthorized identity/session access returns 401/403; unavailable model/database or invalid startup inputs surface as explicit failures. Errors are not converted into plausible HR answers.

Personal names, emails, and phone numbers are redacted before model submission and output. Logs intentionally exclude message text and retrieved evidence. API-key management, strong end-user authentication, TLS, retention, and audit controls remain deployment requirements.

## Evaluation

The deterministic dataset has seven representative cases. The offline runner checks expected source recall, future-document set, evidence terms, restricted-document exclusion, and token budget. With the explicit `--mock-ocr` option (required locally because Tesseract is unavailable), acceptance is 7/7 cases with zero access leaks and zero token-limit violations. The scanned memo itself is not covered by that OCR-stubbed run. Unit/API tests passed 21/21. A compile check passed.

Live mode additionally checks expected answer terms and citation identity/locator against retrieved sources; it records latency and usage tokens. When both model token rates are provided, it estimates cost. Live evaluation was not run because no API key was configured. Expected-term matching is not a robust semantic correctness or hallucination metric; human review and a larger, adjudicated benchmark are necessary for meaningful release quality.

## Trade-offs and follow-up

Lexical retrieval and SQLite minimize setup and make this assignment easy to reproduce, but they are not appropriate for broad semantic search or horizontal write scaling. Effective-date/version selection includes a text-overlap heuristic; a curated precedence graph and conflict-focused tests are preferred next. Before production, replace the assignment header identity with verified identity-provider claims, add rate limiting, migrations, metrics/traces, database backups and retention, and increase the evaluation set with human-scored relevance, completeness, and faithfulness.
