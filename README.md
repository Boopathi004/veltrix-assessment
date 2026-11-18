# Veltrix PeopleOps Assistant

## Executive summary

This submission implements the assignment as a small, auditable FastAPI service rather than a general-purpose HR chatbot. It combines access-filtered retrieval over the supplied policy corpus with narrowly scoped SQLite leave tools. The application treats employee identity and permissions as server-side facts, cites policy passages, requires explicit confirmation before leave mutations, and bounds the evidence sent to the model. The evaluation separates deterministic retrieval/access checks from optional live-model answer checks.

## Assignment analysis

| Area | Interpretation |
|---|---|
| Core objective | Answer employee PeopleOps questions from current, applicable policy evidence and provide safe leave self-service. |
| Required deliverables | `POST /chat`, `GET /health`, corpus ingestion including scanned PDF text, precedence/effective-date handling, citations, five leave tools, role/region restrictions, confirmation before write actions, a 3,000-token evidence cap, and Docker Compose startup. |
| Implicit expectations | Do not expose restricted HR records or other employees' leave data; distinguish upcoming policy from policy in force; avoid unsupported answers; provide reproducible setup and observable failures. |
| Assumptions | The supplied SQLite database is authoritative for employee identity, role, region, balances, and requests. The supplied manifest is authoritative for document metadata. A write request means creating a pending leave request, not approving it. `APP_TODAY` can pin dates for reproducible evaluation. |
| Important gaps/risks | No assignment-defined model, exact answer schema, or scoring rubric was supplied. OCR quality on the scanned memo depends on Tesseract being installed. Policy precedence is metadata- and text-overlap-based rather than a human-curated conflict graph. Live model cost, latency, and answer quality require an API key and are not represented by offline retrieval results. |

## Architecture and data flow

```text
Client
  | POST /chat (session_id, message; X-Employee-Id)
  v
FastAPI validation and identity lookup
  |-- session ownership check (SQLite)
  v
Chat orchestrator
  |-- policy question --> eligibility/access filter --> BM25-style ranker
  |                         |--> precedence + date selection
  |                         |--> token-budget context + citations
  |                         v
  |                     OpenAI JSON answer model
  |
  |-- leave question/action --> typed tool dispatch --> authorization + SQLite
  |                                                |--> read results
  |                                                `--> proposal, confirm, transact
  v
Validated response: message, route, citations, tool calls, confirmation state
```

At startup, the service validates the manifest and parses PDF, DOCX, HTML, and Markdown files. PyMuPDF and Tesseract provide OCR for image-based PDF pages; parsed passages retain document IDs and locators. Each chat identifies an employee using the supplied employee ID, binds a session to that employee, and filters policy documents by access level, region, and effective date *before* text reaches the model. The retriever selects a bounded set of passages, applies policy version precedence, and computes a conservative `cl100k_base` token budget that includes evidence metadata. Leave records are accessed through parameterized SQLite queries and tool-level authorization.

### Model, retrieval, and prompt strategy

- Retrieval is local BM25-style lexical ranking; it requires no embedding service or vector database and is reproducible with the supplied corpus.
- The default answer model is `gpt-4o-mini`, configurable through `OPENAI_MODEL`. The model is asked for a structured answer and citations; returned citations are checked against selected passages.
- The system prompt is kept in `app/prompts/hr_system.txt`. It instructs the model to treat evidence as untrusted, not follow instructions embedded in retrieved content, abstain when evidence is insufficient, and cite only the provided evidence.
- Tool routing and write confirmation are orchestrator responsibilities. The language model cannot grant access or directly change the database.
- The allow-listed leave tools are `get_leave_balance`, `list_leave_requests`, `get_team_leave` (manager only), `apply_leave`, and `cancel_leave`.

### Guardrails and operational behavior

- Pydantic rejects extra request properties and bounds message/session lengths.
- Missing/unknown identities return 401; cross-employee session reuse returns 403.
- Document access is filtered before retrieval context is constructed. Leave tools independently enforce self/manager/HR authorization.
- Names, emails, and phone numbers are redacted from model inputs and generated output.
- Leave application requires a persisted proposal and a separate confirmation; balance/request changes are atomic, revalidated inside an immediate SQLite transaction, and rolled back on failure.
- The service refuses restricted or unsupported requests rather than attempting an ungrounded answer. Startup parsing/configuration failures are logged and surfaced; `/health` is 503 until startup has completed.
- Logs contain request ID, employee ID, route, and tool count, not chat text. Avoid adding message or evidence logging without a privacy review.

### Security, privacy, latency, cost, and observability

The employee ID header is an assignment-level identity mechanism, not production authentication. A real deployment must replace it with verified identity-provider claims, enforce TLS, protect secrets, restrict network exposure, and define retention/deletion policies. SQLite is appropriate for this single-instance assignment but would need a transactional shared database for horizontal scale. Local lexical retrieval avoids embedding costs; model latency and price depend on the configured provider/model. The live evaluator reports measured request latency and token-derived estimated cost when both per-million token rates are configured. Logs and `/health` provide basic operational visibility; metrics, traces, alerting, rate limits, and audit retention are future deployment work.

## Setup and execution

Requirements: Python 3.10 or later; Docker Compose is optional. Tesseract OCR is needed for scanned PDFs. Windows users can run the Python commands below in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/prepare_data.py
```

Set `OPENAI_API_KEY` in `.env` to enable generated answers. Do not commit `.env`. Without a key, the service can start, but model-backed answers report an explicit unavailable-model error; retrieval/evaluation and tests can still run.

The repository database is a synthetic anonymized seed: employee names, email addresses, and phone numbers are replaced with example.test values. Run `python scripts/create_synthetic_seed.py` to regenerate both seed locations from `data/veltrix.db`; do not use real employee records in a public repository.

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check: `GET http://127.0.0.1:8000/health`.

Example request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/chat `
  -Headers @{"X-Employee-Id" = "E1001"} `
  -ContentType "application/json" `
  -Body '{"session_id":"demo-1","message":"How do I request annual leave?"}'
```

The response contains `request_id`, `session_id`, `message`, `route`, `citations`, `tool_calls`, and `pending_confirmation`. Use an employee ID present in the supplied database. For other shells, the equivalent is a JSON `POST /chat` with `X-Employee-Id`.

### Docker Compose

Configure `.env`, including `OPENAI_API_KEY`, then run:

```powershell
docker compose up --build
```

The image includes the assignment corpus, manifest, and initial SQLite database. The named `peopleops-data` volume is initialized from those image assets and persists database updates. Compose startup was not executable in the development environment because Docker was unavailable.

## Evaluation and test results

Run the deterministic retrieval/evidence checks and test suite:

```powershell
python -m pytest -q
python scripts/evaluate.py --mock-ocr
python -m compileall -q app tests scripts
```

`--mock-ocr` explicitly stubs the scanned page for environments without Tesseract; it does not evaluate OCR quality. With Tesseract installed, omit the flag to exercise real OCR ingestion. To evaluate generated answers, citations, answer latency, and cost, configure `OPENAI_API_KEY`. For cost estimation, set both `INPUT_COST_PER_MILLION` and `OUTPUT_COST_PER_MILLION` to the chosen model's current rates, then run `python scripts/evaluate.py --live` (add `--mock-ocr` only if Tesseract is unavailable).

| Metric | Offline acceptance criterion | Measured result |
|---|---|---|
| Evidence/retrieval cases | All expected evidence, applicable/upcoming documents, and terms selected | 7/7 passed (`--mock-ocr`) |
| Access control | No excluded restricted document in selected context | 0 leaks |
| Corpus token limit | Every request context is at most 3,000 tokens | 0 violations |
| Automated tests | All unit and API integration tests pass | 21 passed |
| Compile check | App, tests, and scripts compile | Passed |
| OCR | Scanned memo parsed from image pages | Not verified: Tesseract unavailable locally; mocked only for offline retrieval cases |
| Live answer correctness/citation grounding | All live cases meet expected-term and evidence-backed citation checks | Not run: no API key configured |
| Live latency and cost | Report model request time; estimate cost from actual usage and configured rates | Not measured |

The seven-case evidence set covers current and upcoming global policy, a historical question, region-specific policy and holidays, an upcoming benefit, and restricted compensation data. Offline checks are deterministic and useful for retrieval/access regressions; they are not a semantic hallucination benchmark. Live expected-term/citation checks are narrow proxies and do not replace human review or a calibrated faithfulness judge.

## Design decisions and trade-offs

- **Local BM25 over a vector database:** fewer dependencies and deterministic results for this small corpus; lexical mismatch and semantic paraphrases can lower recall.
- **SQLite tools rather than LLM-generated SQL:** explicit allow-listed operations, parameterized queries, and independent authorization checks; SQLite limits multi-instance scaling.
- **Metadata/date eligibility before ranking:** prevents ineligible content from becoming model context; precedence between conflicting documents still uses a heuristic.
- **Confirmation proposal before writes:** makes user intent explicit and stores the exact validated action; confirmation is revalidated against live state before mutation.
- **Single service and local ingestion:** easy to run and review; no asynchronous indexing pipeline or distributed cache is justified by assignment scale.

## Limitations and next steps

- The employee ID header is not suitable as production authentication.
- OCR output for the scanned memo is not verified against a locally installed Tesseract binary; tests use mocked OCR. Docker includes Tesseract, but image build/runtime could not be exercised here.
- Policy conflict resolution is heuristic and should evolve into an explicitly reviewed precedence map with regression cases.
- Evaluation is small; add independently authored question/answer references, adversarial and ambiguity cases, reviewer-rated relevance/completeness/faithfulness, and versioned model runs.
- Add deployment-grade authentication, migrations, structured metrics/tracing, rate limiting, backup/retention controls, and a shared transactional database before scaling.

See [DESIGN.md](./DESIGN.md) for a condensed technical design.
