from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.database import Database
from app.ingestion import ingest
from app.llm import AnswerModel, OpenAIAnswerModel
from app.orchestrator import ChatOrchestrator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


def create_app(
    settings: Settings | None = None,
    model: AnswerModel | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ready = False
        try:
            database = Database(configured.db_path)
            database.initialize()
            documents = ingest(configured.corpus_dir, configured.manifest_path)
            answer_model = model or OpenAIAnswerModel(
                configured.openai_api_key, configured.openai_model
            )
            application.state.database = database
            application.state.orchestrator = ChatOrchestrator(
                configured, database, documents, answer_model
            )
            application.state.ready = True
            logging.getLogger(__name__).info(
                "service_ready document_count=%d", len(documents)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            logging.getLogger(__name__).error(
                "service_startup_failed error_type=%s", type(exc).__name__
            )
            raise
        yield
        application.state.ready = False

    application = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def invalid_request_handler(
        request: Request, exception: RequestValidationError
    ) -> JSONResponse:
        del request, exception
        return JSONResponse(
            status_code=400,
            content={"detail": "The request body is missing or invalid."},
        )

    @application.get("/health")
    async def health() -> JSONResponse:
        if not getattr(application.state, "ready", False):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(status_code=200, content={"status": "ok"})

    @application.post("/chat")
    async def chat(
        body: ChatRequest,
        employee_id: Annotated[str | None, Header(alias="X-Employee-Id")] = None,
    ) -> dict[str, object]:
        if not employee_id:
            raise HTTPException(status_code=401, detail="A valid employee ID is required.")
        database: Database = application.state.database
        employee = database.get_employee(employee_id)
        if employee is None:
            raise HTTPException(status_code=401, detail="A valid employee ID is required.")
        if not database.bind_session(body.session_id, employee.employee_id):
            raise HTTPException(
                status_code=403,
                detail="This session belongs to a different employee.",
            )
        response = application.state.orchestrator.handle(
            employee, body.session_id, body.message
        )
        logging.getLogger(__name__).info(
            "chat_handled request_id=%s employee_id=%s route=%s tool_count=%d",
            response.request_id,
            employee.employee_id,
            response.route,
            len(response.tool_calls),
        )
        return {
            "request_id": response.request_id,
            "session_id": response.session_id,
            "message": response.message,
            "route": response.route,
            "citations": [
                {"doc_id": item.doc_id, "locator": item.locator}
                for item in response.citations
            ],
            "tool_calls": [
                {"name": item.name, "args": item.args, "status": item.status}
                for item in response.tool_calls
            ],
            "pending_confirmation": response.pending_confirmation,
        }

    return application


app = create_app()
