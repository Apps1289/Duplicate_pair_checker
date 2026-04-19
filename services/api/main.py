import os
from typing import Any, Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

EventType = Literal[
    "line_exec",
    "var_update",
    "call",
    "return",
    "stdout",
    "end",
    "error",
]


class ExecutionEvent(BaseModel):
    event: EventType
    line: int | None = None
    function: str | None = None
    call_stack: list[str] = Field(default_factory=list)
    locals: dict[str, Any] = Field(default_factory=dict)
    stdout: str | None = None
    message: str | None = None


class ExecuteRequest(BaseModel):
    language: Literal["python", "javascript"]
    code: str


class ExecuteResponse(BaseModel):
    language: str
    events: list[ExecutionEvent]


RUNNER_ENDPOINTS = {
    "python": os.getenv("PYTHON_RUNNER_URL", "http://runner-python:8001") + "/run",
    "javascript": os.getenv("JS_RUNNER_URL", "http://runner-js:8002") + "/run",
}

app = FastAPI(title="viz-compiler-api", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_events(payload: dict[str, Any]) -> list[ExecutionEvent]:
    events = payload.get("events", [])
    normalized: list[ExecutionEvent] = []
    for item in events:
        try:
            normalized.append(ExecutionEvent(**item))
        except Exception:
            normalized.append(
                ExecutionEvent(event="error", message="Runner emitted invalid event payload")
            )
    if not normalized:
        normalized.append(ExecutionEvent(event="end", message="No events emitted"))
    return normalized


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api"}


@app.post("/execute", response_model=ExecuteResponse)
def execute(request: ExecuteRequest) -> ExecuteResponse:
    endpoint = RUNNER_ENDPOINTS.get(request.language)
    if not endpoint:
        raise HTTPException(status_code=400, detail="Unsupported language")

    try:
        response = requests.post(
            endpoint,
            json={"code": request.code},
            timeout=float(os.getenv("RUNNER_TIMEOUT_SECONDS", "10")),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Runner unavailable: {exc}") from exc

    return ExecuteResponse(language=request.language, events=normalize_events(response.json()))
