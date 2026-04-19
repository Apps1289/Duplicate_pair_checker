import json
import os
import subprocess
from typing import Any, Literal

from fastapi import FastAPI
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

JS_RUNNER_TIMEOUT_SECONDS = max(float(os.getenv("JS_RUNNER_TIMEOUT_MS", "2000")) / 1000.0, 1.0)


class RunRequest(BaseModel):
    code: str


class ExecutionEvent(BaseModel):
    event: EventType
    line: int | None = None
    function: str | None = None
    call_stack: list[str] = Field(default_factory=list)
    locals: dict[str, Any] = Field(default_factory=dict)
    stdout: str | None = None
    message: str | None = None


class RunResponse(BaseModel):
    events: list[ExecutionEvent]


app = FastAPI(title="runner-js", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "runner-js"}


@app.post("/run", response_model=RunResponse)
def run_js(req: RunRequest) -> RunResponse:
    proc = subprocess.run(
        ["node", "trace.js"],
        input=json.dumps({"code": req.code}),
        text=True,
        capture_output=True,
        timeout=JS_RUNNER_TIMEOUT_SECONDS,
        check=False,
    )
    if proc.returncode != 0:
        return RunResponse(events=[ExecutionEvent(event="error", message=proc.stderr or "JS runner failed")])

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return RunResponse(events=[ExecutionEvent(event="error", message="Invalid JS trace output")])

    events: list[ExecutionEvent] = []
    for item in payload.get("events", []):
        try:
            events.append(ExecutionEvent(**item))
        except Exception:
            events.append(ExecutionEvent(event="error", message="Malformed JS event"))

    if not events:
        events = [ExecutionEvent(event="end", message="No events emitted")]
    return RunResponse(events=events)
