import ast
import io
import sys
from contextlib import redirect_stdout
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

MAX_LOCAL_REPR_CHARS = 200
DISALLOWED_CALLS = {"eval", "exec", "open", "compile", "__import__", "input"}
SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "enumerate": enumerate,
    "Exception": Exception,
}


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


app = FastAPI(title="runner-python", version="0.1.0")


def _safe_locals(frame_locals: dict[str, Any]) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in frame_locals.items():
        if key.startswith("__"):
            continue
        text = repr(value)
        safe[key] = text[:MAX_LOCAL_REPR_CHARS]
    return safe


def _validate_code(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are disabled in the MVP Python runner")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in DISALLOWED_CALLS:
                raise ValueError(f"Call to '{node.func.id}' is not allowed")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "runner-python"}


@app.post("/run", response_model=RunResponse)
def run_python(req: RunRequest) -> RunResponse:
    events: list[ExecutionEvent] = []
    stack: list[str] = []

    def tracer(frame, event, arg):
        filename = frame.f_code.co_filename
        if filename != "<user_code>":
            return tracer

        function_name = frame.f_code.co_name
        if event == "call":
            stack.append(function_name)
            events.append(
                ExecutionEvent(
                    event="call",
                    function=function_name,
                    line=frame.f_lineno,
                    call_stack=stack.copy(),
                    locals=_safe_locals(frame.f_locals),
                )
            )
        elif event == "line":
            events.append(
                ExecutionEvent(
                    event="line_exec",
                    line=frame.f_lineno,
                    function=function_name,
                    call_stack=stack.copy(),
                    locals=_safe_locals(frame.f_locals),
                )
            )
        elif event == "return":
            events.append(
                ExecutionEvent(
                    event="return",
                    function=function_name,
                    line=frame.f_lineno,
                    call_stack=stack.copy(),
                    locals=_safe_locals(frame.f_locals),
                )
            )
            if stack:
                stack.pop()
        return tracer

    stdout_buffer = io.StringIO()
    globals_dict: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    old_trace = sys.gettrace()
    try:
        _validate_code(req.code)
        compiled = compile(req.code, "<user_code>", "exec")
        sys.settrace(tracer)
        with redirect_stdout(stdout_buffer):
            exec(compiled, globals_dict, globals_dict)
        output = stdout_buffer.getvalue()
        if output:
            for line in output.splitlines():
                events.append(ExecutionEvent(event="stdout", stdout=line))
        events.append(ExecutionEvent(event="end", message="Execution finished"))
    except Exception as exc:  # pragma: no cover - defensive runtime path
        events.append(ExecutionEvent(event="error", message=f"{type(exc).__name__}: {exc}"))
    finally:
        sys.settrace(old_trace)
    return RunResponse(events=events)
