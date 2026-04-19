import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from main import ExecutionEvent, normalize_events


def test_normalize_valid_event():
    events = normalize_events({"events": [{"event": "line_exec", "line": 1, "locals": {"x": "1"}}]})
    assert len(events) == 1
    assert isinstance(events[0], ExecutionEvent)
    assert events[0].event == "line_exec"


def test_normalize_invalid_event_becomes_error():
    events = normalize_events({"events": [{"unknown": True}]})
    assert events[0].event == "error"
