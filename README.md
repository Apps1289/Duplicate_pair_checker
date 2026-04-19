# Visualization Compiler MVP

This repository contains a deployment-ready MVP for a visualization compiler platform with modular services.

## Architecture

- `services/web`: Web IDE (editor + execution controls) + visualization panel
- `services/api`: orchestration API (`/execute`, `/health`) with normalized event schema
- `services/runner-python`: Python tracer runner
- `services/runner-js`: JavaScript instrumentation runner

```text
Web (8080) -> API (8000) -> runner-python (8001)
                     \--> runner-js (8002)
```

## Shared execution event schema

The API and runners use this schema:

- `line_exec`
- `var_update`
- `call`
- `return`
- `stdout`
- `end`
- `error`

Common fields: `line`, `function`, `call_stack[]`, `locals{}`, `stdout`, `message`.

## Local run

```bash
cp .env.example .env
docker compose up --build
```

- Web UI: `http://localhost:8080`
- API health: `http://localhost:8000/health`

## API contract

### `POST /execute`

Request:

```json
{
  "language": "python",
  "code": "print('hi')"
}
```

Response:

```json
{
  "language": "python",
  "events": [
    { "event": "line_exec", "line": 1, "locals": {}, "call_stack": [] },
    { "event": "stdout", "stdout": "hi" },
    { "event": "end", "message": "Execution finished" }
  ]
}
```

## Execution controls in UI

- `Run`: sends code to API and receives event list
- `Play/Pause/Step`: timeline playback through events
- Side-by-side view: code line highlight + call stack + variables + stdout

## Extend with new languages

1. Add a new runner service exposing `POST /run` + `/health`.
2. Emit events in the shared schema.
3. Register runner endpoint in `services/api/main.py`.
4. Add language option to `services/web/index.html`.

## Deployment notes

- Services are containerized with independent Dockerfiles.
- Configuration is environment-variable driven (`.env.example`).
- `docker-compose.yml` provides full local stack.
- Starter CI workflow runs API tests on push/PR.

## Targeted tests

```bash
cd services/api
pip install -r requirements.txt
pytest -q
```
