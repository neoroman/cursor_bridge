# cursor_bridge

**한국어** | [README.ko.md](README.ko.md)

Exposes the Cursor CLI as an **OpenAI-compatible HTTP API** (e.g. for [OpenClaw](https://docs.openclaw.ai) or any client that speaks OpenAI-style `/v1/chat/completions`).

- **License**: Apache-2.0

## Overview

```
[Client e.g. OpenClaw]  --HTTP (OpenAI format)-->  [cursor_bridge :18081]  --subprocess-->  [cursor agent]
```

- FastAPI app: `/v1/chat/completions`, `/v1/models`, `/health`
- Forwards requests to `cursor agent --print --output-format json ...` and returns parsed text in OpenAI response shape

## Requirements

- **Cursor CLI** on `PATH` with `cursor agent` working
- **Python 3** and deps: `pip install -r requirements.txt` (FastAPI, uvicorn, pydantic v2)
- Optional: `CURSOR_API_KEY` in env (or in `.env` / `cursor_bridge.env`) for Cursor auth

## Run

```bash
# From repo root
pip install -r requirements.txt
./run.sh
# => uvicorn cursor_bridge:app --host 127.0.0.1 --port 18081
```

Or with venv (recommended if you have dependency conflicts):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run.sh   # run.sh uses .venv if present
```

Secrets: put them in `.env` or `cursor_bridge.env` (not in repo); `run.sh` loads them.

## Check

```bash
curl -s http://127.0.0.1:18081/health
curl -s http://127.0.0.1:18081/v1/models
```

## OpenClaw

In `openclaw.json`, add a custom provider:

- `baseUrl`: `http://127.0.0.1:18081/v1`
- `apiKey`: e.g. `dummy` (bridge does not validate it by default)

## Env / options

- `CURSOR_BRIDGE_DEBUG=1` — verbose logs
- `CURSOR_BRIDGE_TIMEOUT` — seconds to wait for `cursor agent` (default 180)
- `CURSOR_BRIDGE_WORKSPACE` — path passed as `--workspace` to cursor agent
- See `run.sh` and code for more.

## Docs

- **OpenClaw** deployment, troubleshooting, 422/timeout: see `docs/cursor_bridge/` in your OpenClaw workspace (e.g. [custom-cursor-setup](https://docs.openclaw.ai) pattern) or project docs.

## Troubleshooting

- **ImportError (pydantic/FastAPI)**: `pip install -r requirements.txt`
- **"Cursor agent is not logged in"**: set `CURSOR_API_KEY` in `.env` or `cursor_bridge.env` and run via `./run.sh`
- See `TROUBLESHOOTING-CURSOR-API-KEY.md` (or [한국어](TROUBLESHOOTING-CURSOR-API-KEY.ko.md)) for API key issues.
