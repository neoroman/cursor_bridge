#!/bin/bash
# Run cursor_bridge. Do not put secrets (API keys etc.) in this file; use the env files below or export.
# For nohup/background runs, PATH is extended so cursor CLI can be found (e.g. mini2018).
# Dependencies: pip install -r requirements.txt (FastAPI 0.100+ / Pydantic v2. Uninstalled may cause Undefined from pydantic.fields.)

set -e
cd "$(dirname "$0")"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"

# Load env files in this directory in order (later files override earlier).
# CURSOR_API_KEY etc. in .env/cursor_bridge.env is loaded after .env and takes effect.
for f in .env cursor_bridge.env .env/cursor_bridge.env; do
  if [ -f "$f" ]; then
    set -a
    # shellcheck source=/dev/null
    . "./$f"
    set +a
  fi
done

# Optional venv (recommended if dependency conflict with Airflow/Flask-AppBuilder etc.):
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# .venv created on this machine only (other path/machine .venv may cause "bad interpreter").
# If that happens: rm -rf .venv then recreate with the command above.

if [ -f ".venv/bin/python3" ] && .venv/bin/python3 -c "import uvicorn" 2>/dev/null; then
  exec .venv/bin/python3 -m uvicorn cursor_bridge:app --host 127.0.0.1 --port 18081
fi
# Debug logging and _debug in response: CURSOR_BRIDGE_DEBUG=1
# Streaming placeholder: default "Processing..." + newline + loading. For GIF: CURSOR_BRIDGE_LOADING_GIF_URL=https://discord.com/assets/e541f62450f233be.svg
# Override full text: CURSOR_BRIDGE_STREAM_PLACEHOLDER="Processing...\n\n🔄\n"
CURSOR_BRIDGE_STREAM_PLACEHOLDER="Processing...🔄\n\n"
# cursor agent timeout (seconds, default 180). If exceeded, "Cursor timeout after 180s" → set CURSOR_BRIDGE_TIMEOUT=360 etc.
# Fixed workspace: CURSOR_BRIDGE_WORKSPACE=/path/to/workspace
# cursor agent --model: default auto. To fix: CURSOR_BRIDGE_AGENT_MODEL=gpt-5.2-codex-high-fast etc.
# By default -f is passed to cursor agent to skip workspace trust. To omit -f: CURSOR_BRIDGE_AGENT_SKIP_TRUST=1
# To add --trust --force (some CLIs only): CURSOR_BRIDGE_AGENT_TRUST=1
exec uvicorn cursor_bridge:app --host 127.0.0.1 --port 18081
