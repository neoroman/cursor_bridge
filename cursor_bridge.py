from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, model_validator, field_validator
from pydantic.functional_validators import BeforeValidator
from typing import Annotated, List, Optional, Dict, Any, Tuple, Union
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
import subprocess
import json
import time
import uuid
import os
import logging

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("CURSOR_BRIDGE_DEBUG", "").strip() in ("1", "true", "yes") else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="cursor-bridge", version="0.2.0")

# Allow CORS so OpenClaw/gateway can call GET /v1/models from other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _log_env_startup() -> None:
    """Check env passed to cursor agent subprocess. Key value is not logged."""
    has_key = bool((os.environ.get("CURSOR_API_KEY") or "").strip())
    logger.info(
        "cursor_bridge startup: CURSOR_API_KEY in process env=%s (agent subprocess will inherit this)",
        "set" if has_key else "NOT SET",
    )
    if not has_key:
        logger.warning(
            "CURSOR_API_KEY not set. If you see 'Cursor agent is not logged in', "
            "start cursor_bridge via ./run.sh (which loads .env/cursor_bridge.env) or export CURSOR_API_KEY before starting."
        )


@app.exception_handler(RequestValidationError)
def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    On validation failure always return 422 with JSON body.
    So OpenClaw etc. get full detail instead of just '422 status code (no body)'.
    """
    logger.warning(
        "Request validation failed (422) path=%s detail=%s",
        _request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Request validation failed (cursor_bridge)",
            "body": "present",
        },
    )


# Debug mode: if CURSOR_BRIDGE_DEBUG=1, response includes _debug (stdout/stderr)
DEBUG_RESPONSE = os.environ.get("CURSOR_BRIDGE_DEBUG", "").strip() in ("1", "true", "yes")

# Workspace trust: in headless, -f skips "Workspace Trust Required". If CURSOR_BRIDGE_AGENT_SKIP_TRUST=1, -f is not used.
AGENT_SKIP_TRUST = os.environ.get("CURSOR_BRIDGE_AGENT_SKIP_TRUST", "").strip().lower() not in ("1", "true", "yes")
# CURSOR_BRIDGE_AGENT_TRUST=1 adds --trust --force (some CLIs only; unsupported may show unknown option)
AGENT_USE_TRUST = os.environ.get("CURSOR_BRIDGE_AGENT_TRUST", "").strip().lower() in ("1", "true", "yes")

# cursor agent --model: default auto (Cursor picks account default). To fix: CURSOR_BRIDGE_AGENT_MODEL=gpt-5.2-codex-high-fast etc.
def _agent_model() -> str:
    return (os.environ.get("CURSOR_BRIDGE_AGENT_MODEL", "auto") or "auto").strip()


def _agent_workspace() -> str:
    """
    Path to pass to cursor agent --workspace.
    Use CURSOR_BRIDGE_WORKSPACE if set.
    Else use OpenClaw workspace (~/.openclaw/workspace) if it exists,
    else cursor_bridge directory (project root for .cursor/AGENTS.md).
    """
    raw = os.environ.get("CURSOR_BRIDGE_WORKSPACE", "").strip()
    if raw:
        return raw
    openclaw_ws = os.path.expanduser("~/.openclaw/workspace")
    if os.path.isdir(openclaw_ws):
        return openclaw_ws
    return os.path.dirname(os.path.abspath(__file__))

# Cursor IDE requires at least 16000 context tokens for custom models. Unset defaults to 4096 and can block.
# Override with CURSOR_BRIDGE_CONTEXT_LENGTH (default 32768).
def _context_length() -> int:
    raw = os.environ.get("CURSOR_BRIDGE_CONTEXT_LENGTH", "32768").strip()
    try:
        n = int(raw)
        return max(16000, n)
    except ValueError:
        return 32768


# Model IDs used by OpenClaw/Cursor. Expose all in list so context_length is applied.
# Override with CURSOR_BRIDGE_MODEL_IDS (comma-separated).
def _list_model_ids() -> List[str]:
    raw = os.environ.get("CURSOR_BRIDGE_MODEL_IDS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return ["gpt-5.2-codex-high-fast", "cursor", "composer-1.5"]


# Phase 1: fetch text file URL (multimodal-expansion-plan)
def _fetch_enabled() -> bool:
    return os.environ.get("CURSOR_BRIDGE_FETCH_ENABLED", "").strip() in ("1", "true", "yes")


def _fetch_allowed_hosts() -> Optional[List[str]]:
    raw = os.environ.get("CURSOR_BRIDGE_FETCH_ALLOWED_HOSTS", "").strip()
    if not raw:
        return None
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def _fetch_timeout_sec() -> int:
    try:
        return max(1, int(os.environ.get("CURSOR_BRIDGE_FETCH_TIMEOUT_SEC", "15").strip()))
    except ValueError:
        return 15


def _fetch_max_bytes() -> int:
    try:
        return max(1024, int(os.environ.get("CURSOR_BRIDGE_FETCH_MAX_BYTES", "524288").strip()))
    except ValueError:
        return 524288


def _fetch_text_from_url(url: str) -> str:
    """
    GET URL and return body as UTF-8 text.
    - If CURSOR_BRIDGE_FETCH_ALLOWED_HOSTS is empty, no fetch (security).
    - Only https. Timeout and max bytes applied.
    """
    if not url or not url.strip():
        return ""
    url = url.strip()
    allowed = _fetch_allowed_hosts()
    if not allowed:
        logger.debug("fetch: skipped (CURSOR_BRIDGE_FETCH_ALLOWED_HOSTS not set)")
        return ""
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme.lower() != "https":
        logger.warning("fetch: only https allowed, got %s", parsed.scheme)
        return ""
    host = (parsed.netloc or "").lower()
    if host not in allowed:
        logger.warning("fetch: host %s not in CURSOR_BRIDGE_FETCH_ALLOWED_HOSTS", host)
        return ""
    timeout = _fetch_timeout_sec()
    max_bytes = _fetch_max_bytes()
    try:
        req = Request(url, headers={"User-Agent": "cursor-bridge/0.2"})
        with urlopen(req, timeout=timeout) as resp:
            content_type = (resp.headers.get_content_type() or "").lower()
            if "text/" not in content_type and "json" not in content_type and "xml" not in content_type:
                logger.debug("fetch: non-text content-type %s, treating as text", content_type)
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                logger.warning("fetch: response truncated at %d bytes", max_bytes)
                data = data[:max_bytes]
            return data.decode("utf-8", errors="replace").strip()
    except (URLError, HTTPError, OSError) as e:
        logger.warning("fetch: failed to get %s: %s", url[:80], e)
        return ""
    except Exception as e:
        logger.warning("fetch: unexpected error for %s: %s", url[:80], e, exc_info=True)
        return ""


def _extract_file_url_from_part(part: Dict[str, Any]) -> Optional[str]:
    """Extract file/file_url type URL from a content part."""
    if not isinstance(part, dict):
        return None
    t = (part.get("type") or "").strip().lower()
    if t == "file":
        sub = part.get("file")
        if isinstance(sub, dict) and sub.get("url"):
            return (sub.get("url") or "").strip()
        if part.get("url"):
            return (part.get("url") or "").strip()
    if t == "file_url" and part.get("url"):
        return (part.get("url") or "").strip()
    return None


# ----------------------------
# OpenAI-compatible schemas (allow extra fields and content array from OpenClaw/OpenAI)
# ----------------------------

def _content_to_str(content: Optional[Union[str, List[Any], Dict[str, Any]]]) -> str:
    """
    OpenAI style: content is string or [{ type: 'text', text: '...' }, ...] or single object (Phase 2 relaxed).
    Phase 1: if CURSOR_BRIDGE_FETCH_ENABLED=1 and URL in ALLOWED_HOSTS, fetch and include as text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return (content or "").strip()
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        parts = []
        for part in content:
            try:
                if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                    parts.append(str(part["text"]).strip())
                elif isinstance(part, str):
                    parts.append(part.strip())
                elif isinstance(part, dict) and _fetch_enabled():
                    file_url = _extract_file_url_from_part(part)
                    if file_url:
                        fetched = _fetch_text_from_url(file_url)
                        if fetched:
                            parts.append(fetched)
            except Exception as e:
                logger.warning("content_to_str: skip part due to %s: %s", type(e).__name__, e)
        return "\n".join(p for p in parts if p)
    return ""


# Phase 2: allow single-object content (422 relief). CURSOR_BRIDGE_RELAXED_CONTENT=1 allows dict, default 1.
def _relaxed_content() -> bool:
    return os.environ.get("CURSOR_BRIDGE_RELAXED_CONTENT", "1").strip().lower() in ("1", "true", "yes")


def _content_none_to_empty(v: Any) -> Any:
    """Replace null content with ''. Runs before Pydantic Union validation to avoid 422."""
    return "" if v is None else v


# content: str | list | dict | null. null replaced with '' by BeforeValidator before Union validation.
_ContentUnion = Union[str, List[Dict[str, Any]], Dict[str, Any]]
ContentType = Annotated[_ContentUnion, BeforeValidator(_content_none_to_empty)]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    # Phase 2: allow single-object dict; null replaced with '' by BeforeValidator (OpenClaw/Discord 422 relief).
    content: ContentType = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_content(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        content = data.get("content")
        # Normalize single-object content to array (Phase 2). null replaced with '' by ContentType BeforeValidator.
        if isinstance(content, dict):
            if not _relaxed_content():
                raise ValueError(
                    "Single object content not allowed when CURSOR_BRIDGE_RELAXED_CONTENT=0. "
                    "Set CURSOR_BRIDGE_RELAXED_CONTENT=1 or send content as string/list."
                )
            data = {**data, "content": [content]}
        return data


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "cursor"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_messages_content_null(cls, data: Any) -> Any:
        """
        Replace null content in messages with ''.
        Avoids 422 when Discord file upload etc. sends content: null on some messages.
        Normalize at top level so simple text flow is not broken (Phase 1 style).
        """
        if not isinstance(data, dict):
            return data
        messages = data.get("messages")
        if not isinstance(messages, list):
            return data
        new_messages = []
        changed = False
        for item in messages:
            if isinstance(item, dict) and item.get("content") is None:
                new_messages.append({**item, "content": ""})
                changed = True
            else:
                new_messages.append(item)
        if changed:
            data = {**data, "messages": new_messages}
        return data

    @model_validator(mode="before")
    @classmethod
    def _normalize_model(cls, data: Any) -> Any:
        """If model missing or empty string, use 'cursor' (Discord etc. 422 relief)."""
        if not isinstance(data, dict):
            return data
        model = data.get("model")
        if model is None or (isinstance(model, str) and not model.strip()):
            data = {**data, "model": "cursor"}
        return data


# ----------------------------
# Helpers
# ----------------------------

class CursorBridgeError(RuntimeError):
    pass


def _build_prompt(messages: List[ChatMessage]) -> str:
    """
    Build prompt for cursor agent.
    Keep OpenAI style but Cursor-friendly:
      - system at top
      - user/assistant as conversation
    """
    system_parts = []
    convo_parts = []

    for m in messages:
        role = (m.role or "").strip().lower()
        content = _content_to_str(m.content)
        if not content:
            continue

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            convo_parts.append(f"User: {content}")
        elif role == "assistant":
            convo_parts.append(f"Assistant: {content}")
        else:
            # unknown role treated as user
            convo_parts.append(f"User: {content}")

    prompt = ""
    if system_parts:
        prompt += "System:\n" + "\n\n".join(system_parts) + "\n\n"

    if convo_parts:
        prompt += "\n".join(convo_parts) + "\n"

    # Cue for Cursor to continue with assistant reply
    prompt += "Assistant:"
    return prompt


def _try_json_loads(s: str) -> Optional[Any]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_text_from_cursor_output(stdout: str, stderr: str = "") -> str:
    """
    Parse Cursor CLI output defensively (format varies by version).

    Cases:
      A) --output-format json -> dict like {"text": "..."}
      B) --output-format json -> OpenAI-style {"choices":[{"message":{"content":"..."}}]}
      C) stdout is a string that is itself a JSON string (double-encoded)
         e.g. content: "{\"type\":\"result\", \"result\": \"...\"}"
      D) If not JSON, use stdout as text
    """
    out = (stdout or "").strip()
    err = (stderr or "").strip()

    if not out and err:
        # In some environments result comes on stderr
        out = err

    # 1) Is stdout itself JSON dict/array?
    obj = _try_json_loads(out)
    if obj is not None:
        # Common keys when dict
        if isinstance(obj, dict):
            # OpenAI style
            try:
                choices = obj.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                    if isinstance(msg, dict):
                        c = msg.get("content")
                        if isinstance(c, str) and c.strip():
                            return _extract_text_from_cursor_output(c, "")
            except Exception:
                pass

            # Cursor style (version-dependent)
            for key in ("result", "text", "output", "message", "content"):
                v = obj.get(key)
                if isinstance(v, str) and v.strip():
                    # value may be JSON string again
                    nested = _try_json_loads(v)
                    if isinstance(nested, dict) and isinstance(nested.get("result"), str):
                        return nested["result"].strip()
                    return v.strip()

            # Otherwise: make dict human-readable
            return json.dumps(obj, ensure_ascii=False)

        # If array, just stringify
        return json.dumps(obj, ensure_ascii=False)

    # 2) stdout might be JSON string (double-encoded): '{...}' shape
    if out.startswith("{") and out.endswith("}"):
        nested = _try_json_loads(out)
        if isinstance(nested, dict):
            if isinstance(nested.get("result"), str):
                return nested["result"].strip()

    # 3) Plain text
    return out


def run_cursor(
    prompt: str,
    model: str,
    timeout_sec: int = 180,
    workspace: Optional[str] = None,
    extra_headers: Optional[List[str]] = None,
) -> Tuple[str, str, str]:
    """
    Run Cursor agent.
    Returns: (text, raw_stdout, raw_stderr)
    """
    if not prompt.strip():
        raise CursorBridgeError("Empty prompt")

    cmd = [
        "cursor", "agent",
        "--print",
        "--output-format", "json",   # JSON is most stable
        "--mode", "ask",
    ]
    if AGENT_SKIP_TRUST:
        cmd += ["-f"]   # Skip Workspace Trust Required (use -f among --trust/--yolo/-f)
    if AGENT_USE_TRUST:
        cmd += ["--trust", "--force"]
    cmd += ["--model", model]
    if workspace:
        cmd += ["--workspace", workspace]

    if extra_headers:
        for h in extra_headers:
            cmd += ["-H", h]

    cmd.append(prompt)

    env = os.environ.copy()
    # Tail style: omit start, keep last part in log (prompt_preview)
    _preview_tail_chars = 120
    if len(prompt) <= _preview_tail_chars:
        prompt_preview = prompt
    else:
        omitted = len(prompt) - _preview_tail_chars
        prompt_preview = f"[… first {omitted} chars omitted …]\n\n{prompt[-_preview_tail_chars:]}"
    logger.info(
        "run_cursor: starting cmd=%s prompt_len=%d prompt_preview=%s timeout=%ds",
        cmd[: len(cmd) - 1],
        len(prompt),
        repr(prompt_preview),
        timeout_sec,
    )
    t0 = time.monotonic()

    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - t0
        logger.error(
            "run_cursor: TIMEOUT after %.1fs (limit=%ds). stdout_len=%d stderr_len=%d",
            elapsed,
            timeout_sec,
            len(e.stdout or ""),
            len(e.stderr or ""),
        )
        if e.stdout or e.stderr:
            logger.debug("run_cursor: timeout stdout snippet: %s", (e.stdout or "")[:500])
            logger.debug("run_cursor: timeout stderr snippet: %s", (e.stderr or "")[:500])
        raise CursorBridgeError(f"Cursor timeout after {timeout_sec}s") from e
    except FileNotFoundError as e:
        logger.error("run_cursor: cursor CLI not found in PATH")
        raise CursorBridgeError("cursor CLI not found in PATH") from e

    elapsed = time.monotonic() - t0
    stdout = (p.stdout or "").strip()
    stderr = (p.stderr or "").strip()
    combined = f"{stdout}\n{stderr}".lower()

    # Re-raise with clear message when login/trust required (for OpenClaw etc.)
    if "workspace trust required" in combined or "do you trust" in combined:
        raise CursorBridgeError(
            "Workspace Trust Required. Run 'cursor agent' interactively once in this directory, "
            "or set CURSOR_BRIDGE_AGENT_SKIP_TRUST=0 and use CURSOR_BRIDGE_AGENT_TRUST=1 if your CLI supports --trust."
        )
    if "login" in combined and ("required" in combined or "authenticate" in combined or "sign in" in combined):
        raise CursorBridgeError(
            "Cursor agent is not logged in. Run 'cursor agent' in a terminal and complete login, "
            "or set CURSOR_API_KEY in the environment."
        )
    logger.info(
        "run_cursor: finished in %.1fs returncode=%d stdout_len=%d stderr_len=%d",
        elapsed,
        p.returncode,
        len(stdout),
        len(stderr),
    )

    if p.returncode != 0:
        detail = stderr or stdout or f"cursor agent failed (code={p.returncode})"
        logger.error("run_cursor: non-zero exit. stderr=%s stdout=%s", stderr[:800] if stderr else "", stdout[:800] if stdout else "")
        # Cursor CLI internal Security process exit 45: trust or security check failed in headless/non-interactive
        if "Security process exited with code: 45" in (stderr or "") or "Security process exited with code: 45" in (stdout or ""):
            raise CursorBridgeError(
                "Cursor Security process exited with code 45 (often in headless/server). "
                "Ensure CURSOR_BRIDGE_AGENT_SKIP_TRUST is unset or 0 so that -f is passed to skip workspace trust; "
                "or run 'cursor agent' once interactively in the bridge workspace to grant trust. "
                "See docs/cursor_bridge/troubleshooting-discord-422-security45.md."
            )
        raise CursorBridgeError(detail)

    text = _extract_text_from_cursor_output(stdout, stderr)
    if not text.strip():
        logger.warning("run_cursor: empty extracted text. stdout=%s stderr=%s", stdout[:500], stderr[:500])
        raise CursorBridgeError("Cursor returned empty output")

    logger.debug("run_cursor: extracted text_len=%d", len(text))
    return text, stdout, stderr


def _map_openclaw_model_to_cursor(req_model: str) -> str:
    """
    Map model value from OpenClaw/client to Cursor CLI model id.
    Aliases exposed in list (cursor, composer-1.5 etc.) route to same backend.
    """
    m = (req_model or "").strip()
    if not m:
        return "auto"

    # Exposed aliases -> same Cursor model
    if m in ("cursor", "cursor-default", "composer-1.5", "gpt-5.2-codex-high-fast"):
        return "gpt-5.2-codex-high-fast"

    return m


# ----------------------------
# Routes (OpenAI compatible; OpenClaw etc. use for endpoint type detection)
# ----------------------------

@app.get("/")
def root():
    """Root: indicate OpenAI-compatible service for OpenClaw endpoint detection."""
    return {"openai_compatible": True, "service": "cursor-bridge", "version": "0.2.0"}


@app.get("/v1")
def v1_root():
    """GET /v1: OpenAI API base path. For detection."""
    return {"object": "api", "openai_compatible": True, "service": "cursor-bridge"}


@app.get("/health")
def health():
    return {"ok": True}

def _model_entry(model_id: str = "gpt-5.2-codex-high-fast") -> dict:
    """
    OpenAI + OpenRouter compatible model object.
    - id, object, owned_by, created: OpenAI standard
    - context_length: used by OpenRouter/Cursor etc., min 16000 required
    - max_context_tokens: alias for some clients
    """
    ctx = _context_length()
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "custom-cursor",
        "created": 1730000000,  # Fixed value (compat). OpenAI expects creation time
        "context_length": ctx,
        "max_context_tokens": ctx,
    }


@app.get("/v1/models")
def list_models():
    # Expose all IDs used by OpenClaw/Cursor so context_length applies when queried by ID.
    return {
        "object": "list",
        "data": [_model_entry(mid) for mid in _list_model_ids()]
    }

@app.get("/v1/models/{model_id}")
def get_model(model_id: str):
    # Include context_length on single-model fetch to satisfy Cursor
    return _model_entry(model_id.strip() or "gpt-5.2-codex-high-fast")


@app.get("/models")
def list_models_compat():
    # some clients call /models without /v1 prefix
    return list_models()

# OpenClaw configure "Verification": short probe request gets immediate response without cursor agent to avoid timeout (aborted)
_VERIFICATION_PHRASES = frozenset({"hi", "hello", "ping", "test", "ok", ""})

# Streaming: send placeholder in first chunk to ease Discord 3s limit. Override via env.
# "Processing..." + newline + loading (GIF URL or emoji). CURSOR_BRIDGE_STREAM_PLACEHOLDER overrides full text.
_def = os.environ.get("CURSOR_BRIDGE_STREAM_PLACEHOLDER", "").strip()
if _def:
    _STREAM_PLACEHOLDER = _def or "Processing... "
else:
    _loading = os.environ.get("CURSOR_BRIDGE_LOADING_GIF_URL", "").strip() or "🔄"
    _STREAM_PLACEHOLDER = "Processing..." + _loading + "\n\n"


def _is_verification_probe(messages: List[ChatMessage]) -> bool:
    """True only when single user message with short verification phrase content."""
    if not messages or len(messages) != 1:
        return False
    m = messages[0]
    if (m.role or "").strip().lower() != "user":
        return False
    content = _content_to_str(m.content)
    return content.lower() in _VERIFICATION_PHRASES or len(content) <= 3


def _sse_chat_stream(
    req: ChatCompletionsRequest,
    x_cursor_model_override: Optional[str] = None,
):
    """
    OpenAI-compatible SSE stream generator.
    Eases Discord 3s limit: send placeholder in first event, then body after agent completes.
    """
    cursor_model = _map_openclaw_model_to_cursor(req.model)
    if x_cursor_model_override and x_cursor_model_override.strip():
        cursor_model = x_cursor_model_override.strip()
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def sse_event(choices_payload: list) -> str:
        return json.dumps({
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": cursor_model,
            "choices": choices_payload,
        }, ensure_ascii=False) + "\n"

    # 1) Verification probe: stream short response immediately
    if _is_verification_probe(req.messages):
        yield f"data: {sse_event([{'index': 0, 'delta': {'role': 'assistant', 'content': 'Cursor bridge is ready.'}, 'finish_reason': 'stop'}])}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 2) Send placeholder immediately (so client receives something within 3s)
    yield f"data: {sse_event([{'index': 0, 'delta': {'role': 'assistant', 'content': _STREAM_PLACEHOLDER}, 'finish_reason': None}])}\n\n"

    prompt = _build_prompt(req.messages)
    agent_model = _agent_model()
    workspace = _agent_workspace()
    timeout_sec = 180
    try:
        t = os.environ.get("CURSOR_BRIDGE_TIMEOUT", "").strip()
        if t:
            timeout_sec = max(30, int(t))
    except ValueError:
        pass

    try:
        text, _raw_stdout, _raw_stderr = run_cursor(
            prompt=prompt,
            model=agent_model,
            timeout_sec=timeout_sec,
            workspace=workspace,
        )
    except CursorBridgeError as e:
        logger.error("chat stream: CursorBridgeError: %s", e)
        yield f"data: {sse_event([{'index': 0, 'delta': {'content': f'[Error: {e!s}]'}, 'finish_reason': 'stop'}])}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        logger.exception("chat stream: unexpected error: %s", e)
        yield f"data: {sse_event([{'index': 0, 'delta': {'content': f'[Error: {e!s}]'}, 'finish_reason': 'stop'}])}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 3) Send body at once (cursor agent has no streaming output, single chunk)
    yield f"data: {sse_event([{'index': 0, 'delta': {'content': text}, 'finish_reason': 'stop'}])}\n\n"
    yield "data: [DONE]\n\n"


def _do_chat_completion(
    req: ChatCompletionsRequest,
    x_cursor_model_override: Optional[str] = None,
) -> dict:
    """Handle one chat. Shared by /v1/chat/completions and /v1/messages. cursor agent uses --model auto (override via env)."""
    cursor_model = _map_openclaw_model_to_cursor(req.model)
    if x_cursor_model_override and x_cursor_model_override.strip():
        cursor_model = x_cursor_model_override.strip()
    now = int(time.time())

    # Verification short request: respond immediately without cursor agent (avoid configure "Verification failed: aborted")
    if _is_verification_probe(req.messages):
        logger.info("chat: verification probe, skipping cursor agent")
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": cursor_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Cursor bridge is ready."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    prompt = _build_prompt(req.messages)
    agent_model = _agent_model()
    workspace = _agent_workspace()
    timeout_sec = 180
    try:
        t = os.environ.get("CURSOR_BRIDGE_TIMEOUT", "").strip()
        if t:
            timeout_sec = max(30, int(t))
    except ValueError:
        pass
    first_content_str = _content_to_str(req.messages[0].content) if req.messages else ""
    first_content = (first_content_str[:80] + "…") if len(first_content_str) > 80 else first_content_str
    logger.info(
        "chat: INVOKING cursor agent model=%s req_model=%s messages=%d prompt_len=%d timeout=%ds first_content=%s",
        agent_model,
        req.model,
        len(req.messages),
        len(prompt),
        timeout_sec,
        repr(first_content),
    )
    t0 = time.monotonic()
    try:
        text, raw_stdout, raw_stderr = run_cursor(
            prompt=prompt,
            model=agent_model,
            timeout_sec=timeout_sec,
            workspace=workspace,
        )
    except CursorBridgeError as e:
        logger.error("chat: CursorBridgeError after %.1fs: %s", time.monotonic() - t0, e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("chat: unexpected error after %.1fs: %s", time.monotonic() - t0, e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    logger.info("chat: success model=%s response_len=%d elapsed=%.1fs", cursor_model, len(text), time.monotonic() - t0)
    now = int(time.time())
    out = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": cursor_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        },
    }
    if DEBUG_RESPONSE:
        out["_debug"] = {"stdout": raw_stdout[:2000], "stderr": raw_stderr[:2000]}
    return out


@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatCompletionsRequest,
    authorization: Optional[str] = Header(default=None),
    x_cursor_model: Optional[str] = Header(default=None),
):
    """OpenAI compatible. When stream=true, send placeholder then body via SSE (eases Discord 3s limit)."""
    logger.info("POST /v1/chat/completions model=%s messages=%d stream=%s", req.model, len(req.messages), req.stream)
    if req.stream:
        return StreamingResponse(
            _sse_chat_stream(req, x_cursor_model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return _do_chat_completion(req, x_cursor_model)


@app.post("/v1/messages")
def messages(
    req: ChatCompletionsRequest,
    authorization: Optional[str] = Header(default=None),
    x_cursor_model: Optional[str] = Header(default=None),
):
    """OpenClaw etc. call /v1/messages. SSE streaming when stream=true."""
    logger.info("POST /v1/messages model=%s messages=%d stream=%s", req.model, len(req.messages), req.stream)
    if req.stream:
        return StreamingResponse(
            _sse_chat_stream(req, x_cursor_model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return _do_chat_completion(req, x_cursor_model)
