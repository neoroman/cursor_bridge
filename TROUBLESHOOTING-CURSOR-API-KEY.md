# "Cursor agent is not logged in" / CURSOR_API_KEY errors

## Error message

```
Cursor agent is not logged in. Run 'cursor agent' in a terminal and complete login,
or set CURSOR_API_KEY in the environment.
```

**In the same vein**, if you see "The provided API key is **invalid**", the key was passed but the value is invalid.  
→ See **workspace/docs/cursor_bridge/cursor-api-key-invalid.md**.

This message is detected from the **cursor agent CLI** subprocess stderr/stdout, not from **cursor_bridge** itself.  
That is, when cursor_bridge runs `cursor agent`, if the CLI prints "login required" or "authentication required", it is turned into the above error.

## Why CURSOR_API_KEY can still fail when set

`CURSOR_API_KEY` must be in the **cursor_bridge process (uvicorn) environment**,  
and cursor_bridge passes that environment to the **cursor agent subprocess** as-is.

So the following must be true:

1. **Always start cursor_bridge with `./run.sh`**  
   - `run.sh` loads `.env`, `cursor_bridge.env`, and **`.env/cursor_bridge.env`** in that order.  
   - `CURSOR_API_KEY` set there goes into the shell environment and is passed to uvicorn (and cursor_bridge) via `exec uvicorn ...`.

2. **Starting without run.sh means the key is not loaded**  
   - e.g. running only `uvicorn cursor_bridge:app --host 127.0.0.1 --port 18081`  
   - e.g. systemd / pm2 / OpenClaw running python/uvicorn **without** run.sh  
   → `.env`/`.env/cursor_bridge.env` are not loaded, **CURSOR_API_KEY is empty**,  
   → and the cursor agent subprocess inherits an empty environment, so "not logged in".

3. **.env file location and format**  
   - Path: `cursor_bridge/.env/cursor_bridge.env` (file `cursor_bridge.env` inside directory `.env`)  
   - Format: one line `CURSOR_API_KEY=value` (no spaces around `=`, quote value if it contains spaces)  
   - `run.sh` uses `set -a` then source, so no separate `export` is needed for child processes.

## How to verify

1. **cursor_bridge startup log**  
   - Recent cursor_bridge logs on startup:  
     `cursor_bridge startup: CURSOR_API_KEY in process env=set` or `NOT SET`  
   - If `NOT SET`, the current process has no key; **restart via run.sh**.

2. **Confirm you actually started with run.sh**  
   - From terminal: `cd /path/to/cursor_bridge && ./run.sh`  
   - With systemd/pm2: ensure `ExecStart` uses `.../cursor_bridge/run.sh`.  
     If it runs uvicorn directly, switch to run.sh or add  
     `Environment=CURSOR_API_KEY=...` or `EnvironmentFile=.../cursor_bridge/.env/cursor_bridge.env`.

3. **Whether Cursor CLI uses CURSOR_API_KEY**  
   - Depending on Cursor docs/version, headless may support `CURSOR_API_KEY`.  
   - If not, run `cursor agent` once in a terminal to log in, then start cursor_bridge with run.sh on the same user/machine to share the session.

## Summary

| Cause | Action |
|-------|--------|
| cursor_bridge started without run.sh | **Always start with `./run.sh`** (or inject the same env when starting) |
| .env path/filename typo | Check that `cursor_bridge/.env/cursor_bridge.env` exists and is named correctly |
| .env format error | `CURSOR_API_KEY=value` (no spaces, export not required) |
| systemd/pm2 running uvicorn only | Change ExecStart to run.sh or pass CURSOR_API_KEY via Environment/EnvironmentFile |

If the error persists after restart, check in cursor_bridge logs whether `CURSOR_API_KEY in process env=` is `set` or `NOT SET`.
