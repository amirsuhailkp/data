#!/bin/bash
# Combined entrypoint for the HF Space: starts freellmapi (background),
# bootstraps its account + unified key headlessly, starts databroker's
# watch-loop (background), then runs the status web app in the foreground
# — that last one is the process Hugging Face actually supervises, so if it
# dies the whole Space restarts (which also restarts the other two, since
# they're children of this same script).
set -euo pipefail

APP_DIR="/home/user/app"
DB_DIR="$APP_DIR/databroker"
DATA_DIR="$DB_DIR/data"
mkdir -p "$DATA_DIR"

echo "[entrypoint] Starting freellmapi on :3001..."
PORT=3001 FREEAPI_DB_PATH="$DATA_DIR/freeapi.db" \
    node "$APP_DIR/freellmapi/server/dist/index.js" > "$DATA_DIR/freellmapi.log" 2>&1 &
FREELLMAPI_PID=$!

echo "[entrypoint] Bootstrapping freellmapi account + fetching its unified key..."
if "$DB_DIR/venv/bin/python" "$APP_DIR/bootstrap_freellmapi.py"; then
    export FREELLMAPI_API_KEY="$(cat /tmp/freellmapi_key)"
else
    echo "[entrypoint] WARNING: freellmapi bootstrap failed — LLM_BACKEND=freellmapi calls" >&2
    echo "[entrypoint]          will fail until this is fixed. See the log above." >&2
fi
export FREELLMAPI_BASE_URL="http://127.0.0.1:3001/v1"

echo "[entrypoint] Starting databroker watch-loop..."
cd "$DB_DIR"
"$DB_DIR/venv/bin/python" -m databroker.cli watch-loop \
    --interval-minutes "${WATCH_INTERVAL_MINUTES:-5}" \
    --discover-every-hours "${DISCOVER_EVERY_HOURS:-6}" \
    > "$DATA_DIR/watchloop.log" 2>&1 &
WATCHLOOP_PID=$!

# If either background process dies, bring the container down with it —
# HF will restart it, rather than silently running with half the system gone.
( wait "$FREELLMAPI_PID"; echo "[entrypoint] freellmapi exited — stopping." >&2; kill 0 ) &
( wait "$WATCHLOOP_PID"; echo "[entrypoint] watch-loop exited — stopping." >&2; kill 0 ) &

echo "[entrypoint] Starting status web app on :7860..."
exec "$DB_DIR/venv/bin/python" "$APP_DIR/webapp.py"
