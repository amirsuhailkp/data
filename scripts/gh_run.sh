#!/usr/bin/env bash
#
# Runs one databroker pass inside a GitHub Actions runner and commits any
# database changes back to the repo, since each workflow run starts from a
# fresh checkout rather than a persistent machine.
#
# Deliberately NOT `set -e` for the whole script: the final git commit step
# exits non-zero when there is nothing to commit, which is an entirely
# normal outcome here (a sweep with no new events), not a failure. Instead
# each step that genuinely must succeed is checked explicitly.
set -uo pipefail

cd "$(dirname "$0")/.."

export DATABROKER_DB_PATH="${DATABROKER_DB_PATH:-$(pwd)/data/databroker.db}"
mkdir -p "$(dirname "$DATABROKER_DB_PATH")"

echo "[gh_run] DB path: $DATABROKER_DB_PATH"
echo "[gh_run] Running sweep..."
python -m databroker.cli sweep --delay "${SWEEP_DELAY:-1}"
SWEEP_STATUS=$?

if [ "$SWEEP_STATUS" -ne 0 ]; then
    echo "[gh_run] sweep exited with status $SWEEP_STATUS" >&2
    exit "$SWEEP_STATUS"
fi

# Market-wide discovery only on a slower cadence — driven by a counter file
# so it doesn't run every single invocation. DISCOVER_EVERY_N_RUNS=1 (or
# leaving it unset with DISCOVER_BACKEND configured) runs it every time.
if [ -n "${DISCOVER_BACKEND:-}" ]; then
    COUNTER_FILE="$(dirname "$DATABROKER_DB_PATH")/.discover_counter"
    N="${DISCOVER_EVERY_N_RUNS:-1}"
    COUNT=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)
    COUNT=$((COUNT + 1))
    echo "$COUNT" > "$COUNTER_FILE"
    if [ "$((COUNT % N))" -eq 0 ]; then
        echo "[gh_run] Running discover (run #$COUNT, every $N)..."
        python -m databroker.cli discover \
            --min-score "${DISCOVER_MIN_SCORE:-3}" \
            --limit "${DISCOVER_LIMIT:-5}"
    else
        echo "[gh_run] Skipping discover this run ($COUNT/$N)."
    fi
fi

# Commit any DB (and counter file) changes back to the repo. This is the
# step that must NOT be run under `set -e`-with-`&&`-chaining: `git commit`
# returns exit code 1 when the working tree has nothing staged, which is
# the common case (no new events this run) — that must not fail the job.
echo "[gh_run] Checking for changes to commit..."
git add -A data/

if git diff --cached --quiet; then
    echo "[gh_run] No DB changes this run — nothing to commit."
else
    git config user.name "databroker-bot"
    git config user.email "databroker-bot@users.noreply.github.com"
    if ! git commit -m "chore: update databroker.db [skip ci]"; then
        echo "[gh_run] git commit failed unexpectedly." >&2
        exit 1
    fi
    if ! git push; then
        echo "[gh_run] git push failed — DB changes are committed locally but not on the remote." >&2
        exit 1
    fi
    echo "[gh_run] Committed and pushed DB changes."
fi

echo "[gh_run] Done."
