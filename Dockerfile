# syntax=docker/dockerfile:1.7
#
# Combines two things into one Hugging Face Space (Docker SDK), since Spaces
# publish exactly one container/one port:
#   1. freellmapi (github.com/tashfeenahmed/freellmapi) — the OpenAI-compatible
#      gateway databroker's LLM_BACKEND=freellmapi talks to, normally run as
#      its own local Docker container. Pulled here from its own published
#      image (stage 1) purely to copy out its already-built server, so this
#      Dockerfile doesn't have to reimplement their multi-stage Node build.
#   2. databroker itself, running watch-loop in the background.
# entrypoint.sh starts both, headlessly bootstraps freellmapi's account and
# unified key (see bootstrap_freellmapi.py), and runs a tiny status page in
# the foreground on :7860 — the one process HF Spaces actually supervises.
#
# Pin the freellmapi version explicitly rather than :latest, so a freellmapi
# release doesn't unexpectedly change what's running here. Bump deliberately.
FROM ghcr.io/tashfeenahmed/freellmapi:v0.11.0 AS freellmapi

FROM node:20-bookworm-slim

# Python, for databroker — freellmapi itself is already-compiled JS copied
# in below, so it needs no build tooling here (unlike freellmapi's own
# Dockerfile, which compiles better-sqlite3 from source; we're copying its
# ALREADY-BUILT node_modules, not rebuilding them).
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires containers to run as a non-root UID 1000.
RUN useradd -m -u 1000 user
ENV HOME=/home/user
WORKDIR /home/user/app

# --- freellmapi: copy the already-built server + deps from stage 1 ---
COPY --from=freellmapi --chown=user:user /app ./freellmapi

# --- databroker ---
COPY --chown=user:user databroker ./databroker/databroker
COPY --chown=user:user requirements.txt ./databroker/requirements.txt

# --- glue: bootstrap script + status webapp + entrypoint ---
COPY --chown=user:user hf/bootstrap_freellmapi.py ./bootstrap_freellmapi.py
COPY --chown=user:user hf/webapp.py ./webapp.py
COPY --chown=user:user hf/entrypoint.sh ./entrypoint.sh

USER user

RUN python3 -m venv ./databroker/venv \
    && ./databroker/venv/bin/pip install --no-cache-dir -r ./databroker/requirements.txt \
    && ./databroker/venv/bin/pip install --no-cache-dir flask

RUN mkdir -p ./databroker/data && chmod +x ./entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["./entrypoint.sh"]
