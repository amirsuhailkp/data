---
title: databroker
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# databroker (Hugging Face Space)

Runs two things in one container, since Spaces publish a single container/port:

1. **freellmapi** — the OpenAI-compatible LLM gateway, bootstrapped headlessly
   on first boot (see `hf/bootstrap_freellmapi.py`) — no dashboard clicking
   needed for the unified key.
2. **databroker's `watch-loop`** — the actual monitoring/alerting agent, running
   in the background.

The page you're looking at when you open this Space is just a status page
(`hf/webapp.py`) so Hugging Face has something to show as "Running" and so an
external keep-alive pinger has something to hit — it is not the app itself.

## Required Space secrets

Set these under Space settings → **Variables and secrets**:

| Secret | Required | Purpose |
|---|---|---|
| `FREELLMAPI_ADMIN_PASSWORD` | Yes | Password for the auto-created freellmapi admin account (8+ chars). Bootstrap refuses to run without this. |
| `FREEAPI_CONFIG_JSON` | Yes, to get real LLM responses | Your actual provider keys, applied automatically on every boot. Example: `{"keys":[{"platform":"groq","key":"gsk_..."},{"platform":"cerebras","key":"csk-..."}],"routing":{"strategy":"balanced"}}` |
| `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` | For financial/technicals/discovery sources | Same free paper-trading credentials from your local setup |
| `FINNHUB_API_KEY` | Optional | Finnhub news source |
| `ALPHA_VANTAGE_API_KEY` | Optional | Confirmation-mode source |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | For alerts | Same as your local `.env` |
| `SEC_EDGAR_CONTACT` | Optional | Your email, for SEC's fair-use header |
| `WATCH_INTERVAL_MINUTES` | Optional | Default 5 |
| `DISCOVER_EVERY_HOURS` | Optional | Default 6 |

All the `FINANCIAL_BACKEND`/`TECHNICALS_BACKEND`/`DISCOVER_BACKEND`/`NOTIFY_BACKEND`
values from your local `.env` should also be set here as secrets — Space
secrets become container environment variables the same way `.env` does
locally. `LLM_BACKEND` should be `freellmapi` (the whole point of this setup);
`FREELLMAPI_BASE_URL` is fixed by `entrypoint.sh` to the in-container gateway
and doesn't need to be set.

## Keeping it awake

Free Spaces sleep after 48 hours with no incoming web requests — the
background loop doesn't count. Point a free external pinger (UptimeRobot,
cron-job.org — both free, no card) at this Space's public URL, hitting `/health`
every 20–30 minutes.

## Storage caveat

The container's filesystem (where both the databroker SQLite DB and
freellmapi's own DB live) persists across restarts/sleep-wake, but **not**
guaranteed across a rebuild triggered by pushing new code — a code update
could reset the watchlist history and require re-adding companies with `watch`.

## First boot checklist

1. Set the secrets above, then push/create the Space.
2. Watch the build logs, then the container logs, for
   `[bootstrap] Unified freellmapi key retrieved...`
3. Open the Space URL — the status page should load and show recent
   `watch-loop` log lines.
4. If you see the "no provider keys configured" note in the logs, double
   check `FREEAPI_CONFIG_JSON`.
