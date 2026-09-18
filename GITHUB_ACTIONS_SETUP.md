# Running databroker on GitHub Actions (zero-cost, no card)

This runs `sweep` on a schedule instead of as a persistent process. Each run
starts from a fresh checkout, does one sweep (+ discovery on its own slower
cadence), commits `data/databroker.db` back to the repo if anything changed,
and pushes. State (watchlist, seen events, "already notified" flags) carries
forward via that committed DB file.

**Real limitation to accept going in:** this is not a true 5-minute
day-trading loop. GitHub's free-tier scheduler is best-effort — 15–30
minutes is the reliable floor, and runs can slip a few minutes past the
cron time under load. If you need tighter timing than that, this path
genuinely can't give it to you for free; see the README's hosting comparison.

## 1. Push this repo to your own GitHub account
If you haven't already:
```bash
git init
git add -A
git commit -m "databroker with GitHub Actions scheduling"
git branch -M main
git remote add origin https://github.com/<you>/<your-repo>.git
git push -u origin main
```

## 2. Add secrets
**Settings → Secrets and variables → Actions → Secrets → New repository secret.**
Only add the ones for sources you actually use — an unset secret just
disables that source cleanly (see `doctor`'s output).

| Secret | Used for |
|---|---|
| `GROQ_API_KEY` | LLM calls if using `LLM_BACKEND=groq` (simplest, one key) |
| `FREELLMAPI_ENCRYPTION_KEY` | Only if using `LLM_BACKEND=freellmapi` — see step 2b |
| `FREELLMAPI_ADMIN_PASSWORD` | Only if using `LLM_BACKEND=freellmapi` — see step 2b |
| `FREELLMAPI_PROVIDER_KEYS_JSON` | Only if using `LLM_BACKEND=freellmapi` — see step 2b |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit social source |
| `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` | Alpaca news/screener/technicals |
| `FINNHUB_API_KEY` | Finnhub news/fundamentals |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage confirmation source |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Alerts |

**⚠️ Rotate before adding, if these are the same keys pasted anywhere
outside your own machine** (chat logs, etc.) — treat any key that's left
your local `.env` as potentially exposed and get a fresh one from that
provider's dashboard first.

### 2b. If you want freellmapi instead of plain Groq

You mentioned you already run freellmapi locally in Docker with multiple
Groq and Cerebras keys pooled behind it. That local container isn't
reachable from a GitHub Actions runner (it's on your home network), so
`scripts/gh_run.sh` instead starts a **fresh freellmapi container inside
the workflow itself** every run, and headlessly loads your provider keys
into it via `scripts/gh_bootstrap_freellmapi.py` — no dashboard clicking
needed. This was verified directly against freellmapi's own source
(`POST /api/keys`, confirmed to accept `{platform, key, label}` and to
support adding multiple keys for the same platform).

Set `LLM_BACKEND=freellmapi` as a repo **variable** (step 3), then add
these three **secrets**:

- **`FREELLMAPI_ENCRYPTION_KEY`** — any 64-char hex string. Generate one
  with `openssl rand -hex 32`. This doesn't need to match your local
  container's key; it's only used to encrypt keys inside this ephemeral
  container for the few seconds it runs.
- **`FREELLMAPI_ADMIN_PASSWORD`** — any password, 8+ characters. Used to
  create the container's admin account each run; nothing you need to
  remember or reuse elsewhere.
- **`FREELLMAPI_PROVIDER_KEYS_JSON`** — a JSON array with every Groq and
  Cerebras key you want pooled, e.g.:
  ```json
  [
    {"platform": "groq", "key": "gsk_your_first_key", "label": "groq-1"},
    {"platform": "groq", "key": "gsk_your_second_key", "label": "groq-2"},
    {"platform": "cerebras", "key": "csk_your_key", "label": "cerebras-1"}
  ]
  ```
  Paste this whole JSON blob as the secret's value (GitHub secrets support
  multi-line values). Pull the actual key values from your local
  freellmapi dashboard's Keys page, or straight from each provider's own
  console if you still have them.

Each run: a brand-new freellmapi container starts, gets these keys loaded
into it, serves the sweep, and is torn down when the job ends — so nothing
persists between runs and there's no separate always-on freellmapi to
maintain in the cloud. Optionally set `FREELLMAPI_MODEL` as a variable
(e.g. `auto:groq-cerebras`) to match your local routing preference.

## 3. Add repository variables (non-secret config)
**Same page → Variables tab → New repository variable.**

| Variable | Example value | Notes |
|---|---|---|
| `LLM_BACKEND` | `groq` or `freellmapi` | `groq` needs only `GROQ_API_KEY`; `freellmapi` needs the three secrets in step 2b |
| `GROQ_MODEL` | (leave unset) | only if using `LLM_BACKEND=groq` and want to override the default |
| `FREELLMAPI_MODEL` | `auto:groq-cerebras` | only if using `LLM_BACKEND=freellmapi` |
| `SOCIAL_BACKEND` | `both` | or `reddit`, `stocktwits`, `off` |
| `REDDIT_SUBREDDITS` | `daytrading,stocks,wallstreetbets` | |
| `FINANCIAL_BACKEND` | `alpaca,sec,finnhub,stocktwits,alphavantage` | comma-separated, any subset |
| `SEC_EDGAR_CONTACT` | `you@example.com` | SEC requires a contact in the User-Agent |
| `ALPHA_VANTAGE_MODE` | `confirm` | keeps the 25/day cap a non-issue |
| `TECHNICALS_BACKEND` | `alpaca` | |
| `DISCOVER_BACKEND` | `alpaca,stocktwits` | leave unset to disable discovery entirely |
| `DISCOVER_EVERY_N_RUNS` | `8` | with the 30-min cron, 8 ≈ every 4 hours |

## 4. Enable Actions write permission
**Settings → Actions → General → Workflow permissions** → select **"Read
and write permissions"**. Without this, the workflow's push step in
`scripts/gh_run.sh` will fail with a 403, since the default token is
read-only.

## 5. Seed your watchlist
The DB starts empty. Either:
- Run `add`/`watch` locally once against `data/databroker.db`, then commit
  and push that file, **or**
- Trigger the workflow manually (Actions tab → databroker sweep → Run
  workflow) after adding a temporary `workflow_dispatch` step that runs
  `watch`/`add` — simplest is just doing it locally and pushing the file.

## 6. Test it
- **Actions tab → databroker sweep → Run workflow** — triggers it
  immediately instead of waiting for the next cron tick.
- Check the run's log for `[gh_run] ...` lines: DB path, sweep output,
  whether it committed.
- Confirm `data/databroker.db` shows a new commit from `databroker-bot`
  after a run that found something.
- Send yourself a test alert: temporarily set `NOTIFY_BACKEND=mock` as a
  variable, run once, confirm no crash, then switch back to `telegram`.

## Known tradeoffs of this approach (from the wrapper script, `scripts/gh_run.sh`)
- **Not a persistent process** — off-hours/holiday gating still applies
  inside `sweep`, but there's no `watch-loop`; the cron schedule itself is
  the only "loop."
- **Concurrency-guarded** — overlapping runs are prevented (`concurrency:`
  in the workflow) so two runs can't race to commit the same DB file.
- **Push failures fail the job on purpose** — if `git push` fails (e.g. a
  race, or Actions write permission not enabled per step 4), the job exits
  non-zero rather than silently reporting success, since a commit that
  never reached the remote would cause duplicate alerts on the next run.
- **`LLM_BACKEND=freellmapi` adds ~10-20 seconds per run** for the
  container to start and get bootstrapped, and a new failure surface: if
  a run fails at the bootstrap step, check the job log for
  `[gh_bootstrap]` lines first — a `401`/`409` there usually means the
  admin password secret changed between runs (harmless, it just logs in
  instead of signing up), while a `400` on a specific key means that
  provider/key pair was rejected (check the platform name is exactly
  `groq` or `cerebras`, lowercase, in `FREELLMAPI_PROVIDER_KEYS_JSON`).
