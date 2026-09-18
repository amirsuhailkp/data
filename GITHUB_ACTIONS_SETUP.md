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
| `GROQ_API_KEYS` | Only if using `LLM_BACKEND=pool` — see step 2b |
| `CEREBRAS_API_KEYS` | Only if using `LLM_BACKEND=pool` — see step 2b |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit social source |
| `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` | Alpaca news/screener/technicals |
| `FINNHUB_API_KEY` | Finnhub news/fundamentals |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage confirmation source |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Alerts |

**⚠️ Rotate before adding, if these are the same keys pasted anywhere
outside your own machine** (chat logs, etc.) — treat any key that's left
your local `.env` as potentially exposed and get a fresh one from that
provider's dashboard first.

### 2b. If you want to pool multiple Groq + Cerebras keys instead of plain Groq

This pools your keys **directly in databroker's own code** (`PooledProvider`
in `databroker/llm.py`) — no separate gateway, no container, nothing extra
to run. Each sweep tries a fixed chain of (provider, model) pairs, rotating
through every key you give it for a provider before moving to the next
entry in the chain:

1. Groq — `openai/gpt-oss-120b`
2. Cerebras — `gpt-oss-120b`
3. Cerebras — `zai-glm-4.7`
4. Cerebras — `qwen-3.8-27b`

Set `LLM_BACKEND=pool` as a repo **variable** (step 3), then add these two
**secrets** — each a comma-separated list of keys (one key is fine too):

- **`GROQ_API_KEYS`** — one or more Groq keys, e.g.:
  ```
  gsk_your_first_key,gsk_your_second_key
  ```
- **`CEREBRAS_API_KEYS`** — one or more Cerebras keys, e.g.:
  ```
  csk_your_first_key,csk_your_second_key
  ```

Pull the actual key values from console.groq.com and cloud.cerebras.ai.
No JSON, no admin password, no encryption key — just the raw comma-separated
key lists. If you want a different model chain than the default above,
set the optional variable `LLM_POOL_CHAIN` (step 3) instead of editing code.

## 3. Add repository variables (non-secret config)
**Same page → Variables tab → New repository variable.**

| Variable | Example value | Notes |
|---|---|---|
| `LLM_BACKEND` | `groq` or `pool` | `groq` needs only `GROQ_API_KEY`; `pool` needs the two secrets in step 2b |
| `GROQ_MODEL` | (leave unset) | only if using `LLM_BACKEND=groq` and want to override the default |
| `LLM_POOL_CHAIN` | (leave unset) | only if using `LLM_BACKEND=pool` and want a different chain than the built-in default — comma-separated `provider:model` pairs, e.g. `groq:openai/gpt-oss-120b,cerebras:zai-glm-4.7` |
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
- **`LLM_BACKEND=pool` fails a model over per-key, not just per-model** —
  if a run's LLM calls fail entirely, check the job log for a
  `PooledProvider: every (provider, model, key) combination in the chain
  failed` line; it lists exactly which provider/model/key attempts were
  made and why each one failed (401 = bad/revoked key, 429 = that key is
  rate-limited right now), which is normally enough to tell you whether to
  rotate a key or just wait out a rate limit.
