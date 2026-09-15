# DataBroker AI — Personal Investment Research Agent (MVP)

A working Phase 1–3 implementation of the DataBroker spec: a personal, persistent
AI research agent that investigates companies, tracks your investment thesis,
remembers past research, and tells you what actually changed and why it
matters — without predicting prices or placing trades.

**No paid model is required**, and **the LLM is not on the hook for every
decision.** Two design choices work together to make this cheap enough to run
on free-tier APIs or a local model:

1. **Free/local model backends** (`llm.py`) — Ollama (fully local),
   freellmapi (self-hosted multi-provider router with built-in rate-limit
   fallover), Groq or Gemini (free-tier cloud), or a hybrid mix.
2. **The LLM only does what only it can do** (`heuristics.py`) — reading
   fetched web text and extracting claims, and resolving genuinely ambiguous
   judgment calls. Every mechanical decision is handled by plain Python with
   zero tokens spent: building search queries, recognizing an obvious
   duplicate or an obviously new item, scoring importance from keyword +
   source-reliability signals, deciding whether two claims are even about the
   same topic before checking for a contradiction, and rolling up an overall
   thesis-impact label from per-point results.

## How much this actually saves

A single research question used to cost up to 7 LLM calls per claim found
(plan → search-plan → extract → conflict-check → dedupe → score → thesis →
report). After this change, a typical run looks like:

| Step | Before | Now |
|---|---|---|
| Decompose a simple objective ("Anything new?") | 1 LLM call | **0** — recognized as already-atomic |
| Build search queries | 1 LLM call per sub-question | **0** — template-based |
| Extract claims from fetched text | 1 LLM call | 1 LLM call *(unavoidable — this is the actual research)* |
| Check for conflicts between claims | 1 LLM call, sent the whole claim set | **0 most of the time** — only overlapping pairs are ever sent, and only if any overlap is found at all |
| Classify vs. history (new/duplicate/update) | 1 LLM call per claim, full event history in the prompt | **0 for confident cases** (clear duplicate or clearly unrelated); LLM only sees the top 3 closest candidates when genuinely ambiguous |
| Score importance | 1 LLM call per claim | **0 for confident cases** (keyword + source-type match, or "no thesis on file" default); LLM only asked when a thesis exists AND the heuristic can't confidently classify |
| Assess thesis impact | 1 LLM call per claim | Only for claims that clear the importance bar (skipped for "low") |
| Build final report | 1 LLM call, even with zero evidence | **0 when no evidence was found** — deterministic message instead |

In the worked example in this repo's test suite, a research run that found one
relevant claim and had a thesis on file made **3 LLM calls total** (extract,
thesis, report) instead of the 6–7 the naive version would have made.

## Where the line is drawn, and why

The rule of thumb: **if a human analyst would need to actually read and
weigh the specific wording, it goes to the LLM; if it's a threshold, a
lookup, or a pattern match, it's Python.**

- Search queries don't need judgment — a template ("{company} {topic}",
  "{ticker} {topic} news") covers it.
- Whether a new claim is an exact duplicate or clearly unrelated to
  everything on file is usually obvious from text similarity alone; only the
  "is this an update to something I already knew" middle ground needs a
  model's read on it — and even then, only the 3 closest matches are shown,
  not the full history.
- Whether "the company filed for bankruptcy" is important doesn't need an
  LLM to tell you — a keyword list does. Whether a subtler development
  matters to a *specific* investment thesis genuinely does need semantic
  judgment, so that case is preserved.
- A conflict check across every claim gathered this session is wasteful when
  most claims are about unrelated sub-questions; a fast word-overlap filter
  narrows it to only the pairs that plausibly describe the same fact.

None of this is airtight — heuristics.py documents its own known blind spots
(see the docstring on `find_overlapping_claim_pairs`) — but the failure mode
is deliberately the safe direction: a missed heuristic shortcut just means an
occasional extra LLM call, not a wrong answer, and the LLM is always the
fallback for the genuinely unclear cases, never skipped entirely.

## Choosing your setup

| Backend | Cost | Quality | Setup |
|---|---|---|---|
| **Ollama** (local) | Free | Depends on your hardware — `qwen2.5:14b` is a meaningfully better JSON-follower than `7b` if you can run it | `ollama pull qwen2.5:7b` (or `14b`), then `ollama serve` |
| **freellmapi** (local router) | Free | Depends on which free-tier providers you register behind it — typically as good as your best configured provider, with automatic fallover away from whichever one is rate-limited at the moment | Run [freellmapi](https://github.com/tashfeenahmed/freellmapi) locally (`docker compose up`), grab a unified key from its dashboard |
| **Groq** (cloud) | Free tier (rate-limited) | Currently the best free-tier reasoning quality — `openai/gpt-oss-120b` | Free key at console.groq.com |
| **Gemini** (cloud) | Free tier (rate-limited), Flash-only since Pro was pulled from free tier in 2026 | Good, but no longer Gemini's best model | Free key at aistudio.google.com/apikey |
| **Hybrid** | Free | Best free option — cloud model for plan/report/thesis, local for the rest | Set up both Ollama and one cloud key |

**Why freellmapi is worth it if you're hitting rate limits:** a `research`
run fires several LLM calls back-to-back (plan → one extract per
subquestion → conflict-check → thesis → report — 8-9+ calls with no
pacing between them), which reliably trips a single free-tier provider's
per-minute cap. `GroqProvider`/`GeminiProvider` now retry with backoff on
a 429 (see below), but if you're still hitting the wall, running
freellmapi locally and pointing `LLM_BACKEND=freellmapi` at it spreads
that same burst across every free-tier provider you've registered in its
dashboard — Groq, Gemini, Cerebras, Mistral, OpenRouter, and more — so one
provider's cooldown doesn't stall the whole run. Model defaults to
`"auto"`, which follows whichever fallback chain is active in the
dashboard; `auto:fast`/`auto:smart`/`auto:<profile-name>` steer a single
request without touching the dashboard. With `LLM_BACKEND=freellmapi`,
if the container isn't reachable (or `FREELLMAPI_API_KEY` isn't set),
databroker automatically falls back to local Ollama if it's running,
then `MockProvider` as a last resort — so it never hard-crashes just
because the container was restarted.

**`LLM_BACKEND` is validated** — an unrecognized value (e.g. a typo like
`api`) raises a clear error listing the valid options, rather than
silently behaving like `auto`. `doctor` catches this and prints it
inline instead of a raw traceback.

**Free-tier rate limits (429s):** `GroqProvider` and `GeminiProvider` both
retry automatically on a 429, honoring the provider's `Retry-After` header
when present and falling back to exponential backoff (1s/2s/4s/8s/16s)
otherwise, printing each retry attempt so you can see it happening. If it's
still 429ing after 5 retries, that's a real "you've exceeded free-tier
capacity for this burst" — either lower `RESEARCH_MAX_SUBQUESTIONS` in
`.env`, switch to `LLM_BACKEND=hybrid`/`freellmapi` to spread load, or just
wait a minute and retry.

**Model IDs move fast on both free tiers — worth knowing before you hit a
400 error:**
- Groq deprecated `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` for
  free/dev-tier accounts in June–August 2026. The current default here is
  **`openai/gpt-oss-120b`** (Groq's own recommended replacement, and
  currently the strongest free-tier reasoning model available) —
  `openai/gpt-oss-20b` is the lighter/faster option for high-volume
  repetitive calls. Override with `GROQ_MODEL` if Groq's lineup has moved on
  again by the time you read this; check console.groq.com/docs/models.
- Gemini's free tier has been Flash-only since Pro models were pulled from
  it in ~April 2026. The default here is **`gemini-2.5-flash`**; check
  ai.google.dev/gemini-api/docs/models for whatever's newest.
- `python -m databroker.cli doctor` shows exactly what's configured, but
  won't catch a model ID that's since been deprecated server-side — that
  still shows up as a request error, so if `research`/`sweep` starts
  failing, check the model first.

Search defaults to DuckDuckGo (free, no signup) via the `ddgs` package. It
tries DuckDuckGo's dedicated news index first (real articles with real
dates — much better for "what changed recently" questions than general web
search, which tends to surface evergreen homepage/Wikipedia pages for a bare
company name), falling back to general web search only if that comes up
empty. Tavily is an optional upgrade (`SEARCH_BACKEND=tavily`, needs a free
key) if you want a second opinion on source quality.

Run `python -m databroker.cli doctor` any time to see which backend is active.

## What's implemented vs. the spec

| Spec section | Status |
|---|---|
| §3 Watchlist / thesis management | ✅ `add`, `watch`, `set-thesis` |
| §4 Company research | ✅ `research` |
| §7 Investment Thesis Monitor | ✅ gated to claims that clear the importance bar (see above) |
| §8 Change detection | ✅ heuristic-first, LLM only for the ambiguous middle band |
| §9 Importance Engine | ✅ keyword + source-reliability baseline, LLM only when ambiguous and a thesis exists |
| §10 Autonomous Investigation Loop | ✅ `agent.py: ResearchAgent.investigate()` |
| §11 Source Verification | ✅ every claim stores source, type, reliability tier, verification status |
| §12 Conflict Detection | ✅ overlap-filtered before any LLM call |
| §13 Personal Research Memory | ✅ SQLite-backed |
| §16 Model abstraction | ✅ Ollama / freellmapi / Groq / Gemini / Hybrid / Claude(optional) / Mock, all behind one interface |
| §23 Daily Intelligence Brief | ✅ `digest` — pure DB read, zero LLM calls |
| §26 Boundary: research, not trading | ✅ enforced in the system prompt |
| Scheduled/background monitoring (Phase 2) | ✅ `sweep` (cron-friendly one-shot) and `watch-loop` (long-running, stdlib-only, no new dependency) |
| §5 Social & Community Intelligence (Phase 4) | ✅ opt-in (`SOCIAL_BACKEND`) — Reddit (OAuth pool, multi-app throughput) + Hacker News, plus opt-in `FINANCIAL_BACKEND` for SEC EDGAR filings + Finnhub news + Alpaca real-time news + Alpha Vantage sentiment news + StockTwits as "first line" ticker-scoped sources — see below |
| §18 Full browser agent (dynamic navigation) | ✅ opt-in (`FETCHER_BACKEND=browser`) — JS rendering + one bounded, mostly-deterministic navigation hop past index/listing pages — see below |
| Knowledge graph (Phase 5) | ✅ lightweight property graph on SQLite — entities/relationships extracted alongside regular research, zero extra LLM calls — see below |

## Social & Community Intelligence (Phase 4)

Off by default — turn it on with `SOCIAL_BACKEND`:

```bash
export SOCIAL_BACKEND=reddit       # or: hackernews | both
export REDDIT_SUBREDDITS=stocks,investing,wallstreetbets   # optional, scopes Reddit search
```

Hacker News uses Algolia's official public Search API — free, no signup, no
key. Reddit has two modes:

**OAuth (recommended)** — Reddit's official API, documented and predictable:
```bash
export REDDIT_CLIENT_ID=...
export REDDIT_CLIENT_SECRET=...
```
Register a free app at reddit.com/prefs/apps (type "script") to get these —
takes about two minutes, no approval wait. A single app is capped at roughly
100 requests/minute on the free tier.

**Pooling multiple apps for higher throughput** — since the cap is per-app,
not per-account, registering a few free apps and pooling their credentials
multiplies the effective combined budget:
```bash
export REDDIT_CLIENT_ID_1=... ; export REDDIT_CLIENT_SECRET_1=...
export REDDIT_CLIENT_ID_2=... ; export REDDIT_CLIENT_SECRET_2=...
export REDDIT_CLIENT_ID_3=... ; export REDDIT_CLIENT_SECRET_3=...
```
`reddit_client.py`'s `RedditOAuthPool` round-robins requests across whichever
credential currently has budget left, each tracked with its own independent
rate-limit bucket (95/min per credential, a small safety margin under
Reddit's ~100 figure) — three registered apps gives roughly 285 req/min in
aggregate, not because Reddit raised the limit, but because three
independent per-app budgets are being drawn from. `doctor` reports the
pooled combined budget once configured. If a request comes in and every
credential is currently out of budget, it's treated as "no results this
round" rather than blocking or crashing — same fail-soft behavior as every
other provider in this codebase.

**No credentials configured** — falls back automatically to Reddit's
unauthenticated public search endpoint (zero setup, but a lower and less
predictable rate limit, and more prone to layout/behavior changes since it's
not an intentionally-documented API surface).

Both sources feed into the *same* extraction call already made per
sub-question — turning this on adds no extra LLM calls.

Two things are handled deliberately carefully here, because forum content is
a different trust category than a filing or a news article:

- **Source type is never left to the model's guess.** Every hit from Reddit/HN
  is tagged `source_type="community"` by the code, and that tag overrides
  whatever the LLM classified it as during extraction — smaller/free models
  are more likely to mistake a Reddit post for "news" than a larger one
  would, so this is corrected deterministically rather than trusted.
- **A serious-sounding keyword hit from social content is never
  auto-confirmed.** "Recall"/"bankruptcy"/"lawsuit"-type language from an
  unverified forum post is exactly the case where a keyword heuristic alone
  would be dangerous (rumors escalate fast on social media) — so it's
  explicitly *not* one of the confident heuristic shortcuts, and gets a real
  LLM look even if you haven't set a thesis for that company yet, since
  "possible bankruptcy" is worth a second look regardless of whether you've
  written a thesis down.

## Real-time financial/trading platforms ("first line" sources)

Off by default — turn it on with `FINANCIAL_BACKEND=<comma-separated list>`,
e.g. `FINANCIAL_BACKEND=sec,finnhub,stocktwits,alpaca,alphavantage` to combine
all five, or just `FINANCIAL_BACKEND=sec` for the free, no-key option on its own.

| Backend | What it is | Cost | Source type / reliability tier |
|---|---|---|---|
| `sec` | Official SEC EDGAR filings (8-K, 10-K, 10-Q, S-1, proxy statements, 13D/G, etc.) via `data.sec.gov` | Free, no API key | `filing`, tier **1** — the single highest-reliability source this tool has |
| `finnhub` | Finnhub's company-news wire (`company-news` endpoint) | Free tier (~60 calls/min), needs `FINNHUB_API_KEY` from finnhub.io | `news`, tier 5 — same tier as general web news |
| `alpaca` | Alpaca's Benzinga-sourced real-time news feed — pushed close to publication time, the fastest source here | Free, needs a paper-trading account (`ALPACA_API_KEY_ID`/`ALPACA_API_SECRET_KEY`) at alpaca.markets | `news`, tier 5 — same tier as Finnhub, just faster |
| `alphavantage` | Alpha Vantage's NASDAQ-licensed news feed, with an AI sentiment score per article (overall AND per-ticker) | Free, needs `ALPHA_VANTAGE_API_KEY` — but capped at **25 requests/day total** on the account | `news`, tier 5 — differentiated by the sentiment label, not speed |
| `stocktwits` | StockTwits' public symbol stream — real trader chatter/sentiment | Free, no key (historically) | `community`, tier 7 — sentiment/discussion, not verified fact |

**If you're using this for day trading**, `alpaca` is the one to turn on —
it's specifically built for low-latency news, not just another news source
API. Pair it with `watch-loop --interval-minutes` (see "Scheduled
monitoring" below) instead of a daily sweep, since once-a-day is too slow
for that use case. **`alphavantage` is the wrong fit for that same fast-
polling setup** — its 25-req/day cap is account-wide, so a handful of
interval-polling ticks will exhaust the whole day's quota in minutes. It's
a good match for the once-a-day `sweep`/`research` workflow instead, on a
watchlist small enough to fit inside 25 checks a day, where its sentiment
labels add something Finnhub/Alpaca's plain news doesn't have.

Unlike Reddit/HN (free-text searchable), these are all inherently
ticker-scoped — activity about one specific symbol, not something you query
with arbitrary text. That's reflected in how they're wired in (`social.py`'s
`FinancialProvider` interface and `agent.py`'s `investigate()`):

- Fetched **once per research session**, not once per sub-question —
  repeating identical content across every sub-question's extraction prompt
  would cost tokens without adding information.
- Merged into the **first (highest-priority) sub-question's** extraction
  context, and placed **first** in that prompt — "first line of information"
  is reflected in actual prompt ordering, not just in being included. With
  more than one backend enabled, `CombinedFinancialProvider` splits the
  per-session hit budget (`MAX_FINANCIAL_HITS`, default 5) evenly across
  them, since a filing, a news wire, and trader chatter are genuinely
  different information — not redundant copies competing for the same slot.
- Each source carries its own framing label in the extraction prompt —
  `[OFFICIAL SEC FILING]`, `[FINANCIAL NEWS WIRE]`, or
  `[TRADING PLATFORM CHATTER]` — and its `source_type` is trusted from the
  provider rather than re-guessed by the LLM, exactly like Reddit/HN above.
  A claim sourced from StockTwits or Reddit that mentions something serious
  (e.g. "acquisition") still gets the same no-auto-confirm-on-serious-
  keywords escalation to a real LLM check, regardless of whether a thesis
  is on file — confirmed by testing.

**Setup notes:**
- `sec` needs no key, but SEC's fair-access policy asks for a real contact
  string in the User-Agent — set `SEC_EDGAR_CONTACT` (e.g. an email) in
  `.env`. It'll still run without it, just less politely. Only "material"
  form types are surfaced (8-K, 10-K, 10-Q, S-1, proxy statements, 13D/G,
  6-K, 20-F, etc.) — routine Form 4 insider-trade filings are filtered out
  since they're frequent enough to drown out everything else. The filing's
  own metadata (form type, filing date, item codes, description) is what
  gets extracted from — the full filing text isn't fetched, to keep this
  cheap and fast.
- `finnhub` needs `FINNHUB_API_KEY`; if it's missing, that backend is
  silently skipped (not a crash) — `doctor` will flag it.
- `alpaca` needs both `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` from a
  free **paper trading** account (email + MFA only — no ID verification or
  funding needed since you're only reading market data, not trading). If
  either is missing, the backend is silently skipped — `doctor` will flag
  it. Framed in the extraction prompt as `[REAL-TIME NEWS WIRE]`.
- `alphavantage` needs `ALPHA_VANTAGE_API_KEY` (free, alphavantage.co).
  **The free tier is capped at 25 requests/day total on the account** —
  not per ticker, not per backend, the whole key. Each `get_ticker_activity`
  call (i.e. each company checked in a `research`/`sweep` run) costs exactly
  one request regardless of how many articles come back. Prefers the
  article's ticker-specific sentiment label over its overall label when
  available, since a broad-market piece can be "Neutral" overall while
  genuinely bullish or bearish for the one ticker you're tracking. Alpha
  Vantage signals quota exhaustion with a 200 OK carrying an `Information`/
  `Note` field instead of the normal article list — handled explicitly as a
  soft-fail (`[]`), since it isn't an HTTP error status.
- `stocktwits`'s public symbol-stream endpoint has historically required no
  API key for basic reads, but financial data APIs tend to tighten terms
  over time.

All five fail soft (return no results, never raise) on a bad key, a
rate-limit hit, or a network error — a broken financial-data backend should
never take down a whole research run.

## Browser agent (§18, dynamic navigation)

Off by default (`FETCHER_BACKEND=static`) — turn it on with:

```bash
pip install playwright
playwright install chromium
```
```
FETCHER_BACKEND=browser
```

What it adds over the default static fetch (`requests.get()` +
trafilatura/BeautifulSoup):

- **Renders JavaScript.** Many investor-relations pages and SEC EDGAR filing
  pages build their actual content client-side — a plain HTTP GET sees a
  near-empty shell. A headless Chromium instance (via Playwright) renders
  the page properly before extracting text.
- **Follows one link past an index/listing page.** If a search result turns
  out to be a filings list or news index rather than the article itself,
  the static fetcher just extracts whatever thin text is on that listing.
  The browser agent recognizes this (short visible text + many links) and
  takes one bounded hop to the actual content.

Same token-conservation approach as everywhere else in this codebase:
dismissing a cookie-consent banner and clicking "read more" are pattern-
matched deterministically, no LLM involved. The LLM is only ever asked one
thing, at most once per fetch — "given these links, which one likely
answers the question?" — and only when the heuristic actually flags the
page as a listing in the first place. I tested this end-to-end against
local mock pages (a fake SEC-style filings index with 5 links, one of them
the actual 10-Q): it correctly identified the index page, asked for and
received the right link, navigated there, and extracted the real filing
text — no wasted LLM calls on the common case of a direct article link.

If Playwright isn't installed, `BrowserFetcher` transparently falls back to
the static fetch — nothing breaks if you leave `FETCHER_BACKEND=browser` set
without having installed it yet, it just won't get the JS-rendering benefit
until you do. `python -m databroker.cli doctor` reports whether Playwright
is actually available when this backend is selected.

**Known limitation:** this is one bounded hop past a listing page, not a
general browsing agent — it won't click through a multi-page pagination
sequence or fill out a form to reach gated content. If you need that, the
navigation loop in `browser.py`'s `BrowserFetcher.fetch()` is the place to
raise `max_hops` and extend the stopping condition.

## Knowledge graph (Phase 5)

Always on, no setup needed — this one isn't opt-in like Phase 4/§18, since it
adds no extra LLM calls and no extra dependency.

The same extraction call that already pulls claims out of fetched content
(`gather_evidence()`) is also asked to note any explicit relationship a claim
states between two named things — "NVIDIA acquired Hugging Face", "AWS
partnered with NVIDIA", "Jensen Huang is CEO of NVIDIA" — as a
`{subject, predicate, object}` triple. That's the entire LLM involvement;
everything after that is deterministic:

- **Entity resolution** (`graph.py`) — "NVIDIA Corporation", "Nvidia Corp.",
  and "NVIDIA" all normalize to the same entity via corporate-suffix
  stripping and punctuation normalization, no LLM judgment needed.
- **Linking to your tracked companies** — when an entity's normalized name
  matches a company you've already added (`add <TICKER> <NAME>`), it's
  linked automatically (`db.upsert_entity`), retroactively too if you add
  the company after the entity was first seen.
- **Predicate normalization** — "acquires"/"acquired"/"buys" collapse to one
  edge type (`acquired`) so the same real-world relationship doesn't
  fragment into near-duplicate edges depending on which source's exact
  wording got extracted.

```bash
python -m databroker.cli graph NVDA          # entities/relationships involving one company
python -m databroker.cli connections         # relationships connecting 2+ watchlist companies
```

`connections` is the actual payoff: if you're tracking both NVDA and, say, a
company that depends heavily on Hugging Face, "NVIDIA acquired Hugging Face"
surfaces as a direct connection between two of your holdings instead of
being buried in two separate, seemingly unrelated research reports.

I tested this end-to-end with a fake extraction call that mimicked the real
NVIDIA/Hugging Face acquisition news: the relationship was correctly stored,
both companies were correctly resolved to their tracked entities, and
`connections` correctly surfaced the link — and separately confirmed that a
relationship where only one side is a tracked company shows up in that
company's individual `graph` view but correctly does *not* pollute
`connections` (which requires both sides to be holdings).

**Known limitation:** entity resolution is name-normalization only, not true
disambiguation — two different real-world entities that happen to normalize
to the same string (rare, but possible with common short names) would
incorrectly merge. Given how infrequently claims state explicit
company-to-company relationships, the graph will also fill in slowly at
first — it grows only as fast as `research`/`sweep`/`ask` actually run and
happen to find relationship-bearing claims.

## Scheduled monitoring (Phase 2)

Every company on the watchlist can be checked automatically instead of only
on demand. Three ways to run it — pick whichever fits how you already work.

**Option A — `sweep`, driven by your OS's own scheduler (recommended for
swing/longer-horizon thesis tracking).** No process to keep alive, and it
plays nicely with free-tier API quotas that reset daily.

```bash
# Linux/Mac: crontab -e, then add a line like:
0 8 * * * cd /path/to/databroker && /usr/bin/python3 -m databroker.cli sweep >> sweep.log 2>&1

# Windows: Task Scheduler -> Create Task -> Trigger: Daily at 8:00 AM
#   Action: Program: python   Arguments: -m databroker.cli sweep   Start in: <project folder>
```

**Option B — `watch-loop`, a long-running process, daily mode** for anyone
who'd rather leave a terminal or a small server running than configure an OS
scheduler:

```bash
python -m databroker.cli watch-loop --at 08:00
```

It sleeps until the next occurrence of that local time, sweeps the whole
watchlist, prints the digest, and repeats.

**Option C — `watch-loop`, fast polling mode, for day trading.** Once a day
is too slow if you're trading intraday — this sweeps every N minutes
instead:

```bash
python -m databroker.cli watch-loop --interval-minutes 5
```

By default this only sweeps while the market is actually open (US equities
regular hours, 9:30–16:00 America/New_York) — a closed-market tick is
skipped with no API calls made at all, so free-tier LLM/search/financial
quota isn't burned overnight or on weekends. Use `--market-tz` for a
different market, or `--24-7` to poll around the clock regardless (useful
for crypto tickers, which trade continuously):

```bash
python -m databroker.cli watch-loop --interval-minutes 5 --24-7        # ignore market hours
python -m databroker.cli watch-loop --interval-minutes 5 --market-tz Europe/London
```

If you've also set `ALPACA_API_KEY_ID`/`ALPACA_API_SECRET_KEY` (the same
credentials `FINANCIAL_BACKEND=alpaca` uses, reused here for a different
purpose), the market-hours gate upgrades from a plain weekday+fixed-hours
guess to Alpaca's actual trading calendar: it correctly treats real market
holidays as closed, and uses each day's real close time rather than
assuming 16:00 (e.g. early-close days like the day after Thanksgiving).
Without those credentials, the gate falls back to the plain weekday+hours
check, which is still fine — the only cost of the fallback is an occasional
wasted sweep on a holiday, never a missed one on a real trading day.

None of the three modes need an extra dependency beyond `tzdata` (for the
market-hours timezone lookup — see `requirements.txt`; most Linux/Mac
installs already have this via the OS).

Both commands accept `--delay` (seconds paused between companies during a
sweep, default 3) — a small courtesy delay for free-tier cloud APIs with a
requests-per-minute limit; set `--delay 0` if you're on a fully local Ollama
setup with nothing to rate-limit against.

Each swept company gets investigated with a generic "what's new" objective
and its `last_swept_at` timestamp updated, so you always know how fresh the
watchlist is (`python -m databroker.cli events <TICKER>` and the digest both
reflect whatever the most recent sweep or manual `research` call found).

## Architecture

```
CLI (cli.py)
   |
Agent Orchestrator (agent.py: ResearchAgent)
   |  - plan_research()          skip via heuristics.is_simple_objective(), else 1 LLM call
   |  - plan_search_queries()    heuristics.build_search_queries() — 0 LLM calls
   |  - gather_evidence()        search -> fetch -> extract (1 LLM call, the core "research" step)
   |  - cross_check_and_store()  heuristics.find_overlapping_claim_pairs() filters what reaches the LLM
   |  - dedupe_against_history() heuristics.classify_against_history() resolves most cases for free
   |  - score_importance()       heuristics.baseline_importance() resolves most cases for free
   |  - assess_thesis_points()   1 LLM call, but only for claims that clear the importance bar
   |  - build_report()           0 LLM calls if no evidence was found, else 1 call
   |  - daily_digest()           0 LLM calls — pure DB read
   |
   +--- LLM Provider (llm.py)      Ollama | freellmapi | Groq | Gemini | Hybrid | Claude(optional) | Mock
   +--- Heuristics (heuristics.py) zero-token search/dedup/scoring/conflict logic
   +--- Search Provider (search.py)   DuckDuckGo (default) | Tavily (optional) | Mock
   +--- Page Fetcher (fetcher.py)     static (default) | Browser Agent (browser.py, opt-in — §18)
   +--- Social Provider (social.py)   Reddit (OAuth pool via reddit_client.py, or public fallback) | Hacker News | off (default) — Phase 4, opt-in
   +--- Financial Provider (social.py) SEC EDGAR | Finnhub | Alpaca | Alpha Vantage | StockTwits | off (default) — "first line" ticker-scoped sources, opt-in, combinable
   |
Scheduling (monitor.py)          sweep_once() / run_loop() — Phase 2, stdlib only
   |
Persistent store (db.py)         SQLite: companies, watchlist, thesis,
                                  thesis_points, sources, claims, conflicts,
                                  events, research_sessions, entities,
                                  relationships (Phase 5 knowledge graph)
```

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env    # then edit .env with your chosen backend + key
```

Using `.env` is strongly recommended over `export`/`$env:` — shell-session
env vars vanish the moment you close the terminal, which is a common source
of "it worked yesterday, why is it hitting the wrong backend today" (e.g.
`LLM_BACKEND` resetting to `auto` and silently picking a local Ollama with no
model pulled, instead of the Groq key you meant to use). `.env` persists
across sessions and is loaded automatically — no extra step needed beyond
having the file present in the directory you run the CLI from.

If you'd rather not use `.env`, the equivalent `export`/`$env:` commands work
the same way, just only for that terminal session:

```bash
export LLM_BACKEND=ollama                    # + ollama pull qwen2.5:7b && ollama serve
export LLM_BACKEND=freellmapi; export FREELLMAPI_API_KEY=freellmapi-...  # + freellmapi running locally (docker compose up)
export LLM_BACKEND=groq; export GROQ_API_KEY=gsk_...       # free at console.groq.com
export LLM_BACKEND=gemini; export GEMINI_API_KEY=...        # free at aistudio.google.com/apikey
export LLM_BACKEND=hybrid; export GROQ_API_KEY=gsk_...      # + Ollama running too
```

Either way, run `python -m databroker.cli doctor` to confirm what's
configured — and every command that actually calls an LLM (`research`,
`ask`, `sweep`, `watch-loop`) prints which provider+model it resolved to
before doing any work, e.g. `Using LLM provider: Groq (openai/gpt-oss-120b)`,
so a wrong-backend mistake is visible immediately rather than showing up as
a confusing request error partway through.

Optional tuning via env vars (put these in `.env` too):
- `MAX_SEARCH_HITS_PER_QUESTION` (default 6), `MAX_FETCH_CHARS` (default 1500) —
  smaller values mean smaller prompts for the extraction step, useful for a
  small local model's context window.
- `RESEARCH_MAX_SUBQUESTIONS` (default 6) — caps how many sub-questions (and
  therefore how many downstream extraction calls) one research objective can
  fan out into.
- `SOCIAL_BACKEND` (default `off`), `REDDIT_SUBREDDITS`,
  `MAX_SOCIAL_HITS_PER_QUESTION` (default 3) — see "Social & Community
  Intelligence" below.

## Usage

```bash
python -m databroker.cli add NVDA "NVIDIA Corporation"
python -m databroker.cli watch NVDA --reason "AI infrastructure thesis"
python -m databroker.cli set-thesis NVDA
python -m databroker.cli research NVDA "Is competitive position improving or deteriorating?"
python -m databroker.cli thesis-status NVDA
python -m databroker.cli ask NVDA "Anything new?"
python -m databroker.cli events NVDA
python -m databroker.cli graph NVDA          # knowledge graph for one company
python -m databroker.cli connections         # relationships across your whole watchlist
python -m databroker.cli digest
python -m databroker.cli sweep              # one-shot monitoring pass, see "Scheduled monitoring" below
python -m databroker.cli watch-loop --at 08:00               # daily long-running loop
python -m databroker.cli watch-loop --interval-minutes 5     # fast polling loop, for day trading
```

Data persists in `~/.databroker/databroker.db` (SQLite) between runs.

## Extending further

All phases from the original spec are now covered in some form (§18 and
Phase 5 in a deliberately scoped-down way — see their sections above for
known limitations). The following are natural next increments rather than
missing functionality:

- **Knowledge graph depth** — currently a single relationship type per
  claim, resolved by name-normalization only. A dedicated entity-
  disambiguation pass (e.g. using company ticker/domain as a secondary key,
  not just normalized name) would handle the rare case of two different
  companies sharing a short common name.
- **More financial platforms** — `social.py`'s `FinancialProvider` interface
  now has five implementations (SEC EDGAR, Finnhub, Alpaca, Alpha Vantage,
  StockTwits), combined via `CombinedFinancialProvider`. Adding another
  (e.g. Marketaux for broader international coverage, EDGAR full-text
  search) is one more class plus one line in `build_financial_from_env()` —
  the "first line" merging logic in `agent.py`'s `investigate()` doesn't
  need to change, since it already fetches once per session and merges
  whatever the provider(s) return, and each provider can supply its own
  `context_label` for how the LLM should treat that source type. (Direct
  Benzinga integration was considered and explicitly skipped — Benzinga's
  own API is enterprise/paid, and the one free tier it offers, headline +
  teaser only via AWS Marketplace, is strictly less than what `alpaca`
  already provides for free, since Alpaca's feed is itself Benzinga-sourced.)
- **Market-holiday awareness for fast polling** — done: `is_market_open()`
  now consults Alpaca's `/v2/calendar` endpoint when Alpaca credentials are
  available (per-date cached, falls back to the plain weekday+hours check
  on any lookup failure or if credentials aren't set). Remaining gap: this
  only covers US equities via Alpaca's calendar; a non-US market using
  `--market-tz` still gets the plain heuristic, since Alpaca's calendar is
  US-only.
- **SEC full-text search** — the `sec` backend currently reads a company's
  own recent filings via the submissions endpoint. EDGAR's separate
  full-text search API (`efts.sec.gov`) indexes filing text across *all*
  companies since 2001, which would let the agent search for a company's
  name mentioned in *other* companies' filings (e.g. a supplier or
  competitor's 10-K) — a genuinely different capability, not just more of
  the same data.
- **More social sources** — a subreddit-specific RSS feed or similar would
  be one more `SocialProvider` plus one line in `build_social_from_env()`.
- **Extend the browser agent** — `browser.py`'s `max_hops` is currently 1
  and the stopping condition is a simple text/link-count heuristic; raising
  the hop limit or adding a "this looks like a paywall/login wall" detector
  are the natural next steps if a source needs deeper navigation.
- **Tighten the heuristics further** — `heuristics.py` is deliberately
  simple (stdlib-only, no ML). If you find it escalating too much to the
  LLM (or too little), the thresholds are function parameters, not
  hardcoded — easy to tune per company or sector.
