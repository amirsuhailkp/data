# DataBroker AI — Personal Investment Research Agent (MVP)

A working Phase 1–3 implementation of the DataBroker spec: a personal, persistent
AI research agent that investigates companies, tracks your investment thesis,
remembers past research, and tells you what actually changed and why it
matters — without predicting prices or placing trades.

**No paid model is required**, and **the LLM is not on the hook for every
decision.** Two design choices work together to make this cheap enough to run
on free-tier APIs or a local model:

1. **Free/local model backends** (`llm.py`) — Ollama (fully local), Groq or
   Gemini (free-tier cloud), or a hybrid mix.
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
| **Ollama** (local) | Free | Depends on your hardware/model | `ollama pull qwen2.5:7b`, then `ollama serve` |
| **Groq** (cloud) | Free tier (rate-limited) | Good — runs large open models fast | Free key at console.groq.com |
| **Gemini** (cloud) | Free tier (rate-limited) | Good | Free key at aistudio.google.com/apikey |
| **Hybrid** | Free | Best free option — cloud model for plan/report/thesis, local for the rest | Set up both Ollama and one cloud key |

Search defaults to DuckDuckGo (free, no signup). Tavily is an optional
upgrade (`SEARCH_BACKEND=tavily`, needs a free key).

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
| §16 Model abstraction | ✅ Ollama / Groq / Gemini / Hybrid / Claude(optional) / Mock, all behind one interface |
| §23 Daily Intelligence Brief | ✅ `digest` — pure DB read, zero LLM calls |
| §26 Boundary: research, not trading | ✅ enforced in the system prompt |
| Scheduled/background monitoring (Phase 2) | ✅ `sweep` (cron-friendly one-shot) and `watch-loop` (long-running, stdlib-only, no new dependency) |
| §5 Social & Community Intelligence (Phase 4) | ✅ opt-in (`SOCIAL_BACKEND`) — Reddit public search + Hacker News (Algolia API), both free/no-key, feeding the same claim pipeline — see below |
| §18 Full browser agent (dynamic navigation) | 🚧 `fetcher.py` does a single static fetch per URL, not multi-step navigation |
| Knowledge graph (Phase 5) | 🚧 current schema is relational |

## Social & Community Intelligence (Phase 4)

Off by default — turn it on with `SOCIAL_BACKEND`:

```bash
export SOCIAL_BACKEND=reddit       # or: hackernews | both
export REDDIT_SUBREDDITS=stocks,investing,wallstreetbets   # optional, scopes Reddit search
```

Both sources are free and need no signup or API key: Reddit's public
(unauthenticated) search JSON endpoint, and Hacker News via Algolia's
official public Search API. When enabled, a few community-discussion hits
are pulled alongside the normal web search results for every sub-question
and fed into the *same* extraction call — turning this on does not add any
extra LLM calls.

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

## Scheduled monitoring (Phase 2)

Every company on the watchlist can be checked automatically instead of only
on demand. Two ways to run it — pick whichever fits how you already work:

**Option A — `sweep`, driven by your OS's own scheduler (recommended).**
No process to keep alive, and it plays nicely with free-tier API quotas that
reset daily.

```bash
# Linux/Mac: crontab -e, then add a line like:
0 8 * * * cd /path/to/databroker && /usr/bin/python3 -m databroker.cli sweep >> sweep.log 2>&1

# Windows: Task Scheduler -> Create Task -> Trigger: Daily at 8:00 AM
#   Action: Program: python   Arguments: -m databroker.cli sweep   Start in: <project folder>
```

**Option B — `watch-loop`, a long-running process** for anyone who'd rather
leave a terminal or a small server running than configure an OS scheduler:

```bash
python -m databroker.cli watch-loop --at 08:00
```

It sleeps until the next occurrence of that local time, sweeps the whole
watchlist, prints the digest, and repeats — no extra dependency (`time`/
`datetime` from the standard library only).

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
   +--- LLM Provider (llm.py)      Ollama | Groq | Gemini | Hybrid | Claude(optional) | Mock
   +--- Heuristics (heuristics.py) zero-token search/dedup/scoring/conflict logic
   +--- Search Provider (search.py)   DuckDuckGo (default) | Tavily (optional) | Mock
   +--- Page Fetcher (fetcher.py)     requests + trafilatura/BeautifulSoup
   +--- Social Provider (social.py)   Reddit | Hacker News | off (default) — Phase 4, opt-in
   |
Scheduling (monitor.py)          sweep_once() / run_loop() — Phase 2, stdlib only
   |
Persistent store (db.py)         SQLite: companies, watchlist, thesis,
                                  thesis_points, sources, claims, conflicts,
                                  events, research_sessions
```

## Setup

```bash
pip install -r requirements.txt

# Pick ONE (or set nothing and it auto-detects):
export LLM_BACKEND=ollama                    # + ollama pull qwen2.5:7b && ollama serve
export LLM_BACKEND=groq; export GROQ_API_KEY=gsk_...       # free at console.groq.com
export LLM_BACKEND=gemini; export GEMINI_API_KEY=...        # free at aistudio.google.com/apikey
export LLM_BACKEND=hybrid; export GROQ_API_KEY=gsk_...      # + Ollama running too

python -m databroker.cli doctor   # confirms what got picked
```

Optional tuning via env vars:
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
python -m databroker.cli digest
python -m databroker.cli sweep              # one-shot monitoring pass, see "Scheduled monitoring" below
python -m databroker.cli watch-loop --at 08:00   # or run this as a long-lived process instead
```

Data persists in `~/.databroker/databroker.db` (SQLite) between runs.

## Extending further

- **Social/community intelligence tuning** — `social.py` currently covers
  Reddit and Hacker News. Adding another free source (e.g. a subreddit-
  specific RSS feed, or StockTwits' public symbol streams) means writing one
  more `SocialProvider` and adding it to `build_social_from_env()` — the
  extraction, source-type-override, and importance-scoring logic in
  `agent.py`/`heuristics.py` doesn't need to change.
- **Real browser agent** — swap `fetcher.py`'s static fetch for a headless
  browser driven by an LLM decision loop, still returning plain text into
  the same `gather_evidence()` pipeline.
- **Tighten the heuristics further** — `heuristics.py` is deliberately
  simple (stdlib-only, no ML). If you find it escalating too much to the
  LLM (or too little), the thresholds are function parameters, not
  hardcoded — easy to tune per company or sector.
