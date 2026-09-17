"""
DataBroker CLI — thin interface over the agent + db.

Usage examples:
    python -m databroker.cli add NVDA "NVIDIA Corporation"
    python -m databroker.cli watch NVDA --reason "AI infra thesis"
    python -m databroker.cli set-thesis NVDA
    python -m databroker.cli research NVDA "How is competitive position trending?"
    python -m databroker.cli ask NVDA "Anything new?"
    python -m databroker.cli thesis-status NVDA
    python -m databroker.cli digest
"""

from __future__ import annotations
import argparse
import sys
from .db import DB
from .llm import build_provider_from_env
from .search import build_search_from_env
from .social import build_social_from_env, build_financial_from_env, build_confirmation_financial_from_env
from .technicals import build_technicals_from_env
from .fetcher import PageFetcher
from .agent import ResearchAgent


def _load_dotenv():
    """Optional .env support so config (LLM_BACKEND, GROQ_API_KEY, etc.) persists
    across terminal sessions instead of needing `$env:X="Y"` / `export X=Y` retyped
    every time. Looks for a .env file in the current directory. Silently does
    nothing if python-dotenv isn't installed or no .env file exists — this is a
    convenience, not a requirement."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def build_fetcher(llm):
    """FETCHER_BACKEND=static (default) | browser. Browser mode needs
    `pip install playwright && playwright install chromium` — falls back to
    static automatically if Playwright isn't actually installed, so this is
    safe to leave on `browser` even before you've set that up."""
    import os
    if os.environ.get("FETCHER_BACKEND", "static").lower() == "browser":
        from .browser import BrowserFetcher, NavigationAgent
        return BrowserFetcher(navigator=NavigationAgent(llm))
    return PageFetcher()


def build_agent(db: DB, announce: bool = True, notifier=None) -> ResearchAgent:
    llm = build_provider_from_env()
    if announce:
        print(f"[databroker] Using LLM provider: {llm.describe()}", file=sys.stderr)
    return ResearchAgent(
        db=db,
        llm=llm,
        search=build_search_from_env(),
        fetcher=build_fetcher(llm),
        social=build_social_from_env(),
        financial=build_financial_from_env(),
        notifier=notifier,
        confirmer=build_confirmation_financial_from_env(),
        technicals=build_technicals_from_env(),
    )


def build_parser():
    p = argparse.ArgumentParser(prog="databroker")
    sub = p.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Add a company")
    add.add_argument("ticker")
    add.add_argument("name")

    watch = sub.add_parser("watch", help="Add a company to the watchlist")
    watch.add_argument("ticker")
    watch.add_argument("--reason", default="")

    thesis = sub.add_parser("set-thesis", help="Record an investment thesis interactively")
    thesis.add_argument("ticker")

    thesis_status = sub.add_parser("thesis-status", help="Show current thesis point statuses")
    thesis_status.add_argument("ticker")

    research = sub.add_parser("research", help="Run a research objective now")
    research.add_argument("ticker")
    research.add_argument("objective")

    ask = sub.add_parser("ask", help="Ask a natural question; checks memory + does light research")
    ask.add_argument("ticker")
    ask.add_argument("question")

    sub.add_parser("digest", help="Generate the daily intelligence brief across the watchlist")

    sub.add_parser("doctor", help="Show which LLM backend and search backend would be used right now")

    events = sub.add_parser("events", help="List recorded events for a company")
    events.add_argument("ticker")

    graph = sub.add_parser(
        "graph",
        help="Show the knowledge graph — entities and relationships discovered for a company",
    )
    graph.add_argument("ticker")

    sub.add_parser(
        "connections",
        help="Show relationships that connect two or more of your watchlist companies "
             "(e.g. one acquired another, one is a supplier to another)",
    )

    sweep = sub.add_parser(
        "sweep",
        help="Run one monitoring pass across the whole watchlist, then print the digest "
             "(cron-friendly — see README for crontab/Task Scheduler examples)",
    )
    sweep.add_argument("--delay", type=float, default=3.0,
                        help="Seconds to pause between companies (default 3; use 0 for local-only Ollama setups)")

    watch_loop = sub.add_parser(
        "watch-loop",
        help="Run forever, sweeping the watchlist (alternative to cron/Task Scheduler for a "
             "long-running process). Default: once a day at a fixed local time. Pass "
             "--interval-minutes for fast day-trading-style polling instead.",
    )
    watch_loop.add_argument("--at", default="08:00",
                             help="Local time for the daily sweep, HH:MM (default 08:00). "
                                  "Ignored if --interval-minutes is set.")
    watch_loop.add_argument("--delay", type=float, default=3.0,
                             help="Seconds to pause between companies during each sweep")
    watch_loop.add_argument("--interval-minutes", type=float, default=None,
                             help="Switch to fast polling mode: sweep every N minutes instead "
                                  "of once a day. For day trading, where a daily check is too "
                                  "slow. Combine with --24-7 to also poll outside market hours.")
    watch_loop.add_argument("--24-7", dest="around_the_clock", action="store_true",
                             help="With --interval-minutes: keep polling even when the market "
                                  "is closed (e.g. for crypto tickers). Default is to skip "
                                  "closed-market ticks so free-tier quota isn't wasted overnight.")
    watch_loop.add_argument("--market-tz", default="America/New_York",
                             help="IANA timezone for the market-hours gate (default: America/New_York, "
                                  "i.e. US equities). Ignored with --24-7.")
    watch_loop.add_argument("--discover-every-hours", type=float, default=None,
                             help="Also run market-wide discovery this often (e.g. 6), alerting you "
                                  "with new watchlist candidates. Requires DISCOVER_BACKEND. Kept on "
                                  "its own slow schedule deliberately — the day's top movers don't "
                                  "change every 5 minutes, so tying it to the sweep interval would "
                                  "just re-alert the same symbols all day.")
    watch_loop.add_argument("--discover-min-score", type=int, default=3,
                             help="Corroboration score for scheduled discovery (default 3)")
    watch_loop.add_argument("--discover-limit", type=int, default=5,
                             help="Max candidates per scheduled discovery run (default 5)")

    discover = sub.add_parser(
        "discover",
        help="Scan the market for symbols you're NOT watching that show unusual activity, "
             "and propose them as watchlist candidates (requires DISCOVER_BACKEND)",
    )
    discover.add_argument("--min-score", type=int, default=3,
                           help="Corroboration score a symbol must reach to be proposed "
                                "(default 3). Higher = stricter/fewer; lower = noisier. "
                                "A single weak signal scores 1-2 and won't pass at the default.")
    discover.add_argument("--limit", type=int, default=5,
                           help="Maximum candidates to propose (default 5)")

    selftest = sub.add_parser(
        "selftest",
        help="End-to-end preflight: actually exercise every configured piece (LLM, search, "
             "news sources, scanners, Telegram alert, DB) and report what really works",
    )
    selftest.add_argument("--ticker", default="AAPL",
                           help="Ticker used to probe the data sources (default AAPL — pick a "
                                "liquid, well-covered name so 'no results' means a real problem)")
    selftest.add_argument("--market-tz", default="America/New_York",
                           help="IANA timezone for the market-hours check (default America/New_York)")

    return p


def cmd_add(db: DB, args):
    cid = db.add_company(args.ticker, args.name)
    print(f"Added {args.name} ({args.ticker.upper()}), id={cid}")


def cmd_watch(db: DB, args):
    company = db.get_company(args.ticker)
    if not company:
        print(f"Unknown ticker {args.ticker}. Add it first with `add`.")
        sys.exit(1)
    db.watch(company["id"], args.reason)
    print(f"Watching {args.ticker.upper()}.")


def cmd_set_thesis(db: DB, args):
    company = db.get_company(args.ticker)
    if not company:
        print(f"Unknown ticker {args.ticker}. Add it first with `add`.")
        sys.exit(1)
    print("Enter thesis summary (one line):")
    summary = input("> ")
    print("Enter thesis points, one per line. Blank line to finish.")
    points = []
    while True:
        line = input("- ")
        if not line.strip():
            break
        points.append(line.strip())
    tid = db.set_thesis(company["id"], summary, points)
    print(f"Saved thesis id={tid} with {len(points)} points.")


def cmd_thesis_status(db: DB, args):
    company = db.get_company(args.ticker)
    if not company:
        print(f"Unknown ticker {args.ticker}.")
        sys.exit(1)
    thesis, points = db.get_latest_thesis(company["id"])
    if not thesis:
        print("No thesis recorded yet.")
        return
    print(f"Thesis: {thesis['summary']}\n")
    icon = {"supported": "✓", "weakened": "⚠", "contradicted": "✗",
            "new_risk": "⚠", "new_opportunity": "✓", "unassessed": "·"}
    for p in points:
        mark = icon.get(p["status"], "·")
        note = f" — {p['last_note']}" if p["last_note"] else ""
        print(f"  {mark} {p['point_text']} [{p['status']}]{note}")


def cmd_research(db: DB, args):
    company = db.get_company(args.ticker)
    if not company:
        print(f"Unknown ticker {args.ticker}. Add it first with `add`.")
        sys.exit(1)
    agent = build_agent(db)
    result = agent.investigate(args.ticker, company["name"], company["id"], args.objective)
    print(result.report)
    if result.conflicts:
        print("\n--- Conflicts detected ---")
        for c in result.conflicts:
            print(f"- {c.get('description')}")


def cmd_ask(db: DB, args):
    company = db.get_company(args.ticker)
    if not company:
        print(f"Unknown ticker {args.ticker}. Add it first with `add`.")
        sys.exit(1)
    recent = db.recent_events(company["id"], days=30)
    if recent:
        print(f"(Checked memory first — {len(recent)} development(s) recorded in the last 30 days.)\n")
    agent = build_agent(db)
    result = agent.investigate(args.ticker, company["name"], company["id"], args.question)
    print(result.report)


def cmd_digest(db: DB, args):
    agent = build_agent(db)
    print(agent.daily_digest())


def cmd_events(db: DB, args):
    company = db.get_company(args.ticker)
    if not company:
        print(f"Unknown ticker {args.ticker}.")
        sys.exit(1)
    rows = db.recent_events(company["id"], days=365)
    if not rows:
        print("No events recorded yet.")
        return
    for r in rows:
        print(f"[{r['importance_level'].upper():8}] {r['title']} — {r['description'][:100]} "
              f"(status={r['status']}, thesis_impact={r['thesis_impact']})")


def cmd_graph(db: DB, args):
    company = db.get_company(args.ticker)
    if not company:
        print(f"Unknown ticker {args.ticker}.")
        sys.exit(1)
    entity = db.get_entity_by_company(company["id"])
    if not entity:
        print(f"No graph entity for {args.ticker} yet — run `research` a few times first "
              "(entities/relationships are extracted alongside regular research, not separately).")
        return
    rels = db.get_relationships_for_entity(entity["id"])
    if not rels:
        print(f"No relationships recorded for {company['name']} yet.")
        return
    print(f"Knowledge graph for {company['name']} ({args.ticker.upper()}):\n")
    for r in rels:
        arrow = f"{r['subject_name']} --[{r['predicate']}]--> {r['object_name']}"
        print(f"  {arrow}  (confidence={r['confidence']})")


def cmd_connections(db: DB, args):
    rows = db.find_cross_watchlist_connections()
    if not rows:
        print("No relationships connecting two or more watchlist companies found yet — "
              "run `research`/`sweep` on your watchlist a few times first.")
        return
    print("Connections across your watchlist:\n")
    for r in rows:
        print(f"  {r['subject_name']} ({r['subject_ticker']}) --[{r['predicate']}]--> "
              f"{r['object_name']} ({r['object_ticker']})")


def cmd_doctor(db: DB, args):
    import os
    from .llm import OllamaProvider, FreeLLMAPIProvider

    print("LLM_BACKEND     =", os.environ.get("LLM_BACKEND", "auto (not set)"))
    print("Ollama running? =", OllamaProvider.is_available())
    if os.environ.get("FREELLMAPI_API_KEY"):
        freellmapi_url = os.environ.get("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")
        if FreeLLMAPIProvider.is_available(freellmapi_url):
            print(f"freellmapi      = reachable at {freellmapi_url}, model="
                  f"{os.environ.get('FREELLMAPI_MODEL', 'auto')}")
        else:
            print(f"freellmapi      = FREELLMAPI_API_KEY is set but {freellmapi_url} is NOT reachable — "
                  f"is the container running? (`docker compose up` in the freellmapi repo)")
    else:
        print("freellmapi      = not configured (set FREELLMAPI_API_KEY to use it)")
    print("GROQ_API_KEY    =", "set" if os.environ.get("GROQ_API_KEY") else "not set")
    print("GEMINI_API_KEY  =", "set" if os.environ.get("GEMINI_API_KEY") else "not set")
    print("SEARCH_BACKEND  =", os.environ.get("SEARCH_BACKEND", "duckduckgo (default)"))
    print("SOCIAL_BACKEND  =", os.environ.get("SOCIAL_BACKEND", "off (default)"))
    if os.environ.get("SOCIAL_BACKEND", "off").lower() in ("reddit", "both"):
        from .reddit_client import parse_credentials_from_env
        creds = parse_credentials_from_env()
        if creds:
            print(f"  Reddit auth   = OAuth, {len(creds)} credential(s) pooled "
                  f"(~{len(creds) * 95} req/min combined budget)")
        else:
            print("  Reddit auth   = none configured — using unauthenticated public search "
                  "(lower, less predictable rate limit). Set REDDIT_CLIENT_ID(_N)/"
                  "REDDIT_CLIENT_SECRET(_N) to use the OAuth pool instead.")
    fin_backend = os.environ.get("FINANCIAL_BACKEND", "off")
    print("FINANCIAL_BACKEND =", fin_backend or "off (default)")
    fin_names = [b.strip().lower() for b in fin_backend.split(",") if b.strip()]
    if "finnhub" in fin_names:
        print("  Finnhub       =", "API key set" if os.environ.get("FINNHUB_API_KEY") else
              "NOT configured — set FINNHUB_API_KEY (free tier at finnhub.io) or this source is skipped")
    if "sec" in fin_names:
        print("  SEC EDGAR     =", f"contact set ({os.environ.get('SEC_EDGAR_CONTACT')})" if os.environ.get("SEC_EDGAR_CONTACT")
              else "no SEC_EDGAR_CONTACT set — works, but SEC's fair-access policy asks for a contact string in the User-Agent")
    if "alpaca" in fin_names:
        has_key = bool(os.environ.get("ALPACA_API_KEY_ID"))
        has_secret = bool(os.environ.get("ALPACA_API_SECRET_KEY"))
        if has_key and has_secret:
            print("  Alpaca news   = credentials set")
        else:
            missing = [n for n, v in (("ALPACA_API_KEY_ID", has_key), ("ALPACA_API_SECRET_KEY", has_secret)) if not v]
            print(f"  Alpaca news   = NOT configured — missing {', '.join(missing)} "
                  "(free from a paper-trading account at alpaca.markets) — this source is skipped")
    # Alpha Vantage's confirmation mode is independent of FINANCIAL_BACKEND —
    # it's driven purely by the key + ALPHA_VANTAGE_MODE, so report it separately.
    av_mode = os.environ.get("ALPHA_VANTAGE_MODE", "confirm").lower()
    if not os.environ.get("ALPHA_VANTAGE_API_KEY"):
        if "alphavantage" in fin_names:
            print("  Alpha Vantage = NOT configured — set ALPHA_VANTAGE_API_KEY (free at "
                  "alphavantage.co) or this source is skipped")
    elif av_mode == "confirm":
        print("  Alpha Vantage = CONFIRM mode (default) — not polled every tick; spends one "
              "call only to corroborate an event another source already flagged as notable. "
              "Safe to run alongside 24/7 fast polling.")
    elif av_mode == "continuous":
        print("  Alpha Vantage = CONTINUOUS mode — polled every sweep like any other source. "
              "WARNING: 25 requests/day total will exhaust in minutes under fast interval "
              "polling; set ALPHA_VANTAGE_MODE=confirm unless you're doing daily sweeps only.")
        if "alphavantage" not in fin_names:
            print("                  ...but `alphavantage` is NOT in FINANCIAL_BACKEND, so it "
                  "isn't actually being polled. Add it, or switch back to confirm mode.")
    elif av_mode == "off":
        print("  Alpha Vantage = off (ALPHA_VANTAGE_MODE=off) — key set but unused.")
    else:
        print(f"  Alpha Vantage = unknown ALPHA_VANTAGE_MODE='{av_mode}' — treated as off.")
    if os.environ.get("ALPACA_API_KEY_ID") and os.environ.get("ALPACA_API_SECRET_KEY"):
        print("  watch-loop --interval-minutes will use these same Alpaca credentials for a "
              "holiday-aware market-hours calendar (falls back to a plain weekday+hours check "
              "without them).")
    tech_backend = os.environ.get("TECHNICALS_BACKEND", "off")
    print("TECHNICALS_BACKEND =", tech_backend or "off (default)")
    if tech_backend.lower() == "alpaca":
        has_key = bool(os.environ.get("ALPACA_API_KEY_ID"))
        has_secret = bool(os.environ.get("ALPACA_API_SECRET_KEY"))
        if has_key and has_secret:
            print("  Chart data    = Alpaca historical bars (IEX feed), reusing Alpaca credentials")
        else:
            missing = [n for n, v in (("ALPACA_API_KEY_ID", has_key), ("ALPACA_API_SECRET_KEY", has_secret)) if not v]
            print(f"  Chart data    = NOT configured — missing {', '.join(missing)} — this source is skipped")
    disc_backend = os.environ.get("DISCOVER_BACKEND", "off")
    print("DISCOVER_BACKEND =", disc_backend or "off (default)")
    if disc_backend.lower() not in ("off", ""):
        from .discover import build_scanners_from_env
        scanners = build_scanners_from_env()
        if scanners:
            for sc in scanners:
                print(f"  scanner       = {sc.describe()}")
        else:
            print("  scanner       = NONE usable — `alpaca` needs ALPACA_API_KEY_ID + "
                  "ALPACA_API_SECRET_KEY; `discover` will do nothing")
    notify_backend = os.environ.get("NOTIFY_BACKEND", "off")
    print("NOTIFY_BACKEND  =", notify_backend or "off (default)")
    if notify_backend.lower() == "telegram":
        has_token = bool(os.environ.get("TELEGRAM_BOT_TOKEN"))
        has_chat = bool(os.environ.get("TELEGRAM_CHAT_ID"))
        if has_token and has_chat:
            print(f"  Telegram      = configured, chat {os.environ.get('TELEGRAM_CHAT_ID')}")
        else:
            missing = [n for n, v in (("TELEGRAM_BOT_TOKEN", has_token), ("TELEGRAM_CHAT_ID", has_chat)) if not v]
            print(f"  Telegram      = NOT configured — missing {', '.join(missing)} (see README's "
                  "Alerts section) — sweep/watch-loop will run but won't send alerts")
    print("FETCHER_BACKEND =", os.environ.get("FETCHER_BACKEND", "static (default)"))
    if os.environ.get("FETCHER_BACKEND", "static").lower() == "browser":
        try:
            import playwright  # noqa: F401
            print("Playwright       = installed")
        except ImportError:
            print("Playwright       = NOT installed — will silently fall back to static fetch. "
                  "Run: pip install playwright && playwright install chromium")
    try:
        provider = build_provider_from_env()
        print("\n-> Resolved LLM provider:", type(provider).__name__)
    except ValueError as e:
        print(f"\n-> LLM provider resolution FAILED: {e}")


def _maybe_notify(notifier, digest: str) -> None:
    """Push the digest as an alert if there's actually something new in it.
    daily_digest() returns a fixed placeholder string when there's nothing to
    report, which is the signal not to send a pointless empty alert."""
    if notifier is None:
        return
    if digest.startswith("No new developments"):
        return
    if notifier.send(digest):
        print("[databroker] Alert sent.")
    else:
        print("[databroker] Alert FAILED to send (see error above) — digest is still shown here.")


def format_candidates(candidates: list[dict]) -> str:
    """Shared by the `discover` command's console output and its Telegram
    alert, so both say exactly the same thing."""
    lines = ["🔍 Potential watchlist candidates — unusual market activity today:"]
    for c in candidates:
        lines.append(f"\n**{c['symbol']}** (corroboration score {c['score']})")
        for detail in c["signals"].values():
            lines.append(f"  • {detail}")
        if c.get("note"):
            lines.append(f"  {c['note']}")
    lines.append(
        "\n⚠️ These are research leads, not recommendations. Unusual activity is equally "
        "consistent with good news, bad news, and manipulation — nothing here is evidence "
        "that a symbol is worth buying. Add one with `watch <TICKER>` to start tracking it "
        "properly."
    )
    return "\n".join(lines)


def cmd_discover(db: DB, args):
    from .discover import build_scanners_from_env
    from .notify import build_notifier_from_env

    scanners = build_scanners_from_env()
    if not scanners:
        print("No market scanners configured. Set DISCOVER_BACKEND=alpaca,stocktwits "
              "in your .env (see README's Discovery section).")
        return

    notifier = build_notifier_from_env()
    agent = build_agent(db, notifier=notifier)
    # Don't propose things already being tracked — the point is what you're NOT watching.
    exclude = {(row["ticker"] or "").upper() for row in db.list_watchlist()}
    candidates = agent.discover_candidates(
        scanners, exclude=exclude, min_score=args.min_score, limit=args.limit
    )
    if not candidates:
        print("No candidates cleared the corroboration threshold. Nothing proposed — "
              "lower --min-score to see weaker signals.")
        return

    text = format_candidates(candidates)
    print(text)
    if notifier is not None:
        notifier.send(text)



def cmd_selftest(db: DB, args):
    """End-to-end preflight: checks every configured piece actually works,
    using real network calls where a source is configured. Designed to be the
    one command you run after setup (or after changing .env) to find out what
    is genuinely working versus merely present in the config."""
    import os
    from .discover import build_scanners_from_env
    from .notify import build_notifier_from_env
    from .social import build_financial_from_env, build_confirmation_financial_from_env

    ticker = args.ticker.upper()
    results = []  # (name, ok|warn|fail, message)

    def record(name, status, msg):
        results.append((name, status, msg))
        icon = {"ok": "✅", "warn": "⚠️ ", "fail": "❌"}[status]
        print(f"{icon} {name}: {msg}")

    print(f"Running self-test (probe ticker: {ticker})\n")

    # --- 1. LLM backend: a real round-trip, not just "is a key set" ---
    try:
        llm = build_provider_from_env()
        out = llm.complete_json(
            "You are a test harness. Reply with valid JSON only.",
            "Reply with exactly {\"ok\": true} and nothing else.",
            '{"ok": true}', task="general",
        )
        if isinstance(out, dict) and out:
            record("LLM", "ok", f"{llm.describe()} responded with parseable JSON")
        else:
            record("LLM", "fail", f"{llm.describe()} returned nothing usable — check credentials/model name")
    except Exception as e:
        record("LLM", "fail", f"round-trip failed: {e}")

    # --- 2. Search backend ---
    try:
        search = build_search_from_env()
        hits = search.search(f"{ticker} stock news", max_results=3)
        if hits:
            record("Search", "ok", f"{type(search).__name__} returned {len(hits)} result(s)")
        else:
            record("Search", "warn", f"{type(search).__name__} returned nothing "
                                      "(rate-limited, or mock backend)")
    except Exception as e:
        record("Search", "fail", f"{e}")

    # --- 3. Financial/social sources actually polled ---
    fin = build_financial_from_env()
    if fin is None:
        record("Financial sources", "warn", "none configured (FINANCIAL_BACKEND=off)")
    else:
        try:
            hits = fin.get_ticker_activity(ticker, max_results=5)
            if hits:
                record("Financial sources", "ok", f"{len(hits)} item(s) for {ticker}")
            else:
                record("Financial sources", "warn",
                       f"configured but returned nothing for {ticker} — could be a quiet ticker, "
                       "or a bad key silently failing soft")
        except Exception as e:
            record("Financial sources", "fail", f"{e}")

    social = build_social_from_env()
    if social is None:
        record("Social sources", "warn", "none configured (SOCIAL_BACKEND=off)")
    else:
        try:
            hits = social.search(f"{ticker} stock", max_results=5)
            record("Social sources", "ok" if hits else "warn",
                   f"{len(hits)} item(s)" if hits else "configured but returned nothing")
        except Exception as e:
            record("Social sources", "fail", f"{e}")

    # --- 4. Alpha Vantage confirmation path ---
    conf = build_confirmation_financial_from_env()
    av_mode = os.environ.get("ALPHA_VANTAGE_MODE", "confirm").lower()
    if conf is None:
        record("Confirmation source", "warn",
               f"not active (ALPHA_VANTAGE_MODE={av_mode}, key "
               f"{'set' if os.environ.get('ALPHA_VANTAGE_API_KEY') else 'NOT set'})")
    else:
        try:
            hits = conf.get_ticker_activity(ticker, max_results=3)
            if hits:
                record("Confirmation source", "ok", f"Alpha Vantage returned {len(hits)} item(s) "
                                                     "(1 of your 25 daily calls used by this test)")
            else:
                record("Confirmation source", "warn",
                       "returned nothing — no coverage for this ticker, daily quota exhausted, "
                       "or an invalid key (this API fails soft either way)")
        except Exception as e:
            record("Confirmation source", "fail", f"{e}")

    # --- 4b. Chart/technical data ---
    from .technicals import build_technicals_from_env
    tech = build_technicals_from_env()
    if tech is None:
        record("Chart data", "warn", "none configured (TECHNICALS_BACKEND=off)")
    else:
        try:
            snap = tech.get_snapshot(ticker)
            if snap:
                record("Chart data", "ok", f"{tech.describe()} returned a snapshot for {ticker}")
            else:
                record("Chart data", "warn",
                       f"configured but no snapshot for {ticker} — needs at least 20 days of "
                       "history; check the ticker is a real, actively-traded symbol")
        except Exception as e:
            record("Chart data", "fail", f"{e}")

    # --- 5. Market discovery scanners ---
    scanners = build_scanners_from_env()
    if not scanners:
        record("Discovery scanners", "warn", "none usable (DISCOVER_BACKEND off, or missing creds)")
    for sc in scanners:
        try:
            hits = sc.scan(max_results=10)
            if hits:
                record(f"Discovery: {sc.describe()}", "ok", f"{len(hits)} signal(s)")
            else:
                record(f"Discovery: {sc.describe()}", "warn",
                       "returned nothing — market may be closed, or credentials rejected")
        except Exception as e:
            record(f"Discovery: {sc.describe()}", "fail", f"{e}")

    # --- 6. Telegram alerts: actually send one ---
    notifier = build_notifier_from_env()
    if notifier is None:
        record("Alerts", "warn", "not configured (NOTIFY_BACKEND=off) — "
                                  "nothing will be pushed to you")
    else:
        if notifier.send("🧪 databroker self-test — if you can read this, alerts are working."):
            record("Alerts", "ok", f"{notifier.describe()} — test message sent, check your phone")
        else:
            record("Alerts", "fail", f"{notifier.describe()} — send failed (see error above)")

    # --- 7. Database read/write ---
    try:
        n = len(db.list_watchlist())
        record("Database", "ok", f"readable, {n} company/companies on the watchlist"
                                  + ("" if n else " — add one with `watch <TICKER>`"))
    except Exception as e:
        record("Database", "fail", f"{e}")

    # --- 8. Market-hours gate ---
    try:
        from .monitor import is_market_open
        open_now = is_market_open(
            tz_name=args.market_tz,
            alpaca_api_key=os.environ.get("ALPACA_API_KEY_ID"),
            alpaca_api_secret=os.environ.get("ALPACA_API_SECRET_KEY"),
        )
        holiday_aware = bool(os.environ.get("ALPACA_API_KEY_ID") and os.environ.get("ALPACA_API_SECRET_KEY"))
        record("Market-hours gate", "ok",
               f"market is {'OPEN' if open_now else 'CLOSED'} right now "
               f"({'holiday-aware calendar' if holiday_aware else 'plain weekday+hours heuristic'})")
    except Exception as e:
        record("Market-hours gate", "fail", f"{e}")

    fails = [r for r in results if r[1] == "fail"]
    warns = [r for r in results if r[1] == "warn"]
    print(f"\n{'-' * 60}")
    print(f"{len(results) - len(fails) - len(warns)} ok, {len(warns)} warning(s), {len(fails)} failure(s)")
    if fails:
        print("\nFailures to fix before relying on this:")
        for name, _, msg in fails:
            print(f"  • {name}: {msg}")
    elif warns:
        print("\nNo hard failures. Warnings are usually just unconfigured optional pieces — "
              "check the list above and confirm each one is off on purpose.")
    else:
        print("\nEverything configured is working.")



def cmd_sweep(db: DB, args):
    from .monitor import sweep_once
    from .notify import build_notifier_from_env

    watchlist = db.list_watchlist()
    if not watchlist:
        print("Watchlist is empty — nothing to sweep. Use `watch <TICKER>` first.")
        return
    notifier = build_notifier_from_env()
    # Alerts now fire inline, per company, as investigate() finds things (see
    # agent.py: a quick ping per notable event, then the fuller analysis once
    # that company's report is ready) — no separate end-of-sweep digest push,
    # so nothing gets sent twice.
    agent = build_agent(db, notifier=notifier)
    summary = sweep_once(db, agent, delay_seconds=args.delay)
    for r in summary.per_company:
        status = f"error: {r.error}" if r.error else f"{r.events_found} event(s), {r.llm_calls_made} LLM call(s)"
        print(f"  {r.ticker}: {status}")
    print(f"\nTotal LLM calls this sweep: {summary.total_llm_calls}\n")
    print(summary.digest)


def cmd_watch_loop(db: DB, args):
    import os
    from .monitor import run_loop
    from .notify import build_notifier_from_env

    try:
        hour, minute = (int(x) for x in args.at.split(":"))
    except ValueError:
        print("--at must be HH:MM, e.g. 08:00")
        sys.exit(1)

    notifier = build_notifier_from_env()
    print(f"[databroker] Alerts: {notifier.describe() if notifier else 'disabled (set NOTIFY_BACKEND=telegram to enable)'}")
    agent = build_agent(db, notifier=notifier)

    def on_sweep(summary):
        print(f"\n[databroker] Sweep finished {summary.finished_at} — "
              f"{summary.total_llm_calls} LLM call(s) total.")
        print(summary.digest)

    # Market-wide discovery on its own slower schedule, if configured — see
    # run_loop's docstring for why it's decoupled from the sweep interval.
    from .discover import build_scanners_from_env
    scanners = build_scanners_from_env()
    on_discover = None
    if scanners and args.discover_every_hours:
        def on_discover():
            exclude = {(row["ticker"] or "").upper() for row in db.list_watchlist()}
            candidates = agent.discover_candidates(
                scanners, exclude=exclude, min_score=args.discover_min_score,
                limit=args.discover_limit,
            )
            if not candidates:
                print("[databroker] Discovery: no candidates cleared the threshold.")
                return
            text = format_candidates(candidates)
            print(text)
            if notifier is not None:
                notifier.send(text)
        print(f"[databroker] Market discovery: every {args.discover_every_hours:g}h via "
              + ", ".join(sc.describe() for sc in scanners))
    elif args.discover_every_hours and not scanners:
        print("[databroker] Market discovery requested but no scanners usable — "
              "set DISCOVER_BACKEND (and Alpaca credentials if using `alpaca`).")

    # Reuses the same Alpaca credentials as FINANCIAL_BACKEND=alpaca (if
    # set) to make the market-hours gate holiday-aware; harmless/no-op if
    # they're not configured — is_market_open() falls back to the plain
    # weekday+hours check without them.
    run_loop(
        db, agent, at_hour=hour, at_minute=minute, delay_seconds=args.delay, on_sweep=on_sweep,
        interval_minutes=args.interval_minutes,
        market_hours_only=not args.around_the_clock,
        market_tz=args.market_tz,
        alpaca_api_key=os.environ.get("ALPACA_API_KEY_ID"),
        alpaca_api_secret=os.environ.get("ALPACA_API_SECRET_KEY"),
        discover_every_hours=args.discover_every_hours,
        on_discover=on_discover,
    )


def main(argv=None):
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    db = DB()
    dispatch = {
        "add": cmd_add,
        "watch": cmd_watch,
        "set-thesis": cmd_set_thesis,
        "thesis-status": cmd_thesis_status,
        "research": cmd_research,
        "ask": cmd_ask,
        "digest": cmd_digest,
        "events": cmd_events,
        "graph": cmd_graph,
        "connections": cmd_connections,
        "doctor": cmd_doctor,
        "sweep": cmd_sweep,
        "discover": cmd_discover,
        "selftest": cmd_selftest,
        "watch-loop": cmd_watch_loop,
    }
    dispatch[args.command](db, args)


if __name__ == "__main__":
    main()
