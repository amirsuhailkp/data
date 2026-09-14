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
from .social import build_social_from_env, build_financial_from_env
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


def build_agent(db: DB, announce: bool = True) -> ResearchAgent:
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
        help="Run forever, sweeping the watchlist once a day at a fixed local time "
             "(alternative to cron/Task Scheduler for a long-running process)",
    )
    watch_loop.add_argument("--at", default="08:00", help="Local time to sweep, HH:MM (default 08:00)")
    watch_loop.add_argument("--delay", type=float, default=3.0,
                             help="Seconds to pause between companies during each sweep")

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
    from .llm import OllamaProvider

    print("LLM_BACKEND     =", os.environ.get("LLM_BACKEND", "auto (not set)"))
    print("Ollama running? =", OllamaProvider.is_available())
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
    print("FINANCIAL_BACKEND =", os.environ.get("FINANCIAL_BACKEND", "off (default)"))
    print("FETCHER_BACKEND =", os.environ.get("FETCHER_BACKEND", "static (default)"))
    if os.environ.get("FETCHER_BACKEND", "static").lower() == "browser":
        try:
            import playwright  # noqa: F401
            print("Playwright       = installed")
        except ImportError:
            print("Playwright       = NOT installed — will silently fall back to static fetch. "
                  "Run: pip install playwright && playwright install chromium")
    provider = build_provider_from_env()
    print("\n-> Resolved LLM provider:", type(provider).__name__)


def cmd_sweep(db: DB, args):
    from .monitor import sweep_once

    watchlist = db.list_watchlist()
    if not watchlist:
        print("Watchlist is empty — nothing to sweep. Use `watch <TICKER>` first.")
        return
    agent = build_agent(db)
    summary = sweep_once(db, agent, delay_seconds=args.delay)
    for r in summary.per_company:
        status = f"error: {r.error}" if r.error else f"{r.events_found} event(s), {r.llm_calls_made} LLM call(s)"
        print(f"  {r.ticker}: {status}")
    print(f"\nTotal LLM calls this sweep: {summary.total_llm_calls}\n")
    print(summary.digest)


def cmd_watch_loop(db: DB, args):
    from .monitor import run_loop

    try:
        hour, minute = (int(x) for x in args.at.split(":"))
    except ValueError:
        print("--at must be HH:MM, e.g. 08:00")
        sys.exit(1)

    agent = build_agent(db)

    def on_sweep(summary):
        print(f"\n[databroker] Sweep finished {summary.finished_at} — "
              f"{summary.total_llm_calls} LLM call(s) total.")
        print(summary.digest)

    run_loop(db, agent, at_hour=hour, at_minute=minute, delay_seconds=args.delay, on_sweep=on_sweep)


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
        "watch-loop": cmd_watch_loop,
    }
    dispatch[args.command](db, args)


if __name__ == "__main__":
    main()
