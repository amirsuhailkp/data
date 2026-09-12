"""
Phase 2: scheduled / background monitoring.

Two ways to run this, both free and dependency-free (stdlib only):

  1. One-shot sweep (recommended) — call `sweep_once()` from an OS scheduler:
       Linux/Mac cron:    0 8 * * *  cd /path/to/databroker && python -m databroker.cli sweep
       Windows:            Task Scheduler -> Action: python -m databroker.cli sweep,
                            Start in: the project folder
     This is the simplest option: no process to keep alive, no extra
     dependency, and it fits naturally with how free-tier API rate limits
     reset daily.

  2. Long-running loop — `run_loop()` / `python -m databroker.cli watch-loop`
     for anyone who'd rather leave a terminal/process running (e.g. on a
     home server) than configure an OS scheduler. Uses stdlib `time.sleep`
     computed against wall-clock time, not a busy loop.

Either way, each company on the watchlist gets investigated with a generic
"what's new" objective, then the digest is generated from whatever the sweep
found (plus anything from earlier `research`/`ask` calls that hasn't been
surfaced yet).
"""

from __future__ import annotations
import datetime
import time
from dataclasses import dataclass, field

from .db import DB
from .agent import ResearchAgent

DEFAULT_MONITORING_OBJECTIVE = "What are the latest material developments for this company?"


@dataclass
class SweepResult:
    ticker: str
    llm_calls_made: int
    events_found: int
    error: str | None = None


@dataclass
class SweepSummary:
    started_at: str
    finished_at: str
    per_company: list[SweepResult] = field(default_factory=list)
    digest: str = ""

    @property
    def total_llm_calls(self) -> int:
        return sum(r.llm_calls_made for r in self.per_company)


def sweep_once(db: DB, agent: ResearchAgent, delay_seconds: float = 3.0) -> SweepSummary:
    """Investigate every watchlist company once, then produce the digest.

    `delay_seconds` is a small pause between companies — cheap insurance
    against tripping a free-tier API's requests-per-minute limit on an
    account with a long watchlist. Set to 0 for local-only (Ollama) setups
    where there's no external rate limit to worry about.
    """
    started = datetime.datetime.utcnow().isoformat()
    per_company = []

    watchlist = db.list_watchlist()
    for i, row in enumerate(watchlist):
        try:
            result = agent.investigate(row["ticker"], row["name"], row["id"], DEFAULT_MONITORING_OBJECTIVE)
            db.mark_swept(row["watchlist_id"])
            per_company.append(SweepResult(
                ticker=row["ticker"], llm_calls_made=result.llm_calls_made, events_found=len(result.events),
            ))
        except Exception as e:  # a bad ticker or a transient network/API error shouldn't kill the whole sweep
            per_company.append(SweepResult(ticker=row["ticker"], llm_calls_made=0, events_found=0, error=str(e)))

        if delay_seconds and i < len(watchlist) - 1:
            time.sleep(delay_seconds)

    digest = agent.daily_digest()
    finished = datetime.datetime.utcnow().isoformat()
    return SweepSummary(started_at=started, finished_at=finished, per_company=per_company, digest=digest)


def _seconds_until(hour: int, minute: int = 0) -> float:
    now = datetime.datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += datetime.timedelta(days=1)
    return (target - now).total_seconds()


def run_loop(db: DB, agent: ResearchAgent, at_hour: int = 8, at_minute: int = 0,
             delay_seconds: float = 3.0, on_sweep=None):
    """Runs forever, sweeping once a day at the given local time. `on_sweep`,
    if given, is called with each SweepSummary (e.g. to print it or write it
    somewhere) — this function itself never prints, so it's usable headless."""
    print(f"[databroker] Monitoring loop started. Daily sweep at {at_hour:02d}:{at_minute:02d} local time. "
          "Press Ctrl+C to stop.")
    try:
        while True:
            wait = _seconds_until(at_hour, at_minute)
            print(f"[databroker] Next sweep in {wait / 3600:.1f} hours.")
            time.sleep(wait)
            summary = sweep_once(db, agent, delay_seconds=delay_seconds)
            if on_sweep:
                on_sweep(summary)
    except KeyboardInterrupt:
        print("\n[databroker] Monitoring loop stopped.")
