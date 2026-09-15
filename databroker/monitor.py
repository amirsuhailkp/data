"""
Phase 2: scheduled / background monitoring.

Three ways to run this, all free and dependency-free beyond `tzdata` on
platforms that need it for named timezones (stdlib `zoneinfo` otherwise):

  1. One-shot sweep (recommended for swing/longer-horizon tracking) — call
     `sweep_once()` from an OS scheduler:
       Linux/Mac cron:    0 8 * * *  cd /path/to/databroker && python -m databroker.cli sweep
       Windows:            Task Scheduler -> Action: python -m databroker.cli sweep,
                            Start in: the project folder
     Simplest option: no process to keep alive, no extra dependency, and it
     fits naturally with how free-tier API rate limits reset daily.

  2. Daily long-running loop — `run_loop()` / `python -m databroker.cli
     watch-loop` (no `--interval-minutes`) for anyone who'd rather leave a
     terminal/process running than configure an OS scheduler. Sweeps once a
     day at a fixed local time.

  3. Fast interval loop — `run_loop(..., interval_minutes=N)` / `python -m
     databroker.cli watch-loop --interval-minutes N` for day trading, where
     once-a-day is too slow. Sweeps every N minutes, but ONLY while the
     market session is open (`market_hours_only=True`, the default) — this
     matters because free-tier LLM/search/financial quotas are precious and
     a day-trading thesis genuinely doesn't need checking at 2am when
     nothing is trading. Session bounds default to US equities regular
     hours (9:30-16:00 America/New_York) and are configurable for other
     markets. Set `market_hours_only=False` to poll around the clock anyway
     (e.g. for crypto tickers, which trade 24/7).

     If ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY are set (same credentials
     as FINANCIAL_BACKEND=alpaca, reused here for a different purpose),
     the market-hours check upgrades from a plain weekday+fixed-hours
     guess to Alpaca's real trading calendar — correctly skips actual
     market holidays and uses the real close time on early-close days
     (e.g. the day after Thanksgiving). Falls back to the plain heuristic
     automatically if the calendar lookup fails or credentials aren't set.

Either way, each company on the watchlist gets investigated with a generic
"what's new" objective, then the digest is generated from whatever the sweep
found (plus anything from earlier `research`/`ask` calls that hasn't been
surfaced yet).
"""

from __future__ import annotations
import datetime
import time
from dataclasses import dataclass, field

import requests

from .db import DB
from .agent import ResearchAgent

DEFAULT_MONITORING_OBJECTIVE = "What are the latest material developments for this company?"

# US equities regular session, used as the default market-hours window for
# fast interval polling. Overridable per-call for other markets/timezones.
DEFAULT_MARKET_TZ = "America/New_York"
DEFAULT_MARKET_OPEN = (9, 30)
DEFAULT_MARKET_CLOSE = (16, 0)


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
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
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
    finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return SweepSummary(started_at=started, finished_at=finished, per_company=per_company, digest=digest)


def _seconds_until(hour: int, minute: int = 0) -> float:
    now = datetime.datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += datetime.timedelta(days=1)
    return (target - now).total_seconds()


def is_market_open(
    now: datetime.datetime | None = None,
    tz_name: str = DEFAULT_MARKET_TZ,
    open_time: tuple[int, int] = DEFAULT_MARKET_OPEN,
    close_time: tuple[int, int] = DEFAULT_MARKET_CLOSE,
    alpaca_api_key: str | None = None,
    alpaca_api_secret: str | None = None,
) -> bool:
    """Weekday + time-of-day check against a market session window, upgraded
    to a real holiday-aware check when Alpaca credentials are supplied.

    Without credentials: a plain weekday + fixed-hours check. This can
    occasionally return True on e.g. Christmas that falls on a weekday —
    acceptable for a polling gate on its own, since worst case is one wasted
    sweep, not a missed one.

    With `alpaca_api_key`/`alpaca_api_secret` set: consults Alpaca's free
    `/v2/calendar` endpoint (same credentials already needed for
    `FINANCIAL_BACKEND=alpaca`) for the actual trading calendar that date —
    correctly returns False on market holidays, and uses that day's real
    open/close times rather than the fixed default (covers early-close
    days like the day after Thanksgiving). Falls back to the plain
    weekday+hours check if the calendar lookup fails for any reason (no
    network, bad credentials, unexpected response shape) — a broken
    calendar call should never crash or block polling, just degrade to the
    simpler heuristic for that check.
    """
    from zoneinfo import ZoneInfo

    now = (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(ZoneInfo(tz_name))

    if alpaca_api_key and alpaca_api_secret:
        calendar_result = _fetch_market_calendar_day(now.date(), alpaca_api_key, alpaca_api_secret)
        if calendar_result is not _CALENDAR_FETCH_FAILED:
            if calendar_result is None:
                return False  # confirmed non-trading day (holiday or weekend) per Alpaca's calendar
            day_open, day_close = calendar_result
            open_dt = now.replace(hour=day_open.hour, minute=day_open.minute, second=0, microsecond=0)
            close_dt = now.replace(hour=day_close.hour, minute=day_close.minute, second=0, microsecond=0)
            return open_dt <= now < close_dt
        # calendar lookup failed -> fall through to the plain heuristic below

    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    open_dt = now.replace(hour=open_time[0], minute=open_time[1], second=0, microsecond=0)
    close_dt = now.replace(hour=close_time[0], minute=close_time[1], second=0, microsecond=0)
    return open_dt <= now < close_dt


# Sentinel distinct from None: None means "confirmed non-trading day", this
# means "we couldn't find out" (network/auth/parsing failure) — the caller
# needs to tell those apart to know whether to trust the result or fall back.
_CALENDAR_FETCH_FAILED = object()

# Per-date cache — a given calendar date's trading hours don't change during
# a running process, so there's no reason to re-fetch on every polling tick.
_MARKET_CALENDAR_CACHE: dict[str, tuple[datetime.time, datetime.time] | None] = {}


def _fetch_market_calendar_day(date: datetime.date, api_key: str, api_secret: str):
    """Returns (open_time, close_time) as datetime.time objects in the
    market's local time if `date` is a trading day, None if it's a
    confirmed holiday/weekend per Alpaca's calendar, or the
    `_CALENDAR_FETCH_FAILED` sentinel if the lookup itself couldn't be
    completed (caller should fall back to the simple heuristic in that
    case, not treat it as either of the other two outcomes)."""
    cache_key = date.isoformat()
    if cache_key in _MARKET_CALENDAR_CACHE:
        return _MARKET_CALENDAR_CACHE[cache_key]

    try:
        resp = requests.get(
            "https://paper-api.alpaca.markets/v2/calendar",
            params={"start": cache_key, "end": cache_key},
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return _CALENDAR_FETCH_FAILED
    if not isinstance(data, list):
        return _CALENDAR_FETCH_FAILED  # bad/expired credentials come back as an error object

    if not data:
        _MARKET_CALENDAR_CACHE[cache_key] = None  # empty list = confirmed non-trading day
        return None

    try:
        entry = data[0]
        open_t = datetime.datetime.strptime(entry["open"], "%H:%M").time()
        close_t = datetime.datetime.strptime(entry["close"], "%H:%M").time()
    except (KeyError, ValueError, TypeError, IndexError):
        return _CALENDAR_FETCH_FAILED

    result = (open_t, close_t)
    _MARKET_CALENDAR_CACHE[cache_key] = result
    return result


def run_loop(db: DB, agent: ResearchAgent, at_hour: int = 8, at_minute: int = 0,
             delay_seconds: float = 3.0, on_sweep=None,
             interval_minutes: float | None = None, market_hours_only: bool = True,
             market_tz: str = DEFAULT_MARKET_TZ,
             market_open: tuple[int, int] = DEFAULT_MARKET_OPEN,
             market_close: tuple[int, int] = DEFAULT_MARKET_CLOSE,
             alpaca_api_key: str | None = None, alpaca_api_secret: str | None = None,
             _sleep=time.sleep):
    """Runs forever. Two modes:

    - `interval_minutes=None` (default): daily mode, one sweep at
      `at_hour:at_minute` local time — unchanged behavior from before.
    - `interval_minutes=N`: fast mode for day trading. Sweeps every N
      minutes. If `market_hours_only` is True (default), a closed-market
      tick is skipped without spending any LLM/search/financial calls — it
      just sleeps one more interval and checks again, so quota isn't burned
      overnight or on weekends. Pass `market_hours_only=False` to poll
      around the clock regardless (e.g. crypto tickers).

    `alpaca_api_key`/`alpaca_api_secret`, if given, upgrade the market-hours
    gate from a plain weekday+hours check to a real holiday-aware calendar
    lookup — see `is_market_open()`. Optional; the gate still works without
    them, just without holiday awareness.

    `on_sweep`, if given, is called with each SweepSummary (e.g. to print it
    or write it somewhere) — this function itself only prints loop-level
    status, so callers can stay fully headless via `on_sweep`.

    `_sleep` is swappable for tests so this can be exercised without
    actually waiting in wall-clock time.
    """
    if interval_minutes is not None:
        print(f"[databroker] Monitoring loop started. Fast polling every {interval_minutes:g} "
              f"minute(s){' (market hours only)' if market_hours_only else ' (24/7)'}. "
              "Press Ctrl+C to stop.")
        try:
            while True:
                if market_hours_only and not is_market_open(
                    tz_name=market_tz, open_time=market_open, close_time=market_close,
                    alpaca_api_key=alpaca_api_key, alpaca_api_secret=alpaca_api_secret,
                ):
                    print("[databroker] Market closed — skipping this tick, no API calls made.")
                else:
                    summary = sweep_once(db, agent, delay_seconds=delay_seconds)
                    if on_sweep:
                        on_sweep(summary)
                _sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n[databroker] Monitoring loop stopped.")
        return

    print(f"[databroker] Monitoring loop started. Daily sweep at {at_hour:02d}:{at_minute:02d} local time. "
          "Press Ctrl+C to stop.")
    try:
        while True:
            wait = _seconds_until(at_hour, at_minute)
            print(f"[databroker] Next sweep in {wait / 3600:.1f} hours.")
            _sleep(wait)
            summary = sweep_once(db, agent, delay_seconds=delay_seconds)
            if on_sweep:
                on_sweep(summary)
    except KeyboardInterrupt:
        print("\n[databroker] Monitoring loop stopped.")
