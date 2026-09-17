"""
Price/volume chart data for watchlisted companies — deterministic technical
indicators computed from real historical bars, given to the LLM as CONTEXT
for its report, never as something for the LLM itself to calculate.

Why this is a separate, non-LLM step: SMA/RSI/volatility/support-resistance
are arithmetic, not judgment. Computing them here means the LLM never has to
"do math" it's unreliable at, and it costs zero extra LLM calls — only the
finished text summary is added to the existing report prompt (see agent.py's
build_report()), not one call per indicator or per ticker.

The hard boundary, same as everywhere else in this tool: these describe past
price action, not a forecast. The formatted snapshot below states facts
("price is 3% below its 20-day average") and the report prompt explicitly
tells the LLM to use those facts for framing only — never to turn a pattern
into a predicted price move, a buy/sell call, or a price target.
"""

from __future__ import annotations
import os
import statistics
from abc import ABC, abstractmethod
from datetime import date, timedelta

import requests

HEADERS = {"User-Agent": "databroker/1.0 (personal research tool)"}


class TechnicalsProvider(ABC):
    @abstractmethod
    def get_snapshot(self, ticker: str) -> str | None:
        """A short, deterministic, human-readable technical summary for
        `ticker`, or None if data isn't available (fails soft — a broken
        or unconfigured technicals source should never block a report)."""

    def describe(self) -> str:
        return type(self).__name__


def compute_indicators(bars: list[dict]) -> dict | None:
    """Pure function: bars -> indicators. Kept separate from the HTTP layer
    so it's cheaply testable against known values without any network mock.

    `bars` is a list of {"t","o","h","l","c","v"} dicts, OLDEST FIRST (Alpaca's
    native order). Returns None if there isn't even enough data for the
    shortest indicator (20-day SMA needs 20 closes).

    Indicators that need more history than is available (e.g. a 50-day SMA
    with only 30 bars) are simply omitted from the result rather than
    computed on a too-small window that would be misleading.
    """
    if len(bars) < 20:
        return None

    closes = [b["c"] for b in bars]
    volumes = [b["v"] for b in bars]
    latest = bars[-1]
    prior_close = bars[-2]["c"] if len(bars) >= 2 else None

    out: dict = {
        "as_of": latest["t"][:10],
        "close": latest["c"],
        "change_pct": ((latest["c"] - prior_close) / prior_close * 100) if prior_close else None,
        "sma20": statistics.fmean(closes[-20:]),
    }
    if len(closes) >= 50:
        out["sma50"] = statistics.fmean(closes[-50:])

    rsi = _rsi(closes, period=14)
    if rsi is not None:
        out["rsi14"] = rsi

    # Annualized-ish daily volatility over the trailing window we have,
    # capped at 60 days so a very long history doesn't dilute "recent".
    window = closes[-60:] if len(closes) > 60 else closes
    if len(window) >= 2:
        returns = [(window[i] - window[i - 1]) / window[i - 1] for i in range(1, len(window))]
        if len(returns) >= 2:
            out["volatility_pct"] = statistics.pstdev(returns) * 100

    recent_vol_window = volumes[-20:]
    out["volume"] = latest["v"]
    out["avg_volume_20d"] = statistics.fmean(recent_vol_window)
    out["volume_vs_avg_pct"] = (
        (latest["v"] - out["avg_volume_20d"]) / out["avg_volume_20d"] * 100
        if out["avg_volume_20d"] else None
    )

    window_for_range = bars[-60:] if len(bars) > 60 else bars
    out["range_high"] = max(b["h"] for b in window_for_range)
    out["range_low"] = min(b["l"] for b in window_for_range)
    out["range_days"] = len(window_for_range)
    return out


def _rsi(closes: list[float], period: int = 14) -> float | None:
    """Standard Wilder RSI. Needs at least period+1 closes."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    avg_gain = statistics.fmean(gains[:period])
    avg_loss = statistics.fmean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def format_snapshot(ticker: str, ind: dict) -> str:
    """Deterministic text — same numbers always produce the same wording,
    so this is reproducible and auditable, not the LLM paraphrasing data."""
    lines = [f"Technical snapshot for {ticker} (as of {ind['as_of']}, close ${ind['close']:.2f}):"]
    if ind.get("change_pct") is not None:
        lines.append(f"- Change vs prior close: {ind['change_pct']:+.1f}%")
    lines.append(f"- 20-day SMA: ${ind['sma20']:.2f} "
                 f"(price is {'above' if ind['close'] >= ind['sma20'] else 'below'})")
    if "sma50" in ind:
        lines.append(f"- 50-day SMA: ${ind['sma50']:.2f} "
                     f"(price is {'above' if ind['close'] >= ind['sma50'] else 'below'})")
    if "rsi14" in ind:
        note = ""
        if ind["rsi14"] >= 70:
            note = " (conventionally read as overbought territory)"
        elif ind["rsi14"] <= 30:
            note = " (conventionally read as oversold territory)"
        lines.append(f"- RSI(14): {ind['rsi14']:.1f}{note}")
    if "volatility_pct" in ind:
        lines.append(f"- Recent daily volatility (stdev of returns, "
                     f"trailing window): {ind['volatility_pct']:.1f}%")
    if ind.get("volume_vs_avg_pct") is not None:
        lines.append(f"- Volume: {ind['volume']:,.0f} vs {ind['avg_volume_20d']:,.0f} 20-day "
                     f"average ({ind['volume_vs_avg_pct']:+.0f}%)")
    lines.append(f"- {ind['range_days']}-day range: ${ind['range_low']:.2f} - ${ind['range_high']:.2f} "
                 f"(price is at {(ind['close'] - ind['range_low']) / (ind['range_high'] - ind['range_low']) * 100:.0f}% "
                 f"of that range)" if ind["range_high"] > ind["range_low"] else "")
    return "\n".join(l for l in lines if l)


class AlpacaBarsProvider(TechnicalsProvider):
    """Alpaca's free IEX-feed historical bars — same credentials as the
    Alpaca news backend, no extra signup. IEX covers ~2.5% of US equity
    volume rather than the full consolidated tape, which is fine for the
    indicators here (they're about shape/trend, not exact print-for-print
    volume) but worth knowing if the numbers look slightly off vs. a
    broker's SIP-fed chart."""

    BASE = "https://data.alpaca.markets/v2/stocks"

    def __init__(self, api_key: str, api_secret: str, lookback_days: int = 90):
        self.api_key = api_key
        self.api_secret = api_secret
        self.lookback_days = lookback_days

    def get_snapshot(self, ticker: str) -> str | None:
        bars = self._get_bars(ticker)
        if not bars:
            return None
        ind = compute_indicators(bars)
        if ind is None:
            return None
        return format_snapshot(ticker.upper(), ind)

    def _get_bars(self, ticker: str) -> list[dict]:
        start = (date.today() - timedelta(days=self.lookback_days)).isoformat()
        try:
            resp = requests.get(
                f"{self.BASE}/{ticker.upper()}/bars",
                params={"feed": "iex", "timeframe": "1Day", "start": start, "limit": 1000},
                headers={**HEADERS, "APCA-API-KEY-ID": self.api_key,
                        "APCA-API-SECRET-KEY": self.api_secret},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []
        bars = data.get("bars") if isinstance(data, dict) else None
        if not isinstance(bars, list):
            return []
        return bars

    def describe(self) -> str:
        return "Alpaca historical bars (IEX feed)"


class MockTechnicalsProvider(TechnicalsProvider):
    def __init__(self, snapshot: str | None = "Technical snapshot for TEST: mock data"):
        self.snapshot = snapshot

    def get_snapshot(self, ticker: str) -> str | None:
        return self.snapshot


def build_technicals_from_env() -> TechnicalsProvider | None:
    """
    Env var: TECHNICALS_BACKEND = off (default) | alpaca

    `alpaca` reuses ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY and is
    silently skipped if either is missing — `doctor` flags that.
    """
    backend = os.environ.get("TECHNICALS_BACKEND", "off").lower()
    if backend in ("off", ""):
        return None
    if backend == "alpaca":
        key_id = os.environ.get("ALPACA_API_KEY_ID", "")
        secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
        if key_id and secret:
            return AlpacaBarsProvider(key_id, secret)
        return None
    return None
