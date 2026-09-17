"""
Price history and fundamentals — computed context, not predictions.

Two things live here, both DETERMINISTIC (plain arithmetic, no LLM):
  1. PriceHistoryProvider: pulls daily OHLCV bars and computes a technical
     snapshot (moving averages, RSI, recent volatility, 52-week range,
     volume trend).
  2. FundamentalsProvider: pulls basic company financials (P/E, margins,
     growth, 52-week range) as reported.

Why this is computed here rather than left to the LLM: an LLM asked to
"calculate the RSI" from a table of prices is unreliable — it will produce
plausible-looking wrong numbers. Arithmetic belongs in code. What the LLM
IS asked to do (see agent.py) is describe what a computed snapshot suggests
in plain language and flag it as something to watch — never to predict
where the price goes next, never to compute or reproduce the numbers
itself, and never to issue a buy/sell call. See agent.py's build_report for
exactly how this snapshot is framed to the model.
"""

from __future__ import annotations
import os
import statistics
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import requests

HEADERS = {"User-Agent": "databroker/1.0 (personal research tool)"}


class PriceHistoryProvider(ABC):
    @abstractmethod
    def get_technical_snapshot(self, ticker: str) -> dict | None:
        """Returns a computed technical snapshot dict, or None if
        unavailable (no data, bad credentials, network error — always fails
        soft, never raises)."""


def _sma(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _rsi(closes: list[float], period: int = 14) -> float | None:
    """Standard Wilder RSI over the last `period` daily changes."""
    if len(closes) < period + 1:
        return None
    changes = [closes[i] - closes[i - 1] for i in range(len(closes) - period, len(closes))]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)


class AlpacaPriceHistoryProvider(PriceHistoryProvider):
    """Alpaca's free stock bars endpoint (IEX feed on the free plan — same
    credentials as the Alpaca news backend, no extra signup). Pulls ~4
    months of daily bars, enough for a 50-day SMA and a meaningful
    52-week-range APPROXIMATION (see the caveat in the snapshot output —
    this is not a true 52-week high/low without a full year of bars, which
    the free feed's history depth may not support; FundamentalsProvider's
    52-week figures, when available, are more reliable for that specific
    number)."""

    BASE = "https://data.alpaca.markets/v2/stocks"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def get_technical_snapshot(self, ticker: str) -> dict | None:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=130)  # ~90 trading days, enough for a 50-day SMA
        try:
            resp = requests.get(
                f"{self.BASE}/{ticker.upper()}/bars",
                params={
                    "timeframe": "1Day",
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "limit": 200,
                    "adjustment": "split",
                    "feed": "iex",
                },
                headers={**HEADERS, "APCA-API-KEY-ID": self.api_key,
                        "APCA-API-SECRET-KEY": self.api_secret},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return None

        bars = data.get("bars") if isinstance(data, dict) else None
        if not bars or len(bars) < 15:  # need at least enough for a 14-day RSI
            return None

        closes = [b["c"] for b in bars]
        volumes = [b["v"] for b in bars]
        latest = bars[-1]

        sma20 = _sma(closes, 20)
        sma50 = _sma(closes, 50)
        rsi14 = _rsi(closes, 14)
        avg_vol20 = _sma(volumes, 20) or 0
        recent_high = max(b["h"] for b in bars)
        recent_low = min(b["l"] for b in bars)
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes)) if closes[i - 1]
        ]
        volatility = round(statistics.pstdev(returns) * 100, 2) if len(returns) >= 5 else None

        pct_1d = round((closes[-1] - closes[-2]) / closes[-2] * 100, 2) if len(closes) >= 2 else None
        pct_5d = round((closes[-1] - closes[-6]) / closes[-6] * 100, 2) if len(closes) >= 6 else None
        pct_30d = round((closes[-1] - closes[-min(21, len(closes))]) / closes[-min(21, len(closes))] * 100, 2) \
            if len(closes) >= 2 else None

        return {
            "as_of": latest["t"],
            "bars_used": len(bars),
            "last_close": latest["c"],
            "sma_20": round(sma20, 2) if sma20 else None,
            "sma_50": round(sma50, 2) if sma50 else None,
            "trend_vs_sma": (
                "above both 20d and 50d averages" if sma20 and sma50 and closes[-1] > sma20 > sma50 else
                "below both 20d and 50d averages" if sma20 and sma50 and closes[-1] < sma20 < sma50 else
                "mixed relative to its moving averages" if sma20 and sma50 else None
            ),
            "rsi_14": rsi14,
            "rsi_note": (
                "conventionally read as overbought (>70)" if rsi14 and rsi14 > 70 else
                "conventionally read as oversold (<30)" if rsi14 and rsi14 < 30 else
                "in a neutral range" if rsi14 else None
            ),
            "pct_change_1d": pct_1d,
            "pct_change_5d": pct_5d,
            "pct_change_30d": pct_30d,
            "period_high": round(recent_high, 2),
            "period_low": round(recent_low, 2),
            "period_covered_days": len(bars),
            "avg_volume_20d": int(avg_vol20),
            "latest_volume": latest["v"],
            "volume_vs_20d_avg": (
                round(latest["v"] / avg_vol20, 2) if avg_vol20 else None
            ),
            "_caveat": "period_high/low cover only the bars fetched (~90 trading days), "
                       "not a true 52-week range.",
        }

    def describe(self) -> str:
        return "Alpaca daily bars (technical snapshot)"


class MockPriceHistoryProvider(PriceHistoryProvider):
    def get_technical_snapshot(self, ticker: str) -> dict | None:
        return None


class FundamentalsProvider(ABC):
    @abstractmethod
    def get_fundamentals(self, ticker: str) -> dict | None:
        ...


class FinnhubFundamentalsProvider(FundamentalsProvider):
    """Finnhub's free basic-financials endpoint — reuses FINNHUB_API_KEY,
    the same key as the Finnhub news backend. A genuinely free, official
    52-week high/low and standard valuation/margin metrics, as reported."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_fundamentals(self, ticker: str) -> dict | None:
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/stock/metric",
                params={"symbol": ticker.upper(), "metric": "all", "token": self.api_key},
                headers=HEADERS, timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return None

        metric = data.get("metric") if isinstance(data, dict) else None
        if not metric:
            return None

        def g(*keys):
            for k in keys:
                if metric.get(k) is not None:
                    return metric[k]
            return None

        out = {
            "pe_ttm": g("peBasicExclExtraTTM", "peExclExtraTTM"),
            "eps_growth_ttm_yoy_pct": g("epsGrowthTTMYoy"),
            "revenue_growth_ttm_yoy_pct": g("revenueGrowthTTMYoy"),
            "gross_margin_ttm_pct": g("grossMarginTTM"),
            "operating_margin_ttm_pct": g("operatingMarginTTM"),
            "debt_to_equity": g("totalDebt/totalEquityQuarterly", "totalDebt/totalEquityAnnual"),
            "52_week_high": g("52WeekHigh"),
            "52_week_low": g("52WeekLow"),
            "beta": g("beta"),
        }
        # Drop keys Finnhub didn't have data for, rather than showing the LLM
        # a wall of Nones that invites it to guess at what's missing.
        return {k: v for k, v in out.items() if v is not None} or None

    def describe(self) -> str:
        return "Finnhub basic financials"


class MockFundamentalsProvider(FundamentalsProvider):
    def get_fundamentals(self, ticker: str) -> dict | None:
        return None


def build_price_provider_from_env() -> PriceHistoryProvider | None:
    """
    Env var: PRICE_DATA_BACKEND = off (default) | alpaca | mock
    `alpaca` reuses ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY; skipped if
    either is missing.
    """
    backend = os.environ.get("PRICE_DATA_BACKEND", "off").lower()
    if backend == "alpaca":
        key_id = os.environ.get("ALPACA_API_KEY_ID", "")
        secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
        if key_id and secret:
            return AlpacaPriceHistoryProvider(key_id, secret)
        return None
    if backend == "mock":
        return MockPriceHistoryProvider()
    return None


def build_fundamentals_provider_from_env() -> FundamentalsProvider | None:
    """
    Env var: FUNDAMENTALS_BACKEND = off (default) | finnhub | mock
    `finnhub` reuses FINNHUB_API_KEY; skipped if missing.
    """
    backend = os.environ.get("FUNDAMENTALS_BACKEND", "off").lower()
    if backend == "finnhub":
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        if api_key:
            return FinnhubFundamentalsProvider(api_key)
        return None
    if backend == "mock":
        return MockFundamentalsProvider()
    return None
