"""
Market-wide discovery — surfacing tickers you AREN'T already watching.

Everything else in this tool is ticker-first: you name a company, and the
agent goes and looks at it. This module inverts that — it scans the market
for unusual activity and proposes candidates for your watchlist.

Design notes, and why it works the way it does:

* **Corroboration before the LLM.** The scanners below are cheap and
  key-free (or reuse the Alpaca key you already have). Raw scanner output is
  noisy — the top-volume list is mostly the same mega-caps every day — so a
  deterministic pass ranks candidates by how many INDEPENDENT signals flag
  the same symbol (unusual volume AND a big price move AND retail chatter is
  a much stronger signal than any one alone). Only the top few survivors are
  described to the LLM, so a market-wide scan costs one LLM call, not one
  per symbol.

* **Already-watched symbols are filtered out** before ranking. The whole
  point is to surface what you're NOT tracking.

* **These are research candidates, not recommendations.** A scanner hit
  means "something unusual is happening here", which is equally consistent
  with a great opportunity, a pump, or a company in freefall. The LLM step
  is asked to characterize WHY the symbol is showing up and what a person
  would need to check to find out — never whether to buy it.
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod
from collections import defaultdict

import requests

HEADERS = {"User-Agent": "databroker/1.0 (personal research tool)"}


class MarketScanner(ABC):
    """One market-wide source of 'something unusual here' signals.

    Returns a list of {"symbol", "signal", "detail"} dicts. Fails soft
    (returns []) on any error — a broken scanner degrades the candidate
    list, it doesn't take down the scan.
    """

    @abstractmethod
    def scan(self, max_results: int = 20) -> list[dict]:
        ...

    def describe(self) -> str:
        return type(self).__name__


class AlpacaScreenerScanner(MarketScanner):
    """Alpaca's screener endpoints — most-actives (by volume) and top
    movers (gainers/losers). Same free paper-trading credentials as the
    Alpaca news backend; no extra signup.

    Note both endpoints reflect the CURRENT session and reset at market
    open, so outside market hours they return the previous session's data.
    That's fine for discovery (yesterday's unusual activity is still worth
    a look) but it does mean an overnight scan repeats the prior day.
    """

    BASE = "https://data.alpaca.markets/v1beta1/screener/stocks"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def _headers(self) -> dict:
        return {
            **HEADERS,
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _get(self, path: str, params: dict) -> dict | None:
        try:
            resp = requests.get(f"{self.BASE}/{path}", params=params,
                                headers=self._headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def scan(self, max_results: int = 20) -> list[dict]:
        out: list[dict] = []
        per_endpoint = max(1, max_results // 2)

        actives = self._get("most-actives", {"by": "volume", "top": per_endpoint})
        if actives:
            for item in (actives.get("most_actives") or []):
                sym = (item.get("symbol") or "").upper()
                if not sym:
                    continue
                vol = item.get("volume")
                out.append({
                    "symbol": sym,
                    "signal": "unusual volume",
                    "detail": f"among the day's most-traded names"
                              + (f" ({int(vol):,} shares)" if isinstance(vol, (int, float)) else ""),
                })

        movers = self._get("movers", {"top": per_endpoint})
        if movers:
            for direction, key in (("gainer", "gainers"), ("loser", "losers")):
                for item in (movers.get(key) or []):
                    sym = (item.get("symbol") or "").upper()
                    if not sym:
                        continue
                    pct = item.get("percent_change")
                    pct_str = f" ({pct:+.1f}% on the day)" if isinstance(pct, (int, float)) else ""
                    out.append({
                        "symbol": sym,
                        "signal": f"large price move ({direction})",
                        "detail": f"among the day's top {direction}s{pct_str}",
                    })
        return out

    def describe(self) -> str:
        return "Alpaca screener (most-actives + movers)"


class StockTwitsTrendingScanner(MarketScanner):
    """StockTwits' trending-symbols endpoint — what retail traders are
    suddenly talking about. No API key. Deliberately weighted as a WEAK
    signal on its own in rank_candidates(): trending chatter is the most
    easily manufactured of these signals, and a symbol trending here with
    no corresponding volume or price move is often just noise (or a pump).
    Its value is as corroboration for the other two.
    """

    def scan(self, max_results: int = 20) -> list[dict]:
        try:
            resp = requests.get("https://api.stocktwits.com/api/2/trending/symbols.json",
                                headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []

        out = []
        for item in (data.get("symbols") or [])[:max_results]:
            sym = (item.get("symbol") or "").upper()
            if not sym:
                continue
            title = item.get("title") or ""
            out.append({
                "symbol": sym,
                "signal": "trending retail chatter",
                "detail": f"trending on StockTwits" + (f" ({title})" if title else ""),
            })
        return out

    def describe(self) -> str:
        return "StockTwits trending symbols"


class MockScanner(MarketScanner):
    def __init__(self, hits: list[dict] | None = None):
        self.hits = hits or []

    def scan(self, max_results: int = 20) -> list[dict]:
        return self.hits[:max_results]


# ---------------------------------------------------------------------------
# Deterministic ranking — runs BEFORE any LLM call, so a market-wide scan
# costs one LLM call rather than one per symbol.
# ---------------------------------------------------------------------------

# A symbol flagged by several independent signal TYPES is far more
# interesting than one flagged repeatedly by the same type. Weights reflect
# how easily each signal is manufactured or how often it's just noise.
SIGNAL_WEIGHTS = {
    "unusual volume": 2,
    "large price move (gainer)": 3,
    "large price move (loser)": 3,
    "trending retail chatter": 1,
}


def rank_candidates(raw_hits: list[dict], exclude: set[str],
                    min_score: int = 3, limit: int = 5) -> list[dict]:
    """Collapse raw scanner hits into scored candidates.

    `exclude` is the set of symbols already on the watchlist (upper-cased).
    `min_score` defaults to 3, which deliberately means a symbol flagged ONLY
    by retail chatter (weight 1) or ONLY by volume (weight 2) doesn't make
    the cut — it takes either a genuine price move or a combination of
    signals. Raise it to be stricter, lower it to see more.

    Returns candidates sorted by score, highest first.
    """
    by_symbol: dict[str, dict] = defaultdict(lambda: {"signals": {}, "score": 0})
    for hit in raw_hits:
        sym = (hit.get("symbol") or "").upper()
        if not sym or sym in exclude:
            continue
        signal = hit.get("signal") or "unknown"
        entry = by_symbol[sym]
        # Count each signal TYPE once — a symbol appearing twice in the same
        # list shouldn't double its own score.
        if signal not in entry["signals"]:
            entry["signals"][signal] = hit.get("detail") or signal
            entry["score"] += SIGNAL_WEIGHTS.get(signal, 1)

    candidates = [
        {"symbol": sym, "score": e["score"], "signals": e["signals"]}
        for sym, e in by_symbol.items() if e["score"] >= min_score
    ]
    candidates.sort(key=lambda c: (-c["score"], c["symbol"]))
    return candidates[:limit]


def build_scanners_from_env() -> list[MarketScanner]:
    """
    Env var: DISCOVER_BACKEND = off (default) | comma-separated list of:
      alpaca | stocktwits

    e.g. DISCOVER_BACKEND=alpaca,stocktwits. `alpaca` reuses
    ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY and is silently skipped if
    either is missing; `stocktwits` needs no key.
    """
    backend = os.environ.get("DISCOVER_BACKEND", "off").lower()
    if backend in ("off", ""):
        return []

    scanners: list[MarketScanner] = []
    for name in [b.strip() for b in backend.split(",") if b.strip()]:
        if name == "alpaca":
            key_id = os.environ.get("ALPACA_API_KEY_ID", "")
            secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
            if key_id and secret:
                scanners.append(AlpacaScreenerScanner(key_id, secret))
        elif name == "stocktwits":
            scanners.append(StockTwitsTrendingScanner())
        elif name == "mock":
            scanners.append(MockScanner())
    return scanners
