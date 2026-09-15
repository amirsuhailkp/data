"""
Phase 4: Social & Community Intelligence, plus real-time financial/trading
platforms treated as "first line" evidence sources.

Two different kinds of provider here, because they're queried differently:

  SocialProvider   — free-text query in, list of hits out (Reddit, Hacker
                      News). Used per sub-question, same as web search.
  FinancialProvider — ticker symbol in, list of recent activity out
                      (StockTwits). These platforms are inherently
                      ticker-scoped, not free-text searchable, and are
                      fetched ONCE per research session (not once per
                      sub-question) and merged into the highest-priority
                      sub-question's extraction as "first line" content —
                      see agent.py's investigate().

Reddit specifically now uses the official OAuth API (oauth.reddit.com) via
reddit_client.py's credential pool, which supports registering multiple free
Reddit apps to multiply past the ~100 req/min-per-app limit. Falls back to
the old unauthenticated www.reddit.com/search.json endpoint if no OAuth
credentials are configured, so this still works with zero setup — just at a
lower and less predictable rate limit.

Both social and financial content feed into the SAME extraction call
gather_evidence() already makes — no extra LLM call. Every hit is tagged
source_type="community" deterministically (overriding whatever the LLM
guesses), and the extraction prompt is told explicitly to treat this content
as sentiment/discussion, not verified fact.
"""

from __future__ import annotations
import datetime
import os
from abc import ABC, abstractmethod

import requests

HEADERS = {"User-Agent": "databroker-research-bot/0.2 (personal research use)"}


class SocialProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Return {"title","url","snippet","source_type","publication_date"} dicts."""


class FinancialProvider(ABC):
    @abstractmethod
    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        """Same dict shape as SocialProvider hits, but keyed by ticker symbol
        rather than free-text query — these platforms are inherently
        ticker-scoped (a stream of chatter about $NVDA), not searchable by
        arbitrary text."""


# ---------------------------------------------------------------------------
# Reddit — OAuth (preferred) with graceful fallback to unauthenticated search
# ---------------------------------------------------------------------------
class RedditOAuthProvider(SocialProvider):
    """Uses the credential pool in reddit_client.py — supports multiple
    registered apps pooled together to multiply the free-tier ~100 req/min
    per-app limit."""

    def __init__(self, pool, subreddits: list[str] | None = None):
        self.pool = pool
        self.subreddits = subreddits

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        q = query
        if self.subreddits:
            scope = " OR ".join(f"subreddit:{s.strip()}" for s in self.subreddits if s.strip())
            if scope:
                q = f"({query}) ({scope})"

        data = self.pool.request("/search", {"q": q, "sort": "relevance", "limit": max_results, "type": "link"})
        if not data:
            return []  # every credential exhausted, or the request failed — skip this round, don't crash
        return _parse_reddit_listing(data)


class RedditPublicSearchProvider(SocialProvider):
    """Fallback used when no OAuth credentials are configured — Reddit's
    unauthenticated public search JSON endpoint. Works with zero setup, but
    has its own (undocumented, generally stricter) rate limit and is more
    prone to layout/behavior changes than the official API."""

    def __init__(self, subreddits: list[str] | None = None):
        self.subreddits = subreddits

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        q = query
        if self.subreddits:
            scope = " OR ".join(f"subreddit:{s.strip()}" for s in self.subreddits if s.strip())
            if scope:
                q = f"({query}) ({scope})"
        try:
            resp = requests.get(
                "https://www.reddit.com/search.json",
                params={"q": q, "sort": "relevance", "limit": max_results, "type": "link"},
                headers=HEADERS, timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []
        return _parse_reddit_listing(data)


def _parse_reddit_listing(data: dict) -> list[dict]:
    """Shared parsing — both OAuth and public-search responses have the same
    Reddit "Listing" JSON shape."""
    results = []
    for child in data.get("data", {}).get("children", []):
        d = child.get("data", {})
        created = d.get("created_utc")
        pub_date = datetime.datetime.utcfromtimestamp(created).strftime("%Y-%m-%d") if created else None
        snippet = (d.get("selftext") or "").strip()[:800] or d.get("title", "")
        results.append({
            "title": f"r/{d.get('subreddit', '')}: {d.get('title', '')}",
            "url": "https://www.reddit.com" + d.get("permalink", ""),
            "snippet": snippet,
            "source_type": "community",
            "publication_date": pub_date,
        })
    return results


# ---------------------------------------------------------------------------
# Hacker News — official Algolia Search API, free, no key
# ---------------------------------------------------------------------------
class HackerNewsSocialProvider(SocialProvider):
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            resp = requests.get(
                "http://hn.algolia.com/api/v1/search",
                params={"query": query, "tags": "story", "hitsPerPage": max_results},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []

        results = []
        for hit in data.get("hits", []):
            pub_date = (hit.get("created_at") or "")[:10] or None
            object_id = hit.get("objectID")
            results.append({
                "title": hit.get("title") or "",
                "url": hit.get("url") or (f"https://news.ycombinator.com/item?id={object_id}" if object_id else ""),
                "snippet": hit.get("story_text") or hit.get("title") or "",
                "source_type": "community",
                "publication_date": pub_date,
            })
        return [r for r in results if r["url"]]


class CombinedSocialProvider(SocialProvider):
    def __init__(self, providers: list[SocialProvider]):
        self.providers = providers

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        out = []
        per_provider = max(1, max_results // max(1, len(self.providers)))
        for p in self.providers:
            out.extend(p.search(query, max_results=per_provider))
        return out


class MockSocialProvider(SocialProvider):
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        return []


# ---------------------------------------------------------------------------
# Financial/trading platforms — ticker-keyed, treated as "first line" (see
# agent.py: fetched once per research session, merged into the top-priority
# sub-question rather than repeated across every sub-question).
# ---------------------------------------------------------------------------
class StockTwitsProvider(FinancialProvider):
    """StockTwits' public symbol-stream endpoint — trader chatter and
    self-reported bullish/bearish sentiment for a specific ticker. No API
    key historically required for basic read access; StockTwits has tightened
    API terms before and may do so again, so this fails soft (returns [])
    rather than raising if the endpoint starts rejecting requests."""

    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        try:
            resp = requests.get(
                f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
                headers=HEADERS, timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []

        results = []
        for msg in (data.get("messages") or [])[:max_results]:
            body = (msg.get("body") or "").strip()
            if not body:
                continue
            pub_date = (msg.get("created_at") or "")[:10] or None
            sentiment = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic")
            username = (msg.get("user") or {}).get("username", "user")
            msg_id = msg.get("id")
            title = f"StockTwits @{username}" + (f" [{sentiment}]" if sentiment else "")
            results.append({
                "title": title,
                "url": f"https://stocktwits.com/symbol/{ticker}/message/{msg_id}" if msg_id else "https://stocktwits.com",
                "snippet": body,
                "source_type": "community",
                "publication_date": pub_date,
            })
        return results


class MockFinancialProvider(FinancialProvider):
    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        return []


# ---------------------------------------------------------------------------
# Finnhub — company news, free tier (~60 calls/min with a free API key from
# finnhub.io). This is a dedicated financial-news wire, distinct from the
# general web search results — genuinely "first line" in the sense the user
# meant (real-time, trade-related), not just news that happens to rank well
# on a search engine.
# ---------------------------------------------------------------------------
class FinnhubNewsProvider(FinancialProvider):
    """Fails soft (returns []) on a missing/invalid key, a rate-limit hit, or
    any network error — a broken financial-news backend should never take
    down a whole research run."""

    def __init__(self, api_key: str, lookback_days: int = 14):
        self.api_key = api_key
        self.lookback_days = lookback_days

    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        today = datetime.date.today()
        start = today - datetime.timedelta(days=self.lookback_days)
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": ticker,
                    "from": start.isoformat(),
                    "to": today.isoformat(),
                    "token": self.api_key,
                },
                headers=HEADERS, timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []
        if not isinstance(data, list):
            return []  # Finnhub returns an error object, not a list, on a bad key/rate-limit

        results = []
        for item in data[:max_results]:
            headline = (item.get("headline") or "").strip()
            if not headline:
                continue
            ts = item.get("datetime")
            pub_date = (
                datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
                if ts else None
            )
            summary = (item.get("summary") or "").strip()
            url = item.get("url") or ""
            if not url:
                continue
            results.append({
                "title": f"{item.get('source', 'Finnhub')}: {headline}",
                "url": url,
                "snippet": summary or headline,
                "source_type": "news",
                "publication_date": pub_date,
                "context_label": (
                    "[FINANCIAL NEWS WIRE — company-specific news feed, treat like any "
                    "other news content]"
                ),
            })
        return results


class MockFinnhubProvider(FinancialProvider):
    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        return []


# ---------------------------------------------------------------------------
# Alpaca — real-time (Benzinga-sourced) financial news, free on Alpaca's
# Basic market-data plan. Needs a free PAPER trading account (email + MFA,
# no KYC/funding required for paper-only) to get an API key/secret pair —
# see https://alpaca.markets/docs/api-references/market-data-api/news-data/.
# This is the fastest "first line" source this tool has: news is pushed to
# the feed close to publication time rather than picked up on a search-engine
# crawl delay, which is the actual gap day-trading use cases care about.
# ---------------------------------------------------------------------------
class AlpacaNewsProvider(FinancialProvider):
    """Fails soft (returns []) on missing/invalid credentials, a rate-limit
    hit, or any network error — same contract as the other financial
    providers here."""

    def __init__(self, api_key: str, api_secret: str, lookback_hours: int = 48):
        self.api_key = api_key
        self.api_secret = api_secret
        self.lookback_hours = lookback_hours

    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=self.lookback_hours)
        try:
            resp = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                params={
                    "symbols": ticker.upper(),
                    "start": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": max_results,
                    "sort": "desc",
                    "include_content": "false",
                },
                headers={
                    **HEADERS,
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []

        news = data.get("news") if isinstance(data, dict) else None
        if not isinstance(news, list):
            return []  # bad/expired credentials come back as an error object, not this shape

        results = []
        for item in news[:max_results]:
            headline = (item.get("headline") or "").strip()
            url = item.get("url") or ""
            if not headline or not url:
                continue
            pub_date = (item.get("created_at") or "")[:10] or None
            summary = (item.get("summary") or "").strip()
            source = item.get("source") or "Alpaca/Benzinga"
            results.append({
                "title": f"{source}: {headline}",
                "url": url,
                "snippet": summary or headline,
                "source_type": "news",
                "publication_date": pub_date,
                "context_label": (
                    "[REAL-TIME NEWS WIRE — pushed near publication time, treat like any "
                    "other news content but note it may be very recent/still developing]"
                ),
            })
        return results


class MockAlpacaProvider(FinancialProvider):
    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        return []


# ---------------------------------------------------------------------------
# Alpha Vantage News & Sentiment — free, NASDAQ-licensed, and genuinely
# distinct from the other news providers here: every article carries an
# AI-assigned sentiment score, both overall and per-ticker (so a "Neutral"
# headline that specifically calls out one of your tracked tickers as
# "Somewhat Bearish" is visible directly in the extraction prompt, not left
# for the LLM to infer from tone).
#
# The real constraint: the free key is capped at 25 requests/day, TOTAL,
# across every function on the account — not just this one. That's fine for
# `sweep`/`research` on a modest watchlist (each company costs exactly one
# call here, regardless of how many articles come back), but it runs out
# fast combined with `watch-loop --interval-minutes` polling, and doesn't
# reset until the API's own daily window rolls over. Treat this as a
# daily/periodic-checkup source, not a day-trading fast-polling one — see
# README for the full trade-off.
# ---------------------------------------------------------------------------
class AlphaVantageNewsProvider(FinancialProvider):
    """Fails soft (returns []) on a missing/invalid key, the daily quota
    being exhausted, or any network error — same contract as the other
    financial providers here. Alpha Vantage signals a quota/param problem
    with a 200 OK carrying an "Information" or "Note" key instead of the
    normal "feed" list, so that's checked explicitly rather than relying on
    an HTTP error status, which this API doesn't raise for rate limits."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker.upper(),
                    "limit": max_results,
                    "apikey": self.api_key,
                },
                headers=HEADERS, timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []

        if not isinstance(data, dict) or "feed" not in data:
            # "Information"/"Note" keys mean quota exhausted or a bad
            # param — either way, fail soft rather than raising.
            return []

        feed = data.get("feed")
        if not isinstance(feed, list):
            return []

        results = []
        for item in feed[:max_results]:
            title = (item.get("title") or "").strip()
            url = item.get("url") or ""
            if not title or not url:
                continue
            time_published = item.get("time_published") or ""  # e.g. "20260910T143000"
            pub_date = (
                f"{time_published[0:4]}-{time_published[4:6]}-{time_published[6:8]}"
                if len(time_published) >= 8 else None
            )
            summary = (item.get("summary") or "").strip()
            source = item.get("source") or "Alpha Vantage"

            # Prefer the sentiment specific to THIS ticker over the
            # article's overall sentiment, when available — a broad-market
            # article mentioning the ticker in passing may be "Neutral"
            # overall but genuinely bullish/bearish for this name specifically.
            ticker_label, ticker_score = None, None
            for ts in (item.get("ticker_sentiment") or []):
                if (ts.get("ticker") or "").upper() == ticker.upper():
                    ticker_label = ts.get("ticker_sentiment_label")
                    ticker_score = ts.get("ticker_sentiment_score")
                    break
            sentiment_label = ticker_label or item.get("overall_sentiment_label")
            sentiment_note = f" [AI sentiment for {ticker.upper()}: {sentiment_label}]" if sentiment_label else ""

            results.append({
                "title": f"{source}: {title}{sentiment_note}",
                "url": url,
                "snippet": summary or title,
                "source_type": "news",
                "publication_date": pub_date,
                "context_label": (
                    "[FINANCIAL NEWS + AI SENTIMENT SCORE — this source tags each article "
                    "with an automated bullish/bearish sentiment label; treat the label as "
                    "a machine-generated signal to note, not a substitute for reading what "
                    "the article actually reports]"
                ),
            })
        return results


class MockAlphaVantageProvider(FinancialProvider):
    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        return []


# ---------------------------------------------------------------------------
# SEC EDGAR — official filings. Free, no API key, but the fair-access policy
# requires a descriptive User-Agent identifying who's calling (see
# https://www.sec.gov/search-filings/edgar-application-programming-interfaces).
# Two calls: resolve ticker -> CIK once via the public ticker map (cached for
# the process lifetime — no reason to re-download ~1000 companies' worth of
# mapping per research session), then pull that CIK's recent filings from
# the submissions endpoint.
#
# This is the single highest-reliability source type the tool has (spec
# §11 puts filings at tier 1) — a real primary source, not news about one.
# ---------------------------------------------------------------------------
_SEC_TICKER_MAP_CACHE: dict[str, str] | None = None

# Material/newsworthy forms worth surfacing as "first line" evidence — not
# every routine form the submissions feed returns (e.g. this skips ordinary
# Form 4 insider-trade filings, which are frequent enough to drown out
# everything else and rarely relevant to a thesis-tracking use case).
_MATERIAL_FORMS = {
    "8-K", "10-K", "10-K/A", "10-Q", "10-Q/A", "S-1", "S-1/A",
    "424B4", "424B5", "DEF 14A", "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
    "6-K", "20-F",
}


def _sec_headers() -> dict:
    contact = os.environ.get("SEC_EDGAR_CONTACT", "").strip()
    ua = (
        f"databroker-research-bot/0.2 ({contact})" if contact
        else "databroker-research-bot/0.2 (set SEC_EDGAR_CONTACT in .env — see README)"
    )
    return {"User-Agent": ua}


def _load_sec_ticker_map() -> dict[str, str]:
    """Ticker (upper-case) -> zero-padded 10-digit CIK string."""
    global _SEC_TICKER_MAP_CACHE
    if _SEC_TICKER_MAP_CACHE is not None:
        return _SEC_TICKER_MAP_CACHE
    mapping: dict[str, str] = {}
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_sec_headers(), timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        for entry in data.values():
            ticker = (entry.get("ticker") or "").upper()
            cik = entry.get("cik_str")
            if ticker and cik is not None:
                mapping[ticker] = str(cik).zfill(10)
    except (requests.RequestException, ValueError):
        pass  # empty map -> provider just returns [] per-ticker below, doesn't crash
    _SEC_TICKER_MAP_CACHE = mapping
    return mapping


class SECFilingsProvider(FinancialProvider):
    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        cik = _load_sec_ticker_map().get(ticker.upper())
        if not cik:
            return []
        try:
            resp = requests.get(
                f"https://data.sec.gov/submissions/CIK{cik}.json",
                headers=_sec_headers(), timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return []

        recent = (data.get("filings") or {}).get("recent") or {}
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        descs = recent.get("primaryDocDescription", [])
        items = recent.get("items", [])
        cik_unpadded = str(int(cik))  # the Archives URL wants it without the zero-padding

        results = []
        for i, form in enumerate(forms):
            if form not in _MATERIAL_FORMS:
                continue
            accession = accessions[i] if i < len(accessions) else None
            doc = docs[i] if i < len(docs) else None
            if not accession or not doc:
                continue
            accession_nodash = accession.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{accession_nodash}/{doc}"
            filing_date = dates[i] if i < len(dates) else None
            item_codes = items[i] if i < len(items) else ""
            desc = descs[i] if i < len(descs) else ""

            snippet = f"Form {form} filed on {filing_date or 'an unknown date'}."
            if item_codes:
                snippet += f" Items: {item_codes}."
            if desc:
                snippet += f" {desc}."

            results.append({
                "title": f"SEC filing: {form} ({filing_date or 'undated'})",
                "url": url,
                "snippet": snippet,
                "source_type": "filing",
                "publication_date": filing_date,
                "context_label": (
                    "[OFFICIAL SEC FILING — primary regulatory source, the highest-reliability "
                    "evidence this tool sees; extract the concrete fact(s) the filing describes "
                    "from the form type/items/description below — the full filing text isn't "
                    "fetched, so don't invent details beyond what's stated here]"
                ),
            })
            if len(results) >= max_results:
                break
        return results


class MockSECProvider(FinancialProvider):
    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        return []


class CombinedFinancialProvider(FinancialProvider):
    """Merges several ticker-keyed financial providers (e.g. SEC filings +
    Finnhub news + StockTwits chatter). Splits the per-session hit budget
    across them like CombinedSocialProvider does for search — each source
    type is genuinely different information, not redundant copies of the
    same thing, so all of them earn a slice rather than one crowding out
    the others."""

    def __init__(self, providers: list[FinancialProvider]):
        self.providers = providers

    def get_ticker_activity(self, ticker: str, max_results: int = 10) -> list[dict]:
        out = []
        per_provider = max(1, max_results // max(1, len(self.providers)))
        for p in self.providers:
            out.extend(p.get_ticker_activity(ticker, max_results=per_provider))
        return out


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_social_from_env() -> SocialProvider | None:
    """
    Env var: SOCIAL_BACKEND = off (default) | reddit | hackernews | both

    Reddit prefers OAuth (reddit_client.py) when REDDIT_CLIENT_ID(_N) /
    REDDIT_CLIENT_SECRET(_N) are set, falling back to the unauthenticated
    public search endpoint otherwise — see RedditPublicSearchProvider's
    docstring for the trade-off.
    """
    backend = os.environ.get("SOCIAL_BACKEND", "off").lower()
    providers: list[SocialProvider] = []

    if backend in ("reddit", "both"):
        subreddits = os.environ.get("REDDIT_SUBREDDITS", "").split(",") if os.environ.get("REDDIT_SUBREDDITS") else None
        from .reddit_client import parse_credentials_from_env, RedditOAuthPool

        creds = parse_credentials_from_env()
        if creds:
            user_agent = os.environ.get("REDDIT_USER_AGENT", "databroker-research-bot/0.2")
            pool = RedditOAuthPool(creds, user_agent)
            providers.append(RedditOAuthProvider(pool, subreddits))
        else:
            providers.append(RedditPublicSearchProvider(subreddits))

    if backend in ("hackernews", "hn", "both"):
        providers.append(HackerNewsSocialProvider())

    if not providers:
        return None
    return providers[0] if len(providers) == 1 else CombinedSocialProvider(providers)


def build_financial_from_env() -> FinancialProvider | None:
    """
    Env var: FINANCIAL_BACKEND = off (default) | comma-separated list of:
      stocktwits | finnhub | sec | alpaca | alphavantage

    e.g. FINANCIAL_BACKEND=sec,finnhub,stocktwits,alpaca,alphavantage combines
    all five. `sec` is free with no key; `finnhub` needs FINNHUB_API_KEY (free
    tier, finnhub.io) and is silently skipped (not a crash) if the key is
    missing — `doctor` will flag that. `sec` works better with
    SEC_EDGAR_CONTACT set (a contact string SEC's fair-access policy asks
    for) but still runs without it. `alpaca` needs ALPACA_API_KEY_ID +
    ALPACA_API_SECRET_KEY (free from a paper-trading account, alpaca.markets)
    and is silently skipped if either is missing. `alphavantage` needs
    ALPHA_VANTAGE_API_KEY (free, alphavantage.co) and is silently skipped if
    missing — note its free tier is capped at 25 requests/day TOTAL across
    the whole account, so it's a poor fit combined with fast interval
    polling; see README.
    """
    backend = os.environ.get("FINANCIAL_BACKEND", "off").lower()
    if backend in ("off", ""):
        return None

    names = [b.strip() for b in backend.split(",") if b.strip()]
    providers: list[FinancialProvider] = []
    for name in names:
        if name == "stocktwits":
            providers.append(StockTwitsProvider())
        elif name == "finnhub":
            api_key = os.environ.get("FINNHUB_API_KEY", "")
            if api_key:
                providers.append(FinnhubNewsProvider(api_key))
        elif name == "sec":
            providers.append(SECFilingsProvider())
        elif name == "alpaca":
            key_id = os.environ.get("ALPACA_API_KEY_ID", "")
            secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
            if key_id and secret:
                providers.append(AlpacaNewsProvider(key_id, secret))
        elif name == "alphavantage":
            av_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            if av_key:
                providers.append(AlphaVantageNewsProvider(av_key))

    if not providers:
        return None
    return providers[0] if len(providers) == 1 else CombinedFinancialProvider(providers)
