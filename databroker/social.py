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
    Env var: FINANCIAL_BACKEND = off (default) | stocktwits
    """
    backend = os.environ.get("FINANCIAL_BACKEND", "off").lower()
    if backend == "stocktwits":
        return StockTwitsProvider()
    return None
