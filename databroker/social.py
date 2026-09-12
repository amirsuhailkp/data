"""
Phase 4: Social & Community Intelligence.

Opt-in (off by default — see build_social_from_env) additional evidence
sources beyond mainstream web search: public forum/community discussion.
Both backends here are free and need no API key/signup:

  - RedditSocialProvider     : Reddit's public search JSON endpoint
                                (unauthenticated, read-only)
  - HackerNewsSocialProvider : Algolia's official HN Search API
                                (https://hn.algolia.com/api — free, no key)

These feed into the SAME extraction call `gather_evidence()` already makes
per sub-question (see agent.py) — no extra LLM call is spent on social
content specifically. What's different is: (1) every hit is tagged
`source_type="community"` deterministically, so the LLM's own guess at
classifying it is overridden rather than trusted (a smaller/free model is
more likely to mistake a Reddit post for "news" than a bigger one would),
and (2) the extraction prompt is told explicitly to treat this content as
sentiment/discussion, not verified fact.
"""

from __future__ import annotations
import datetime
import os
from abc import ABC, abstractmethod

import requests

HEADERS = {"User-Agent": "databroker-research-bot/0.1 (personal research use)"}


class SocialProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Return {"title","url","snippet","source_type","publication_date"} dicts.
        source_type should be "community" (forums/aggregators) — reserve "social"
        for individual posts on a personal social media feed, which neither
        backend here actually surfaces."""


class RedditSocialProvider(SocialProvider):
    def __init__(self, subreddits: list[str] | None = None):
        # Optionally scope to finance-relevant subreddits to cut down on noise,
        # e.g. ["stocks", "investing", "wallstreetbets"]. None = unrestricted search.
        self.subreddits = subreddits or (
            os.environ.get("REDDIT_SUBREDDITS", "").split(",") if os.environ.get("REDDIT_SUBREDDITS") else None
        )

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        q = query
        if self.subreddits:
            # Reddit search syntax: OR together explicit subreddit scoping.
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

        results = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            created = d.get("created_utc")
            pub_date = (
                datetime.datetime.utcfromtimestamp(created).strftime("%Y-%m-%d") if created else None
            )
            snippet = (d.get("selftext") or "").strip()[:800] or d.get("title", "")
            results.append({
                "title": f"r/{d.get('subreddit', '')}: {d.get('title', '')}",
                "url": "https://www.reddit.com" + d.get("permalink", ""),
                "snippet": snippet,
                "source_type": "community",
                "publication_date": pub_date,
            })
        return results


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


def build_social_from_env() -> SocialProvider | None:
    """
    Env var: SOCIAL_BACKEND = off (default) | reddit | hackernews | both

    Off by default: unauthenticated social scraping is flakier than the
    primary search path (rate limits, layout changes), and not everyone
    wants forum chatter mixed into their research. Turn it on explicitly.
    """
    backend = os.environ.get("SOCIAL_BACKEND", "off").lower()
    providers: list[SocialProvider] = []
    if backend in ("reddit", "both"):
        providers.append(RedditSocialProvider())
    if backend in ("hackernews", "hn", "both"):
        providers.append(HackerNewsSocialProvider())
    if not providers:
        return None
    return providers[0] if len(providers) == 1 else CombinedSocialProvider(providers)
