"""
Search providers. Local/free LLMs have no built-in web browsing, so DataBroker
does search + page-fetch itself in agent.py and feeds the retrieved text to
the LLM as plain context — this is what replaces Claude's hosted web_search tool.

  - DuckDuckGoSearch : free, no API key, no signup (uses the `duckduckgo-search` package)
  - TavilySearch     : optional, needs a free-tier API key, generally higher-quality
                        results for research/agent use cases if you want to upgrade later
  - MockSearch       : offline, returns nothing (used with MockProvider for testing)
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod

import requests


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Return a list of {"title", "url", "snippet"} dicts."""


class DuckDuckGoSearch(SearchProvider):
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        from duckduckgo_search import DDGS  # pip install duckduckgo-search

        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
        except Exception:
            # Network hiccups / rate limiting shouldn't crash the whole research run —
            # the agent just treats this sub-question as having no evidence this round.
            return []
        return results


class TavilySearch(SearchProvider):
    """Free tier available at https://tavily.com — built specifically for LLM agents,
    often returns cleaner content than raw DuckDuckGo snippets."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY not set. Get a free key at https://tavily.com")

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            return []
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in data.get("results", [])
        ]


class MockSearch(SearchProvider):
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        return []


def build_search_from_env() -> SearchProvider:
    """
    Env var: SEARCH_BACKEND = duckduckgo (default) | tavily | mock
    """
    backend = os.environ.get("SEARCH_BACKEND", "duckduckgo").lower()
    if backend == "tavily":
        return TavilySearch()
    if backend == "mock":
        return MockSearch()
    return DuckDuckGoSearch()
