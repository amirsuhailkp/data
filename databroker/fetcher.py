"""
Fetches and extracts readable text from a URL. Search snippets are often too
short to extract a well-cited claim from, so the agent pulls the actual page
text for the top few results before asking the LLM to extract claims —
this is a lightweight, free stand-in for the "browser research agent" module
in the original spec (full dynamic navigation is a later upgrade; see README).
"""

from __future__ import annotations
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DataBrokerResearchBot/0.1; +personal-use)"}


class PageFetcher:
    def fetch(self, url: str, max_chars: int = 2000) -> str:
        if not url:
            return ""
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
        except requests.RequestException:
            return ""

        text = ""
        try:
            import trafilatura  # pip install trafilatura — best extraction quality
            text = trafilatura.extract(resp.text) or ""
        except ImportError:
            try:
                from bs4 import BeautifulSoup  # pip install beautifulsoup4
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
            except ImportError:
                text = resp.text  # last resort: raw HTML, truncated hard below

        return text[:max_chars]


class MockFetcher(PageFetcher):
    def fetch(self, url: str, max_chars: int = 2000) -> str:
        return ""
