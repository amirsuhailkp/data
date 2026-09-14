"""
Fetches and extracts readable text from a URL. Search snippets are often too
short to extract a well-cited claim from, so the agent pulls the actual page
text for the top few results before asking the LLM to extract claims. This
is the static/default fetcher — see browser.py for the JS-rendering,
dynamic-navigation upgrade (opt-in via FETCHER_BACKEND=browser).
"""

from __future__ import annotations
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DataBrokerResearchBot/0.1; +personal-use)"}


class PageFetcher:
    def fetch(self, url: str, max_chars: int = 2000, sub_question: str | None = None) -> str:
        """`sub_question` is accepted (and ignored here) purely so this and
        BrowserFetcher share one call signature — only BrowserFetcher actually
        uses it, to judge which link to follow on an index/listing page."""
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
    def fetch(self, url: str, max_chars: int = 2000, sub_question: str | None = None) -> str:
        return ""
