"""
Browser agent — dynamic navigation upgrade over fetcher.py's static fetch.

Two things a plain `requests.get()` can't do that this module adds:

  1. Render JavaScript. Many investor-relations pages and SEC EDGAR filing
     pages build their actual content client-side; a static fetch sees a
     near-empty shell. A headless browser (Playwright/Chromium) renders it
     properly.
  2. Follow one link past an index/listing page. If the URL a search result
     points to turns out to be a filings list or a news index rather than
     the actual article, the static fetcher just extracts whatever thin text
     is on that listing page. This module can take one bounded hop to the
     actual content.

Token discipline, consistent with heuristics.py: dismissing a cookie banner
or clicking "read more" is pattern-matched deterministically — no LLM
involved. The LLM is only ever asked one thing, at most once per fetch:
"given these links, which one likely answers the question?" — and only when
a heuristic (looks_like_index_page) actually flags the page as a listing
rather than content in the first place.

Fully optional: needs `pip install playwright && playwright install
chromium`. If Playwright isn't installed, BrowserFetcher transparently falls
back to PageFetcher's static fetch — nothing breaks for people who don't
want a browser dependency.
"""

from __future__ import annotations
import re

from .fetcher import PageFetcher

# Deterministic patterns for cookie-consent and "expand content" buttons —
# near-universal across sites, no LLM needed to recognize them.
CONSENT_BUTTON_PATTERNS = [
    "accept all", "accept cookies", "i agree", "agree and close", "got it", "allow all",
]
EXPAND_BUTTON_PATTERNS = [
    "read more", "load more", "show more", "continue reading", "view more", "expand", "full article",
]


def looks_like_index_page(text: str, num_links: int, min_text_len: int = 400) -> bool:
    """A page with thin visible text but a lot of links is probably a listing
    (an SEC EDGAR filings table, a news category page) rather than the
    article itself — worth spending one navigation hop on."""
    return len(text) < min_text_len and num_links >= 5


class NavigationAgent:
    """The one place a browser hop spends an LLM call: picking which link on
    an index page most likely leads to content answering the sub-question.
    One call per hop, and BrowserFetcher caps hops per fetch (default 1) —
    this is deliberately not a general web-browsing agent, just enough to
    get past a listing page."""

    def __init__(self, llm):
        self.llm = llm

    def choose_link(self, links: list[dict], sub_question: str) -> str | None:
        if not links:
            return None
        options = "\n".join(f"{i}: {l['text']}" for i, l in enumerate(links))
        result = self.llm.complete_json(
            system=(
                "You are helping navigate a website to find specific information. "
                "Pick the single link most likely to lead to content answering the "
                "question. If none look relevant, say so rather than guessing."
            ),
            prompt=f"Question: {sub_question}\n\nLinks found on this page:\n{options}",
            schema_hint='{"link_index": <int or null>}',
            task="navigate",
        )
        idx = result.get("link_index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(links):
            return None
        return links[idx]["url"]


class BrowserFetcher(PageFetcher):
    def __init__(self, navigator: NavigationAgent | None = None, max_hops: int = 1):
        self.navigator = navigator
        self.max_hops = max_hops
        self._playwright_available = self._check_playwright()

    @staticmethod
    def _check_playwright() -> bool:
        try:
            import playwright  # noqa: F401
            return True
        except ImportError:
            return False

    def fetch(self, url: str, max_chars: int = 2000, sub_question: str | None = None) -> str:
        if not url:
            return ""
        if not self._playwright_available:
            return super().fetch(url, max_chars=max_chars)  # graceful fallback, no browser installed

        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    page = browser.new_page(
                        user_agent="Mozilla/5.0 (compatible; DataBrokerResearchBot/0.2; +personal-use)"
                    )
                    text, links = self._render_and_extract(page, url)

                    hops = 0
                    while hops < self.max_hops and looks_like_index_page(text, len(links)):
                        next_url = self._choose_link(links, sub_question)
                        if not next_url:
                            break
                        text, links = self._render_and_extract(page, next_url)
                        hops += 1

                    return text[:max_chars]
                finally:
                    browser.close()
        except Exception:
            # Any browser-side failure (timeout, crashed page, bad selector, etc.)
            # falls back to the static fetch rather than losing this source entirely.
            return super().fetch(url, max_chars=max_chars)

    def _render_and_extract(self, page, url: str) -> tuple[str, list[dict]]:
        page.goto(url, timeout=20000, wait_until="networkidle")
        self._dismiss_consent_and_expand(page)
        html = page.content()
        text = self._html_to_text(html)
        links = self._extract_links(page, url)
        return text, links

    def _dismiss_consent_and_expand(self, page):
        for pattern in CONSENT_BUTTON_PATTERNS + EXPAND_BUTTON_PATTERNS:
            try:
                locator = page.get_by_text(re.compile(pattern, re.IGNORECASE))
                if locator.count() > 0:
                    locator.first.click(timeout=1000)
                    page.wait_for_timeout(300)
            except Exception:
                continue  # best-effort only — a missed button just means slightly thinner text

    def _html_to_text(self, html: str) -> str:
        try:
            import trafilatura
            return trafilatura.extract(html) or ""
        except ImportError:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator=" ", strip=True)

    def _extract_links(self, page, base_url: str) -> list[dict]:
        try:
            raw = page.eval_on_selector_all(
                "a[href]", "els => els.map(e => ({text: e.innerText.trim(), href: e.href}))"
            )
        except Exception:
            return []
        seen = set()
        links = []
        for l in raw:
            href, text = l.get("href", ""), (l.get("text") or "").strip()
            if not href or not text or href in seen or href == base_url:
                continue
            seen.add(href)
            links.append({"text": text[:120], "url": href})
        return links[:30]

    def _choose_link(self, links: list[dict], sub_question: str | None) -> str | None:
        if not links or not self.navigator or not sub_question:
            return None  # no LLM available to judge — don't guess, just stop navigating
        return self.navigator.choose_link(links, sub_question)
