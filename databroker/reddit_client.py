"""
Reddit's official OAuth API (oauth.reddit.com), not the unauthenticated
www.reddit.com/*.json endpoints — OAuth is documented, has a clear free-tier
rate limit (~100 requests/minute per registered app / client_id), and is far
less likely to get silently rate-limited or CAPTCHA'd than scraping the
public JSON endpoints.

Registering a Reddit app is free and takes two minutes: reddit.com/prefs/apps
-> "create app" -> type "script" -> gives you a client_id + client_secret.
Since the free tier caps out at ~100 req/min *per app*, this module supports
pooling several apps' credentials and round-robining requests across
whichever one currently has budget left — 3 registered apps gives roughly
300 req/min in aggregate, not because Reddit raised the limit, but because
three independent buckets are being drawn from.

This is a legitimate use of the free tier (multiple apps under one Reddit
account, each within its own documented limit) — not an attempt to evade
rate limiting on a single credential.
"""

from __future__ import annotations
import os
import threading
import time
from dataclasses import dataclass, field

import requests

REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_BASE = "https://oauth.reddit.com"


class TokenBucket:
    """Simplified fixed-window rate limiter: `capacity` tokens, fully refilled
    every `window_seconds`. A true rolling window would be more precise, but
    a fixed window with a safety margin under the documented limit (see
    `_DEFAULT_CAPACITY` below) is simple, correct enough for this purpose,
    and easy to reason about."""

    def __init__(self, capacity: int, window_seconds: float):
        self.capacity = capacity
        self.window_seconds = window_seconds
        self.tokens = capacity
        self.window_start = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            if now - self.window_start >= self.window_seconds:
                self.tokens = self.capacity
                self.window_start = now
            if self.tokens > 0:
                self.tokens -= 1
                return True
            return False

    def seconds_until_reset(self) -> float:
        with self._lock:
            elapsed = time.monotonic() - self.window_start
            return max(0.0, self.window_seconds - elapsed)


# Reddit's documented free-tier limit is ~100 requests/minute per app.
# A small safety margin (95, not 100) absorbs clock drift between our fixed
# window and Reddit's actual rolling window without meaningfully reducing
# throughput.
_DEFAULT_CAPACITY = 95
_DEFAULT_WINDOW_SECONDS = 60.0


@dataclass
class RedditCredential:
    client_id: str
    client_secret: str
    bucket: TokenBucket = field(default=None)  # set in __post_init__
    access_token: str | None = field(default=None, repr=False)
    token_expires_at: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if self.bucket is None:
            self.bucket = TokenBucket(_DEFAULT_CAPACITY, _DEFAULT_WINDOW_SECONDS)


class RedditOAuthPool:
    """Round-robins requests across however many credentials are configured,
    each with its own independent rate-limit bucket. If every credential is
    currently out of budget, `request()` returns None rather than blocking —
    callers treat that the same as "no results this round" (see
    social.py's RedditOAuthProvider), consistent with every other provider
    in this codebase failing soft rather than crashing a research run."""

    def __init__(self, credentials: list[RedditCredential], user_agent: str):
        if not credentials:
            raise ValueError("RedditOAuthPool needs at least one credential")
        self.credentials = credentials
        self.user_agent = user_agent
        self._rr_index = 0
        self._pick_lock = threading.Lock()

    @property
    def pooled_capacity_per_minute(self) -> int:
        return sum(cred.bucket.capacity for cred in self.credentials)

    def _pick_available_credential(self) -> RedditCredential | None:
        with self._pick_lock:
            n = len(self.credentials)
            for i in range(n):
                idx = (self._rr_index + i) % n
                cred = self.credentials[idx]
                if cred.bucket.try_acquire():
                    self._rr_index = (idx + 1) % n
                    return cred
            return None

    def _get_token(self, cred: RedditCredential) -> str | None:
        if cred.access_token and time.time() < cred.token_expires_at - 30:
            return cred.access_token
        try:
            resp = requests.post(
                REDDIT_TOKEN_URL,
                auth=(cred.client_id, cred.client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": self.user_agent},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            return None
        cred.access_token = data.get("access_token")
        cred.token_expires_at = time.time() + data.get("expires_in", 3600)
        return cred.access_token

    def request(self, path: str, params: dict) -> dict | None:
        """Returns the parsed JSON response, or None if no credential had
        budget available, auth failed, or the request otherwise failed —
        every failure mode is treated the same by callers (skip this round)."""
        cred = self._pick_available_credential()
        if cred is None:
            return None  # every credential exhausted this window

        token = self._get_token(cred)
        if not token:
            return None

        try:
            resp = requests.get(
                f"{REDDIT_API_BASE}{path}",
                headers={"Authorization": f"Bearer {token}", "User-Agent": self.user_agent},
                params=params,
                timeout=15,
            )
            if resp.status_code == 401:
                cred.access_token = None  # force a fresh token next time
                return None
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            return None


def parse_credentials_from_env() -> list[RedditCredential]:
    """
    Supports two shapes:
      - Numbered pairs (recommended for a pool): REDDIT_CLIENT_ID_1 /
        REDDIT_CLIENT_SECRET_1, _2, _3, ... — as many as you've registered.
      - A single unnumbered pair: REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET.
    Numbered pairs take priority if both are present.
    """
    creds = []
    i = 1
    while True:
        cid = os.environ.get(f"REDDIT_CLIENT_ID_{i}")
        csec = os.environ.get(f"REDDIT_CLIENT_SECRET_{i}")
        if not cid or not csec:
            break
        creds.append(RedditCredential(cid, csec))
        i += 1

    if not creds:
        cid = os.environ.get("REDDIT_CLIENT_ID")
        csec = os.environ.get("REDDIT_CLIENT_SECRET")
        if cid and csec:
            creds.append(RedditCredential(cid, csec))

    return creds
