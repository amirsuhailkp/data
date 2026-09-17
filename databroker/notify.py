"""
Push alerts — zero-cost.

Currently just Telegram: a free bot API with instant push and no server of
your own required. Setup is three steps, all inside Telegram:
  1. Message @BotFather -> /newbot -> follow the prompts -> it gives you a
     bot token (looks like 123456:ABC-...).
  2. Open a chat with your new bot and send it any message (bots can't
     message you first).
  3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates in a browser —
     your numeric chat id is in the JSON under message.chat.id.

Wired into `sweep`/`watch-loop` (see cli.py): after every sweep, the digest
(only genuinely new, medium+ importance developments — see
agent.daily_digest()) is pushed as one alert. Nothing is ever sent twice:
daily_digest() marks each event notified before returning, so a re-run with
nothing new sends nothing.
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod

import requests

# Telegram hard-caps a single sendMessage call's text at this many characters.
TELEGRAM_MESSAGE_LIMIT = 4096


class Notifier(ABC):
    @abstractmethod
    def send(self, text: str) -> bool:
        """Send `text` as an alert. Returns True on success, False on failure —
        never raises, since a broken notifier should never take down a sweep."""

    def describe(self) -> str:
        return type(self).__name__


def _chunk(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(0, len(text), size)]


class TelegramNotifier(Notifier):
    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        if not self.bot_token or not self.chat_id:
            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set. See the module "
                "docstring in notify.py (or the README's Alerts section) for the 3-step setup."
            )

    def send(self, text: str) -> bool:
        ok = True
        for chunk in _chunk(text, TELEGRAM_MESSAGE_LIMIT):
            try:
                resp = requests.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                    json={"chat_id": self.chat_id, "text": chunk, "disable_web_page_preview": True},
                    timeout=15,
                )
                if resp.status_code != 200:
                    print(f"[databroker] Telegram alert failed (HTTP {resp.status_code}): {resp.text[:200]}")
                    ok = False
            except requests.RequestException as e:
                print(f"[databroker] Telegram alert failed: {e}")
                ok = False
        return ok

    def describe(self) -> str:
        return f"Telegram bot -> chat {self.chat_id}"


class MockNotifier(Notifier):
    """Offline testing — records what would have been sent instead of calling the network."""

    def __init__(self):
        self.sent: list[str] = []

    def send(self, text: str) -> bool:
        self.sent.append(text)
        return True

    def describe(self) -> str:
        return "Mock (no real alert sent)"


def build_notifier_from_env() -> Notifier | None:
    """
    NOTIFY_BACKEND = off (default) | telegram | mock

    Returns None if alerting isn't configured — callers treat that as
    "alerts disabled", not an error, so sweeps/loops still work without it.
    """
    backend = os.environ.get("NOTIFY_BACKEND", "off").lower()
    if backend == "off":
        return None
    if backend == "telegram":
        try:
            return TelegramNotifier()
        except RuntimeError as e:
            print(f"[databroker] NOTIFY_BACKEND=telegram but not usable yet: {e}")
            return None
    if backend == "mock":
        return MockNotifier()
    print(f"[databroker] Unknown NOTIFY_BACKEND='{backend}' — alerts disabled.")
    return None
