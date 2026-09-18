"""
Headless freellmapi bootstrap for GitHub Actions.

Each workflow run starts a brand-new freellmapi container with an empty
database, so this does three things every single run (fast — a few HTTP
calls, no browser):

  1. GET  /api/ping                — wait for the container to be up
  2. POST /api/auth/setup          — create the admin account (falls back to
                                      /api/auth/login; shouldn't normally be
                                      hit since each run's container is
                                      fresh, but harmless if it is)
  3. POST /api/keys (once per key) — load every Groq/Cerebras (or other
                                      platform) key from FREELLMAPI_PROVIDER_KEYS_JSON
  4. GET  /api/settings/api-key    — the unified key databroker's
                                      LLM_BACKEND=freellmapi needs

All four routes verified directly against freellmapi's own source
(server/src/routes/auth.ts, keys.ts, settings.ts) rather than assumed.
POST /api/keys takes {"platform": ..., "key": ..., "label": ...} and is
called once per entry in FREELLMAPI_PROVIDER_KEYS_JSON — multiple calls
with the same platform (e.g. two "groq" keys) each insert a separate row,
which is exactly the multi-key pooling this is for.

FREELLMAPI_PROVIDER_KEYS_JSON format (set as a GitHub secret, never a
variable — it contains real provider keys):
  [
    {"platform": "groq", "key": "gsk_...", "label": "groq-1"},
    {"platform": "groq", "key": "gsk_...", "label": "groq-2"},
    {"platform": "cerebras", "key": "csk_...", "label": "cerebras-1"}
  ]

On success, writes the unified key to KEY_FILE for the wrapper script to
export as FREELLMAPI_API_KEY. Never prints the unified key, the provider
keys, or the admin password to stdout/logs.
"""

import json
import os
import sys
import time

import requests

BASE = os.environ.get("FREELLMAPI_INTERNAL_BASE_URL", "http://127.0.0.1:3001")
EMAIL = os.environ.get("FREELLMAPI_ADMIN_EMAIL", "admin@databroker.local")
PASSWORD = os.environ.get("FREELLMAPI_ADMIN_PASSWORD")
KEYS_JSON = os.environ.get("FREELLMAPI_PROVIDER_KEYS_JSON", "")
KEY_FILE = os.environ.get("FREELLMAPI_KEY_FILE", "/tmp/freellmapi_key")


def wait_ready(timeout_s: int = 90) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{BASE}/api/ping", timeout=3).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def get_session_token() -> str:
    resp = requests.post(f"{BASE}/api/auth/setup",
                         json={"email": EMAIL, "password": PASSWORD}, timeout=10)
    if resp.status_code == 201:
        print("[gh_bootstrap] Created the freellmapi admin account.")
        return resp.json()["token"]
    if resp.status_code == 409:
        resp = requests.post(f"{BASE}/api/auth/login",
                             json={"email": EMAIL, "password": PASSWORD}, timeout=10)
        resp.raise_for_status()
        print("[gh_bootstrap] Logged in to the existing freellmapi account.")
        return resp.json()["token"]
    resp.raise_for_status()
    raise RuntimeError(f"unexpected /api/auth/setup response: {resp.status_code} {resp.text[:200]}")


def load_provider_keys(token: str) -> int:
    """Returns the count of keys successfully added. Never logs key values."""
    if not KEYS_JSON.strip():
        print("[gh_bootstrap] No FREELLMAPI_PROVIDER_KEYS_JSON set — freellmapi will "
              "start with zero provider keys and every completion will fail. Set it "
              "as a repo secret (see this script's docstring for the format).")
        return 0

    try:
        entries = json.loads(KEYS_JSON)
    except json.JSONDecodeError as e:
        print(f"[gh_bootstrap] FREELLMAPI_PROVIDER_KEYS_JSON is not valid JSON: {e}", file=sys.stderr)
        return 0

    if not isinstance(entries, list):
        print("[gh_bootstrap] FREELLMAPI_PROVIDER_KEYS_JSON must be a JSON array.", file=sys.stderr)
        return 0

    added = 0
    for i, entry in enumerate(entries):
        platform = entry.get("platform")
        key = entry.get("key")
        label = entry.get("label", f"{platform}-{i}")
        if not platform or not key:
            print(f"[gh_bootstrap] Skipping entry {i}: needs 'platform' and 'key'.", file=sys.stderr)
            continue
        try:
            resp = requests.post(
                f"{BASE}/api/keys",
                json={"platform": platform, "key": key, "label": label},
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if resp.status_code == 201:
                added += 1
                print(f"[gh_bootstrap] Added key '{label}' for platform '{platform}'.")
            else:
                print(f"[gh_bootstrap] Failed to add key '{label}' ({platform}): "
                      f"HTTP {resp.status_code} {resp.text[:150]}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"[gh_bootstrap] Network error adding key '{label}' ({platform}): {e}", file=sys.stderr)

    return added


def main() -> int:
    if not PASSWORD:
        print("[gh_bootstrap] FREELLMAPI_ADMIN_PASSWORD is not set — refusing to use a "
              "guessable default. Set it as a repo secret (8+ characters).", file=sys.stderr)
        return 1

    print("[gh_bootstrap] Waiting for freellmapi to come up...")
    if not wait_ready():
        print("[gh_bootstrap] freellmapi did not become ready in time.", file=sys.stderr)
        return 1

    token = get_session_token()

    added = load_provider_keys(token)
    if added == 0:
        print("[gh_bootstrap] WARNING: zero provider keys loaded — the sweep will run "
              "but every LLM call will fail until FREELLMAPI_PROVIDER_KEYS_JSON is set.")

    resp = requests.get(f"{BASE}/api/settings/api-key",
                        headers={"Authorization": f"Bearer {token}"}, timeout=10)
    resp.raise_for_status()
    api_key = resp.json()["apiKey"]

    with open(KEY_FILE, "w") as f:
        f.write(api_key)
    print(f"[gh_bootstrap] Unified freellmapi key retrieved ({added} provider key(s) loaded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
