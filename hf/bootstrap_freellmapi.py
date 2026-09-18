"""
Headless first-run bootstrap for the freellmapi sidecar.

freellmapi's unified API key (the freellmapi-... bearer token databroker's
LLM_BACKEND=freellmapi needs) is normally copied by hand from its dashboard
after logging in — fine for a desktop install, not for a container with no
one clicking through a browser. This script does the same three calls the
dashboard would make, using freellmapi's own REST API, verified directly
against its source (server/src/routes/auth.ts, server/src/routes/settings.ts):

  1. GET  /api/ping                    - wait for the server to be up
  2. POST /api/auth/setup              - first-run account creation (falls
                                          back to POST /api/auth/login if an
                                          account already exists, e.g. on a
                                          container restart where the SQLite
                                          data survived)
  3. GET  /api/settings/api-key        - the unified key, given the session
                                          token from step 2

Calling from 127.0.0.1 counts as a loopback caller in freellmapi's own
isLoopbackRemote() check, so no separate "setup code" is required here the
way a genuinely remote browser would need one.

The retrieved key is written to a plain file rather than printed, so it
doesn't end up in container logs; entrypoint.sh reads it from there and
exports it as FREELLMAPI_API_KEY before starting databroker.
"""

import os
import sys
import time

import requests

BASE = "http://127.0.0.1:3001"
EMAIL = os.environ.get("FREELLMAPI_ADMIN_EMAIL", "admin@databroker.local")
PASSWORD = os.environ.get("FREELLMAPI_ADMIN_PASSWORD")
KEY_FILE = "/tmp/freellmapi_key"


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
        print("[bootstrap] Created the freellmapi admin account.")
        return resp.json()["token"]
    if resp.status_code == 409:
        # Already claimed — normal on a restart where the data volume survived.
        resp = requests.post(f"{BASE}/api/auth/login",
                             json={"email": EMAIL, "password": PASSWORD}, timeout=10)
        resp.raise_for_status()
        print("[bootstrap] Logged in to the existing freellmapi account.")
        return resp.json()["token"]
    resp.raise_for_status()
    raise RuntimeError(f"unexpected /api/auth/setup response: {resp.status_code} {resp.text[:200]}")


def main() -> int:
    if not PASSWORD:
        print("[bootstrap] FREELLMAPI_ADMIN_PASSWORD is not set — refusing to use a guessable "
              "default for an account that may end up reachable at the Space's public URL. "
              "Set it as a Hugging Face Space secret (8+ characters).", file=sys.stderr)
        return 1

    print("[bootstrap] Waiting for freellmapi to come up...")
    if not wait_ready():
        print("[bootstrap] freellmapi did not become ready in time.", file=sys.stderr)
        return 1

    token = get_session_token()
    resp = requests.get(f"{BASE}/api/settings/api-key",
                        headers={"Authorization": f"Bearer {token}"}, timeout=10)
    resp.raise_for_status()
    api_key = resp.json()["apiKey"]

    with open(KEY_FILE, "w") as f:
        f.write(api_key)
    print("[bootstrap] Unified freellmapi key retrieved and written for databroker to use.")

    if not os.environ.get("FREEAPI_CONFIG_JSON") and not os.environ.get("FREEAPI_CONFIG_PATH"):
        print("[bootstrap] NOTE: no FREEAPI_CONFIG_JSON/FREEAPI_CONFIG_PATH was set, so no "
              "provider keys (Groq, Cerebras, ...) have been configured yet. Add them via "
              "FREEAPI_CONFIG_JSON as a Space secret, or open the dashboard once (see README) "
              "and add them on the Keys page — either way, databroker will start getting real "
              "responses only once at least one provider key is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
