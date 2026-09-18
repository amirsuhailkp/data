"""
The one process Hugging Face Spaces actually watches. Spaces are built
around a web app on port 7860 — databroker and freellmapi are both
background daemons with no web interface of their own, so this exists
purely to (a) give the Space something to report "Running" on, and (b)
give an external keep-alive pinger (UptimeRobot, cron-job.org — see the
README) something to hit every ~20-30 minutes so the free tier's
48-hours-of-no-requests sleep timer never fires.

Deliberately does NOT read a PORT environment variable — Hugging Face's
Space config (README.md's `app_port: 7860`) is what actually wires up
port 7860, and freellmapi is separately started with PORT=3001 scoped to
just that one process (see entrypoint.sh). Reading a shared PORT env var
here would risk this process and freellmapi fighting over the same value
if a platform-level PORT ever got injected into the container's general
environment.
"""

import os
import time

from flask import Flask, jsonify

app = Flask(__name__)
START_TIME = time.time()

WATCHLOOP_LOG = "/home/user/app/databroker/data/watchloop.log"


def tail(path: str, n: int = 20) -> str:
    if not os.path.exists(path):
        return "(no log yet — watch-loop may still be starting)"
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:]) or "(log is empty so far)"
    except OSError as e:
        return f"(could not read log: {e})"


@app.route("/")
def index():
    uptime_s = int(time.time() - START_TIME)
    log_tail = tail(WATCHLOOP_LOG).replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <html><head><title>databroker</title></head>
    <body style="font-family: monospace; background:#111; color:#ddd; padding:2rem;">
      <h2>databroker is running</h2>
      <p>Container uptime: {uptime_s}s</p>
      <p>This page exists only so Hugging Face Spaces has a web process to watch,
      and so an external uptime pinger can keep this Space from sleeping.
      The actual work (watch-loop, alerts) runs in the background.</p>
      <h3>Last watch-loop log lines</h3>
      <pre>{log_tail}</pre>
    </body></html>
    """


@app.route("/health")
def health():
    return jsonify(status="ok", uptime_s=int(time.time() - START_TIME))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
