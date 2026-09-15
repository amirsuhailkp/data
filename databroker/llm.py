"""
LLM provider abstraction.

No paid model is required. Supported backends:

  - OllamaProvider     : fully local, free, needs Ollama running (https://ollama.com)
  - GroqProvider       : free-tier cloud, very fast inference (needs a free GROQ_API_KEY)
  - GeminiProvider     : free-tier cloud (needs a free GEMINI_API_KEY from Google AI Studio)
  - FreeLLMAPIProvider : local multi-provider LLM router/gateway (needs a local freellmapi
                         instance — https://github.com/tashfeenahmed/freellmapi — running,
                         default http://localhost:3001). Fans a single request out across
                         whichever free-tier providers you've configured behind it (Groq,
                         Gemini, Cerebras, Mistral, OpenRouter, etc.) and handles rate-limit
                         fallover between them itself, so it's a good fit when a single
                         provider's free-tier cap (e.g. Groq 429s) is the bottleneck.
  - ClaudeProvider     : optional, paid — kept for later, never selected by default
  - MockProvider       : offline, deterministic, for testing without any of the above
  - HybridProvider     : routes each task (plan/extract/score/dedupe/conflict/thesis/report)
                        to whichever of the above providers you assign it to, so you can
                        mix "cheap local model for repetitive extraction" with "a stronger
                        free-cloud model for the harder reasoning steps" (or run 100% local).

Every provider implements one method: complete_json(system, prompt, schema_hint, task).
`task` is a free-text label ("plan", "extract", "score", "dedupe", "conflict", "thesis",
"report", "search_planning") that most providers ignore, but HybridProvider uses it to pick
which underlying model actually handles the call.
"""

from __future__ import annotations
import json
import os
import re
import time
from abc import ABC, abstractmethod

import requests


def _post_with_retry(url: str, max_retries: int = 5, **kwargs) -> requests.Response:
    """POST with retry/backoff on 429 (rate limit) and transient 5xx errors.

    Free-tier LLM backends (Groq, Gemini) enforce per-minute request and/or
    token caps. A multi-step `research` run fires several calls back-to-back
    (plan, one extract per subquestion, conflict-check, thesis, report) with
    no pacing between them, which reliably trips these caps even on light
    usage. Rather than let the whole research run die on the first 429,
    retry with backoff — honoring the server's `Retry-After` header when
    present (both Groq and Gemini send one), falling back to exponential
    backoff (1s, 2s, 4s, 8s, 16s) otherwise. Any other status/error is
    surfaced immediately, unchanged.
    """
    last_resp = None
    for attempt in range(max_retries + 1):
        resp = requests.post(url, **kwargs)
        if resp.status_code != 429 and resp.status_code < 500:
            return resp
        last_resp = resp
        if attempt == max_retries:
            break
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                wait = max(float(retry_after), 0.5)
            except ValueError:
                wait = 2.0 ** attempt
        else:
            wait = 2.0 ** attempt
        print(f"[databroker] Rate limited (HTTP {resp.status_code}) — "
              f"retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
        time.sleep(wait)
    return last_resp


class LLMProvider(ABC):
    @abstractmethod
    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        """Ask the model for a single JSON object matching schema_hint. No web access —
        anything the model needs to reason over must already be in `prompt`."""

    def describe(self) -> str:
        """Human-readable summary of what's actually configured — shown by the CLI
        before any LLM call is made, so a wrong-backend mistake (see OllamaProvider's
        404 case) is visible immediately instead of discovered via a stack trace."""
        return type(self).__name__


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Local/small models sometimes trail off or add stray text. Try to salvage
        # the largest valid-looking JSON object instead of hard-failing the whole call.
        for candidate in re.findall(r"\{.*\}", text, re.DOTALL):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return {}


def _build_prompt(prompt: str, schema_hint: str) -> str:
    return (
        f"{prompt}\n\n"
        "Respond with ONLY a single JSON value matching this shape. No prose, no markdown "
        f"fences, no explanation before or after the JSON:\n{schema_hint}"
    )


# ---------------------------------------------------------------------------
# Local: Ollama
# ---------------------------------------------------------------------------
class OllamaProvider(LLMProvider):
    """Fully local and free. Requires `ollama serve` running and a model pulled,
    e.g. `ollama pull qwen2.5:7b` or `ollama pull llama3.1:8b`."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        full_prompt = _build_prompt(prompt, schema_hint)
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": full_prompt},
                    ],
                    "format": "json",   # Ollama enforces valid JSON output when supported by the model
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=180,
            )
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Is `ollama serve` running? "
                f"(underlying error: {e})"
            ) from e

        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama returned 404 for model '{self.model}' — it likely isn't pulled yet. "
                f"Run `ollama pull {self.model}` (or set OLLAMA_MODEL to a model you already "
                f"have — see `ollama list` for what's installed)."
            )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return _extract_json(content)

    def describe(self) -> str:
        return f"Ollama ({self.model} @ {self.base_url})"

    @staticmethod
    def is_available(base_url: str | None = None) -> bool:
        url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        try:
            requests.get(f"{url}/api/tags", timeout=2)
            return True
        except requests.RequestException:
            return False


# ---------------------------------------------------------------------------
# Free-tier cloud: Groq (OpenAI-compatible API, free tier, fast)
# ---------------------------------------------------------------------------
class GroqProvider(LLMProvider):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        # llama-3.3-70b-versatile and llama-3.1-8b-instant were deprecated by
        # Groq for free/dev-tier accounts (announced June 2026, shut down
        # August 16, 2026). openai/gpt-oss-120b is Groq's recommended
        # replacement for general reasoning quality; openai/gpt-oss-20b is the
        # lighter/faster replacement for high-volume repetitive calls. Check
        # https://console.groq.com/docs/models for the current lineup if
        # this ever 400s on you — Groq's free-tier catalog turns over often.
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")

    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        full_prompt = _build_prompt(prompt, schema_hint)
        resp = _post_with_retry(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
            },
            timeout=60,
        )
        if resp.status_code == 404:
            raise RuntimeError(
                f"Groq returned 404 for model '{self.model}' — it's likely been deprecated/renamed. "
                f"Check https://console.groq.com/docs/models for the current lineup and set GROQ_MODEL."
            )
        if resp.status_code == 401:
            raise RuntimeError("Groq returned 401 Unauthorized — check GROQ_API_KEY is set correctly.")
        if resp.status_code == 429:
            raise RuntimeError(
                "Groq returned 429 Too Many Requests even after retrying with backoff — "
                "you're hitting the free-tier rate limit faster than backoff can clear it. "
                "Try again shortly, lower RESEARCH_MAX_SUBQUESTIONS in .env, or set "
                "LLM_BACKEND=hybrid to split load across Groq and Gemini."
            )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _extract_json(content)

    def describe(self) -> str:
        return f"Groq ({self.model})"


# ---------------------------------------------------------------------------
# Free-tier cloud: Google Gemini (free tier via Google AI Studio)
# ---------------------------------------------------------------------------
class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        # Pro models were removed from Gemini's free tier in ~April 2026 — the
        # free tier is Flash-only now. gemini-2.5-flash is the current safe
        # default; check https://ai.google.dev/gemini-api/docs/models for
        # whatever's newest (Gemini's free-tier lineup moves fast too).
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey")

    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        full_prompt = _build_prompt(prompt, schema_hint)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        resp = _post_with_retry(
            url,
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
                "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json"},
            },
            timeout=60,
        )
        if resp.status_code == 429:
            raise RuntimeError(
                "Gemini returned 429 Too Many Requests even after retrying with backoff — "
                "you're hitting the free-tier rate limit faster than backoff can clear it. "
                "Try again shortly or lower RESEARCH_MAX_SUBQUESTIONS in .env."
            )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _extract_json(text)

    def describe(self) -> str:
        return f"Gemini ({self.model})"


# ---------------------------------------------------------------------------
# Local: freellmapi (self-hosted multi-provider LLM router/gateway)
# https://github.com/tashfeenahmed/freellmapi
# ---------------------------------------------------------------------------
class FreeLLMAPIProvider(LLMProvider):
    """Talks to a locally-running freellmapi gateway over its OpenAI-compatible
    endpoint. freellmapi itself fans a request out across whichever free-tier
    providers you've configured behind it (Groq, Gemini, Cerebras, Mistral,
    OpenRouter, GitHub Models, HuggingFace, Cloudflare, Cohere, ...) and
    retries/fails over between them when one is rate-limited or down — so it
    absorbs exactly the kind of single-provider 429 that GroqProvider/
    GeminiProvider hit on their own, without databroker needing to know which
    underlying provider actually served the request.

    Needs freellmapi running locally (default: `docker compose up` per its
    README, listening on http://localhost:3001) and a unified API key from
    its dashboard's Keys page. Model defaults to "auto", which follows
    whichever fallback chain is active in the freellmapi dashboard; use
    "auto:fast", "auto:smart", "auto:<profile-name>", or a specific catalog
    model id to steer a single request instead (see freellmapi's API docs).
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.environ.get("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("FREELLMAPI_API_KEY")
        self.model = model or os.environ.get("FREELLMAPI_MODEL", "auto")
        if not self.api_key:
            raise RuntimeError(
                "FREELLMAPI_API_KEY not set. Get your unified key from the freellmapi "
                "dashboard's Keys page (freellmapi must be running locally first — see "
                "https://github.com/tashfeenahmed/freellmapi)."
            )
        self.last_routed_via: str | None = None

    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        full_prompt = _build_prompt(prompt, schema_hint)
        try:
            resp = _post_with_retry(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": full_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                },
                timeout=90,
            )
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Could not reach freellmapi at {self.base_url}. Is it running? "
                f"(default: `docker compose up` in the freellmapi repo, listening on "
                f"http://localhost:3001) (underlying error: {e})"
            ) from e
        if resp.status_code == 401:
            raise RuntimeError(
                "freellmapi returned 401 Unauthorized — check FREELLMAPI_API_KEY matches "
                "the unified key shown on its dashboard's Keys page."
            )
        if resp.status_code == 400:
            raise RuntimeError(
                f"freellmapi returned 400 for model '{self.model}' — check it's a valid "
                f"catalog model id or routing alias (e.g. 'auto', 'auto:fast', "
                f"'auto:<profile-name>'). Response: {resp.text[:300]}"
            )
        if resp.status_code == 429:
            raise RuntimeError(
                "freellmapi returned 429 even after retrying with backoff — every provider "
                "in its active fallback chain is currently exhausted or rate-limited. Check "
                "the freellmapi dashboard for provider/key status, or add more free-tier "
                "keys to its chain."
            )
        resp.raise_for_status()
        self.last_routed_via = resp.headers.get("x-routed-via")
        content = resp.json()["choices"][0]["message"]["content"]
        return _extract_json(content)

    def describe(self) -> str:
        via = f", last routed via {self.last_routed_via}" if self.last_routed_via else ""
        return f"FreeLLMAPI ({self.model} @ {self.base_url}{via})"

    @staticmethod
    def is_available(base_url: str | None = None) -> bool:
        url = (base_url or os.environ.get("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")).rstrip("/")
        try:
            requests.get(f"{url}/models", timeout=2)
            return True
        except requests.RequestException:
            return False


# ---------------------------------------------------------------------------
# Optional / paid — kept for anyone who wants to switch back later.
# Never selected by get_default_provider() or build_provider_from_env().
# ---------------------------------------------------------------------------
class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        full_prompt = _build_prompt(prompt, schema_hint)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": full_prompt}],
        )
        text = "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        return _extract_json(text)

    def describe(self) -> str:
        return f"Claude ({self.model}) — paid"


# ---------------------------------------------------------------------------
# Offline testing provider
# ---------------------------------------------------------------------------
class MockProvider(LLMProvider):
    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        return {"mock": True, "task": task, "note": "MockProvider: no reasoning performed.",
                "prompt_preview": prompt[:120]}

    def describe(self) -> str:
        return "Mock (no real LLM — offline testing only)"


# ---------------------------------------------------------------------------
# Hybrid: route each task to a different provider
# ---------------------------------------------------------------------------
class HybridProvider(LLMProvider):
    """
    Example:
        HybridProvider(
            default=OllamaProvider(),                 # local, free, handles most tasks
            overrides={"plan": GroqProvider(), "report": GroqProvider()},  # harder
            # reasoning steps get a stronger free-tier cloud model when a key is available
        )
    """

    def __init__(self, default: LLMProvider, overrides: dict[str, LLMProvider] | None = None):
        self.default = default
        self.overrides = overrides or {}

    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        provider = self.overrides.get(task, self.default)
        return provider.complete_json(system, prompt, schema_hint, task)

    def describe(self) -> str:
        overrides_str = ", ".join(f"{t}->{p.describe()}" for t, p in self.overrides.items())
        return f"Hybrid (default: {self.default.describe()}" + (f"; {overrides_str}" if overrides_str else "") + ")"


# ---------------------------------------------------------------------------
# Factory: build a sensible provider from environment variables alone
# ---------------------------------------------------------------------------
def build_provider_from_env() -> LLMProvider:
    """
    Env vars:
      LLM_BACKEND = auto (default) | ollama | groq | gemini | freellmapi | hybrid | claude | mock

    auto: uses Ollama if it's running locally, else freellmapi if it's running locally
          (FREELLMAPI_API_KEY set), else Groq if GROQ_API_KEY is set, else Gemini if
          GEMINI_API_KEY is set, else falls back to MockProvider.

    freellmapi: freellmapi is primary. If it's unreachable or FREELLMAPI_API_KEY isn't
                set, falls back to local Ollama automatically (if running), then
                MockProvider as a last resort — so a Docker container going down
                doesn't hard-crash your next research run.

    hybrid: builds a HybridProvider. Reasoning-heavy tasks (plan, report, thesis)
            go to freellmapi/Groq/Gemini if available (better quality, and freellmapi
            in particular absorbs single-provider rate limits via its own fallover);
            everything else goes to local Ollama if available. Falls back
            sensibly if only one option exists.
    """
    VALID_BACKENDS = {"auto", "ollama", "groq", "gemini", "freellmapi", "hybrid", "claude", "mock"}
    backend = os.environ.get("LLM_BACKEND", "auto").lower()

    if backend not in VALID_BACKENDS:
        hint = " (did you mean 'freellmapi'? there's no backend called 'api')" if backend == "api" else ""
        raise ValueError(
            f"Unknown LLM_BACKEND='{backend}'{hint}. Valid values: {', '.join(sorted(VALID_BACKENDS))}."
        )

    def try_ollama():
        return OllamaProvider() if OllamaProvider.is_available() else None

    def try_freellmapi():
        if not os.environ.get("FREELLMAPI_API_KEY"):
            return None
        return FreeLLMAPIProvider() if FreeLLMAPIProvider.is_available() else None

    def try_groq():
        return GroqProvider() if os.environ.get("GROQ_API_KEY") else None

    def try_gemini():
        return GeminiProvider() if os.environ.get("GEMINI_API_KEY") else None

    if backend == "ollama":
        return OllamaProvider()
    if backend == "freellmapi":
        primary = try_freellmapi()
        if primary:
            return primary
        fallback = try_ollama()
        import sys
        base_url = os.environ.get("FREELLMAPI_BASE_URL", "http://localhost:3001/v1")
        print(
            f"[databroker] LLM_BACKEND=freellmapi but it isn't reachable/configured "
            f"(check FREELLMAPI_API_KEY is set and the container is up at {base_url}) — "
            + ("falling back to local Ollama." if fallback else
               "falling back to MockProvider (Ollama isn't running either)."),
            file=sys.stderr,
        )
        return fallback or MockProvider()
    if backend == "groq":
        return GroqProvider()
    if backend == "gemini":
        return GeminiProvider()
    if backend == "claude":
        return ClaudeProvider()
    if backend == "mock":
        return MockProvider()

    if backend == "hybrid":
        local = try_ollama()
        cloud = try_freellmapi() or try_groq() or try_gemini()
        if local and cloud:
            heavy_tasks = {"plan", "report", "thesis"}
            return HybridProvider(default=local, overrides={t: cloud for t in heavy_tasks})
        return local or cloud or MockProvider()

    # auto
    local = try_ollama()
    router = try_freellmapi()
    cloud_groq = try_groq()
    cloud_gemini = try_gemini()
    active_extras = [p for p in (router, cloud_groq, cloud_gemini) if p]
    if local and active_extras:
        import sys
        print(
            "[databroker] LLM_BACKEND=auto: Ollama is running locally AND another backend "
            "is configured — defaulting to local Ollama. If you meant to use freellmapi/"
            "Groq/Gemini instead, set LLM_BACKEND explicitly (e.g. LLM_BACKEND=freellmapi) "
            "rather than relying on auto-detection.",
            file=sys.stderr,
        )
    return local or router or cloud_groq or cloud_gemini or MockProvider()
