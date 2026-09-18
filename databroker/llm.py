"""
LLM provider abstraction.

No paid model is required. Supported backends:

  - OllamaProvider     : fully local, free, needs Ollama running (https://ollama.com)
  - GroqProvider       : free-tier cloud, very fast inference (needs a free GROQ_API_KEY)
  - CerebrasProvider   : free-tier cloud, very fast inference (needs a free CEREBRAS_API_KEY)
  - GeminiProvider     : free-tier cloud (needs a free GEMINI_API_KEY from Google AI Studio)
  - PooledProvider     : pools multiple Groq AND Cerebras API keys directly (no external
                         gateway/container needed) and walks a fixed model chain —
                         openai/gpt-oss-120b (Groq) -> gpt-oss-120b (Cerebras) ->
                         zai-glm-4.7 (Cerebras) -> qwen-3.8-27b (Cerebras) — rotating to
                         the next key on a rate limit (429) and falling over to the next
                         model in the chain once every key for the current one is
                         exhausted. Set GROQ_API_KEYS / CEREBRAS_API_KEYS (comma-separated,
                         one or more keys each) and LLM_BACKEND=pool.
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
# Shared: OpenAI-compatible chat-completions call, used by GroqProvider,
# CerebrasProvider, and PooledProvider (which calls it directly per-key
# rather than going through a provider instance).
# ---------------------------------------------------------------------------
def _openai_chat_complete(
    url: str, api_key: str, model: str, system: str, prompt: str, schema_hint: str,
    provider_label: str, docs_url: str, timeout: int = 60,
) -> dict:
    full_prompt = _build_prompt(prompt, schema_hint)
    resp = _post_with_retry(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": full_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        },
        timeout=timeout,
    )
    if resp.status_code == 404:
        raise RuntimeError(
            f"{provider_label} returned 404 for model '{model}' — it's likely been "
            f"deprecated/renamed. Check {docs_url} for the current lineup."
        )
    if resp.status_code == 401:
        raise RuntimeError(f"{provider_label} returned 401 Unauthorized — check the API key is correct.")
    if resp.status_code == 429:
        raise RuntimeError(f"{provider_label} returned 429 Too Many Requests even after retrying with backoff.")
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _extract_json(content)


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
        try:
            return _openai_chat_complete(
                "https://api.groq.com/openai/v1/chat/completions", self.api_key, self.model,
                system, prompt, schema_hint, "Groq", "https://console.groq.com/docs/models",
            )
        except RuntimeError as e:
            if "429" in str(e):
                raise RuntimeError(
                    f"{e} you're hitting the free-tier rate limit faster than backoff can "
                    f"clear it. Try again shortly, lower RESEARCH_MAX_SUBQUESTIONS in .env, "
                    f"or set LLM_BACKEND=pool to spread load across multiple keys."
                ) from e
            raise

    def describe(self) -> str:
        return f"Groq ({self.model})"


# ---------------------------------------------------------------------------
# Free-tier cloud: Cerebras (OpenAI-compatible API, free tier, very fast)
# ---------------------------------------------------------------------------
class CerebrasProvider(LLMProvider):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        # gpt-oss-120b is Cerebras' production reasoning model. zai-glm-4.7
        # and qwen-3.8-27b are currently preview models on Cerebras — fine
        # for this use case, but check https://inference-docs.cerebras.ai
        # /models/overview if either ever 404s (preview models can be
        # withdrawn/renamed with limited notice).
        self.model = model or os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not self.api_key:
            raise RuntimeError("CEREBRAS_API_KEY not set. Get a free key at https://cloud.cerebras.ai")

    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        return _openai_chat_complete(
            "https://api.cerebras.ai/v1/chat/completions", self.api_key, self.model,
            system, prompt, schema_hint, "Cerebras", "https://inference-docs.cerebras.ai/models/overview",
        )

    def describe(self) -> str:
        return f"Cerebras ({self.model})"


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
# Pooled: multiple Groq + Cerebras keys, fixed model chain, no external
# gateway/container needed — the direct-in-code replacement for routing
# through a separate freellmapi instance.
# ---------------------------------------------------------------------------
class PooledProvider(LLMProvider):
    """Pools one or more Groq keys and one or more Cerebras keys and walks a
    fixed chain of (provider, model) attempts, in order:

        1. Groq      openai/gpt-oss-120b
        2. Cerebras  gpt-oss-120b
        3. Cerebras  zai-glm-4.7
        4. Cerebras  qwen-3.8-27b

    For each chain entry, every key configured for that provider is tried
    in turn before moving on to the next entry. A 429 (rate limited) or 401
    (bad/revoked key) on one key advances to the next key for the same
    model; once every key for that model is exhausted, it advances to the
    next (provider, model) pair in the chain. Only raises once every key
    across every entry has failed.

    Configure with GROQ_API_KEYS and/or CEREBRAS_API_KEYS — comma-separated,
    one or more keys each (only one of the two env vars needs to be set).
    Override the chain itself with LLM_POOL_CHAIN, a comma-separated list of
    "provider:model" pairs, e.g.:
        LLM_POOL_CHAIN=groq:openai/gpt-oss-120b,cerebras:zai-glm-4.7
    """

    DEFAULT_CHAIN = [
        ("groq", "openai/gpt-oss-120b"),
        ("cerebras", "gpt-oss-120b"),
        ("cerebras", "zai-glm-4.7"),
        ("cerebras", "qwen-3.8-27b"),
    ]

    _PROVIDER_META = {
        "groq": {
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "label": "Groq",
            "docs_url": "https://console.groq.com/docs/models",
            "keys_env": "GROQ_API_KEYS",
            "single_key_env": "GROQ_API_KEY",
        },
        "cerebras": {
            "url": "https://api.cerebras.ai/v1/chat/completions",
            "label": "Cerebras",
            "docs_url": "https://inference-docs.cerebras.ai/models/overview",
            "keys_env": "CEREBRAS_API_KEYS",
            "single_key_env": "CEREBRAS_API_KEY",
        },
    }

    def __init__(self, chain: list[tuple[str, str]] | None = None, keys: dict[str, list[str]] | None = None):
        self.chain = chain or self._parse_chain_env() or self.DEFAULT_CHAIN
        self.keys = keys or {
            provider: self._load_keys(provider) for provider in self._PROVIDER_META
        }
        if not any(self.keys.values()):
            raise RuntimeError(
                "PooledProvider needs at least one key. Set GROQ_API_KEYS and/or "
                "CEREBRAS_API_KEYS (comma-separated; a single key is fine too)."
            )
        self.last_routed_via: str | None = None

    @classmethod
    def _load_keys(cls, provider: str) -> list[str]:
        meta = cls._PROVIDER_META[provider]
        raw = os.environ.get(meta["keys_env"]) or os.environ.get(meta["single_key_env"], "")
        return [k.strip() for k in raw.split(",") if k.strip()]

    @staticmethod
    def _parse_chain_env() -> list[tuple[str, str]] | None:
        raw = os.environ.get("LLM_POOL_CHAIN")
        if not raw:
            return None
        pairs = []
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            provider, model = entry.split(":", 1)
            pairs.append((provider.strip().lower(), model.strip()))
        return pairs or None

    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        errors: list[str] = []
        for provider, model in self.chain:
            meta = self._PROVIDER_META.get(provider)
            if meta is None:
                errors.append(f"{provider}:{model} — unknown provider (expected 'groq' or 'cerebras')")
                continue
            provider_keys = self.keys.get(provider) or []
            if not provider_keys:
                continue  # no key configured for this provider — skip straight to the next chain entry
            for i, key in enumerate(provider_keys):
                try:
                    result = _openai_chat_complete(
                        meta["url"], key, model, system, prompt, schema_hint,
                        meta["label"], meta["docs_url"],
                    )
                    self.last_routed_via = f"{provider}:{model} (key #{i + 1})"
                    return result
                except RuntimeError as e:
                    msg = str(e)
                    errors.append(f"{provider}:{model} key #{i + 1} — {msg}")
                    # 401 = this specific key is bad/revoked; 429 = this key is
                    # rate-limited right now. Either way, try the next key for
                    # this same model before giving up on it entirely.
                    continue
        raise RuntimeError(
            "PooledProvider: every (provider, model, key) combination in the chain failed. "
            "Details:\n  " + "\n  ".join(errors)
        )

    def describe(self) -> str:
        via = f", last routed via {self.last_routed_via}" if self.last_routed_via else ""
        key_counts = ", ".join(f"{p}={len(ks)} key(s)" for p, ks in self.keys.items() if ks)
        return f"Pooled ({key_counts}{via})"


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
      LLM_BACKEND = auto (default) | ollama | groq | cerebras | gemini | pool | hybrid | claude | mock

    auto: uses Ollama if it's running locally, else the pool if GROQ_API_KEYS/CEREBRAS_API_KEYS
          (or a single GROQ_API_KEY/CEREBRAS_API_KEY) is set, else Groq alone if just
          GROQ_API_KEY is set, else Gemini if GEMINI_API_KEY is set, else MockProvider.

    pool: PooledProvider is primary — see its docstring for the exact chain and key env
          vars. Falls back to local Ollama (if running), then MockProvider, so a bad/
          exhausted key set doesn't hard-crash the whole run.

    hybrid: builds a HybridProvider. Reasoning-heavy tasks (plan, report, thesis) go to
            the pool/Groq/Gemini if available (better quality, and the pool in particular
            absorbs single-key rate limits via its own fallover); everything else goes to
            local Ollama if available. Falls back sensibly if only one option exists.
    """
    VALID_BACKENDS = {"auto", "ollama", "groq", "cerebras", "gemini", "pool", "hybrid", "claude", "mock"}
    backend = os.environ.get("LLM_BACKEND", "auto").lower()

    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown LLM_BACKEND='{backend}'. Valid values: {', '.join(sorted(VALID_BACKENDS))}."
        )

    def try_ollama():
        return OllamaProvider() if OllamaProvider.is_available() else None

    def try_pool():
        if not (os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY")
                or os.environ.get("CEREBRAS_API_KEYS") or os.environ.get("CEREBRAS_API_KEY")):
            return None
        try:
            return PooledProvider()
        except RuntimeError:
            return None

    def try_groq():
        return GroqProvider() if os.environ.get("GROQ_API_KEY") else None

    def try_cerebras():
        return CerebrasProvider() if os.environ.get("CEREBRAS_API_KEY") else None

    def try_gemini():
        return GeminiProvider() if os.environ.get("GEMINI_API_KEY") else None

    if backend == "ollama":
        return OllamaProvider()
    if backend == "pool":
        primary = try_pool()
        if primary:
            return primary
        fallback = try_ollama()
        import sys
        print(
            "[databroker] LLM_BACKEND=pool but no usable GROQ_API_KEYS/CEREBRAS_API_KEYS "
            "were found — "
            + ("falling back to local Ollama." if fallback else
               "falling back to MockProvider (Ollama isn't running either)."),
            file=sys.stderr,
        )
        return fallback or MockProvider()
    if backend == "groq":
        return GroqProvider()
    if backend == "cerebras":
        return CerebrasProvider()
    if backend == "gemini":
        return GeminiProvider()
    if backend == "claude":
        return ClaudeProvider()
    if backend == "mock":
        return MockProvider()

    if backend == "hybrid":
        local = try_ollama()
        cloud = try_pool() or try_groq() or try_gemini()
        if local and cloud:
            heavy_tasks = {"plan", "report", "thesis"}
            return HybridProvider(default=local, overrides={t: cloud for t in heavy_tasks})
        return local or cloud or MockProvider()

    # auto
    local = try_ollama()
    pool = try_pool()
    cloud_groq = try_groq()
    cloud_gemini = try_gemini()
    active_extras = [p for p in (pool, cloud_groq, cloud_gemini) if p]
    if local and active_extras:
        import sys
        print(
            "[databroker] LLM_BACKEND=auto: Ollama is running locally AND another backend "
            "is configured — defaulting to local Ollama. If you meant to use the pool/"
            "Groq/Gemini instead, set LLM_BACKEND explicitly (e.g. LLM_BACKEND=pool) "
            "rather than relying on auto-detection.",
            file=sys.stderr,
        )
    return local or pool or cloud_groq or cloud_gemini or MockProvider()
