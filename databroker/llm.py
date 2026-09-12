"""
LLM provider abstraction.

No paid model is required. Supported backends:

  - OllamaProvider   : fully local, free, needs Ollama running (https://ollama.com)
  - GroqProvider     : free-tier cloud, very fast inference (needs a free GROQ_API_KEY)
  - GeminiProvider   : free-tier cloud (needs a free GEMINI_API_KEY from Google AI Studio)
  - ClaudeProvider   : optional, paid — kept for later, never selected by default
  - MockProvider     : offline, deterministic, for testing without any of the above
  - HybridProvider   : routes each task (plan/extract/score/dedupe/conflict/thesis/report)
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
from abc import ABC, abstractmethod

import requests


class LLMProvider(ABC):
    @abstractmethod
    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        """Ask the model for a single JSON object matching schema_hint. No web access —
        anything the model needs to reason over must already be in `prompt`."""


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
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return _extract_json(content)

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
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")

    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        full_prompt = _build_prompt(prompt, schema_hint)
        resp = requests.post(
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
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _extract_json(content)


# ---------------------------------------------------------------------------
# Free-tier cloud: Google Gemini (free tier via Google AI Studio)
# ---------------------------------------------------------------------------
class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey")

    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        full_prompt = _build_prompt(prompt, schema_hint)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        resp = requests.post(
            url,
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
                "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _extract_json(text)


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


# ---------------------------------------------------------------------------
# Offline testing provider
# ---------------------------------------------------------------------------
class MockProvider(LLMProvider):
    def complete_json(self, system: str, prompt: str, schema_hint: str, task: str = "general") -> dict:
        return {"mock": True, "task": task, "note": "MockProvider: no reasoning performed.",
                "prompt_preview": prompt[:120]}


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


# ---------------------------------------------------------------------------
# Factory: build a sensible provider from environment variables alone
# ---------------------------------------------------------------------------
def build_provider_from_env() -> LLMProvider:
    """
    Env vars:
      LLM_BACKEND = auto (default) | ollama | groq | gemini | hybrid | claude | mock

    auto: uses Ollama if it's running locally, else Groq if GROQ_API_KEY is set,
          else Gemini if GEMINI_API_KEY is set, else falls back to MockProvider.

    hybrid: builds a HybridProvider. Reasoning-heavy tasks (plan, report, thesis)
            go to Groq/Gemini if a free-tier key is present (better quality);
            everything else goes to local Ollama if available. Falls back
            sensibly if only one option exists.
    """
    backend = os.environ.get("LLM_BACKEND", "auto").lower()

    def try_ollama():
        return OllamaProvider() if OllamaProvider.is_available() else None

    def try_groq():
        return GroqProvider() if os.environ.get("GROQ_API_KEY") else None

    def try_gemini():
        return GeminiProvider() if os.environ.get("GEMINI_API_KEY") else None

    if backend == "ollama":
        return OllamaProvider()
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
        cloud = try_groq() or try_gemini()
        if local and cloud:
            heavy_tasks = {"plan", "report", "thesis"}
            return HybridProvider(default=local, overrides={t: cloud for t in heavy_tasks})
        return local or cloud or MockProvider()

    # auto
    return try_ollama() or try_groq() or try_gemini() or MockProvider()
