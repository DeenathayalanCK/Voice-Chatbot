# llm/factory.py
#
# Reads RESPONSE_MODE and LLM_PROVIDER from .env and returns the right client.
#
# RESPONSE_MODE=deterministic  → DeterministicClient (scripted YAML responses)
# RESPONSE_MODE=llm            → whichever LLM_PROVIDER is configured
#
# Default: RESPONSE_MODE=llm  (backward-compatible)

import os
import importlib

from llm.base import BaseLLMClient

# LLM provider registry — add new providers here
_LLM_PROVIDERS = {
    "claude":  "llm.claude_client.ClaudeClient",
    "gemini":  "llm.gemini_client.GeminiClient",
    "ollama":  "llm.ollama_client.OllamaClient",
}


def _load_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_llm_client() -> BaseLLMClient:
    """
    Return the appropriate response client based on RESPONSE_MODE.

    RESPONSE_MODE=deterministic  → DeterministicClient
    RESPONSE_MODE=llm (default)  → LLM provider set by LLM_PROVIDER
    """
    mode = os.getenv("RESPONSE_MODE", "llm").strip().lower()

    if mode == "deterministic":
        from llm.deterministic_client import DeterministicClient
        return DeterministicClient()

    if mode == "llm":
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        if provider not in _LLM_PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER='{provider}'. "
                f"Choose from: {', '.join(_LLM_PROVIDERS)}"
            )
        cls = _load_class(_LLM_PROVIDERS[provider])
        print(f"[llm] Using provider: {provider}")
        return cls()

    raise ValueError(
        f"Unknown RESPONSE_MODE='{mode}'. Choose: llm | deterministic"
    )