# llm/factory.py — Reads LLM_PROVIDER from .env and returns the right client
# Add a new provider: implement BaseLLMClient, register it in PROVIDERS below.

import os
from llm.base import BaseLLMClient

# Map .env value -> provider class (imported lazily to avoid hard dep errors)
PROVIDERS = {
    "claude": "llm.claude_client.ClaudeClient",
    "gemini": "llm.gemini_client.GeminiClient",
    "ollama": "llm.ollama_client.OllamaClient",
}


def get_llm_client() -> BaseLLMClient:
    """
    Instantiate and return the LLM client configured in .env.
    Set LLM_PROVIDER to one of: claude | gemini | ollama  (default: gemini)
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            f"Choose from: {', '.join(PROVIDERS)}"
        )

    # Dynamic import — only the chosen provider's SDK needs to be installed
    module_path, class_name = PROVIDERS[provider].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls    = getattr(module, class_name)

    print(f"[llm] Using provider: {provider}")
    return cls()
