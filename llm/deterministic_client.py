# llm/deterministic_client.py
#
# A BaseLLMClient that returns scripted responses from responses/responses.yaml
# instead of calling any LLM API.
#
# Why this exists
# ───────────────
# In production kiosks / Jetson deployments you often want:
#   • Zero latency  (no network round-trip, no GPU inference)
#   • Zero cost     (no API billing)
#   • 100% predictable wording  (QA, demos, regulated environments)
#
# Switch between modes in .env:
#   RESPONSE_MODE=deterministic   ← this client
#   RESPONSE_MODE=llm             ← whichever LLM_PROVIDER is set
#
# Script format
# ─────────────
# The YAML file maps prompt *keys* to exact response strings.
# session.py's get_next_prompt() returns a key (e.g. "ask.name") when
# RESPONSE_MODE=deterministic, or a natural-language LLM instruction when
# RESPONSE_MODE=llm.  This client resolves the key → string.
#
# Keys support one substitution token: {value}
# Example:  "ack.name" → "Got it, {value}. And your phone number?"
#           generate_response("ack.name|Ravi Kumar") → "Got it, Ravi Kumar. ..."

from __future__ import annotations

import os
from pathlib import Path

from llm.base import BaseLLMClient

# Lazy import so PyYAML is only required when this client is actually used
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_DEFAULT_SCRIPT = _PROJECT_ROOT / "responses" / "responses.yaml"


def _load_script(path: Path) -> dict[str, str]:
    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for deterministic mode.\n"
            "  Install it:  pip install pyyaml"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Responses script not found: {path}\n"
            f"  Create it or set RESPONSES_FILE= in .env"
        )
    with path.open("r", encoding="utf-8") as fh:
        data = _yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"responses.yaml must be a YAML mapping, got {type(data)}")
    # Ensure all values are strings
    return {str(k): str(v) for k, v in data.items()}


class DeterministicClient(BaseLLMClient):
    """
    Returns scripted, pre-written responses from responses/responses.yaml.
    No LLM API is called. Zero latency, zero cost.

    Input contract
    ──────────────
    generate_response() receives a *prompt key*, optionally with a value
    appended via the pipe separator:

        "ask.name"           → looks up script["ask.name"]
        "ack.name|Ravi Kumar"→ looks up script["ack.name"].format(value="Ravi Kumar")

    If the key is not found, the "fallback" entry is returned.
    If "fallback" is also missing, a safe default string is returned.
    """

    def __init__(self) -> None:
        script_path_env = os.getenv("RESPONSES_FILE", "").strip()
        script_path = (
            Path(script_path_env)
            if script_path_env
            else _DEFAULT_SCRIPT
        )
        self._script = _load_script(script_path)
        print(f"[llm] Deterministic mode — script: {script_path}")
        print(f"[llm] Loaded {len(self._script)} response entries.")

    def generate_response(self, prompt_key: str) -> str:
        """
        Resolve a prompt key to its scripted response string.

        prompt_key format:  "<key>"  or  "<key>|<value>"
        """
        # Split off optional value
        if "|" in prompt_key:
            key, value = prompt_key.split("|", 1)
        else:
            key, value = prompt_key, ""

        key = key.strip()
        template = self._script.get(key)

        if template is None:
            print(f"[llm] Warning: key '{key}' not in script — using fallback.")
            template = self._script.get("fallback", "I'm sorry, I didn't understand.")

        if value and "{value}" in template:
            return template.replace("{value}", value)

        return template