# llm/claude_client.py — Anthropic Claude provider (unchanged logic, now extends BaseLLMClient)

import os
import anthropic
from llm.base import BaseLLMClient, SYSTEM_PROMPT, MAX_TOKENS


class ClaudeClient(BaseLLMClient):
    def __init__(self):
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("CLAUDE_API_KEY not set in environment / .env")
        model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model

    def generate_response(self, user_text: str) -> str:
        if not user_text.strip():
            return ""
        message = self.client.messages.create(
            model      = self.model,
            max_tokens = MAX_TOKENS,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_text}],
        )
        return message.content[0].text.strip()
