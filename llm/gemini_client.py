# llm/gemini_client.py — Google Gemini provider via google-genai SDK

import os
from google import genai
from google.genai import types
from llm.base import BaseLLMClient, SYSTEM_PROMPT


class GeminiClient(BaseLLMClient):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment / .env")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.client = genai.Client(api_key=api_key)

    def _extract_text(self, response) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        # Fallback for cases where .text is empty but candidates contain parts.
        candidates = getattr(response, "candidates", None) or []
        chunks: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)
        return "\n".join(chunks).strip()

    def generate_response(self, user_text: str) -> str:
        if not user_text.strip():
            return ""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_text,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return self._extract_text(response)
