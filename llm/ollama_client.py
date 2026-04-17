# llm/ollama_client.py — Local Ollama provider via HTTP API (no SDK needed)
# Ollama must be running locally: https://ollama.com
# Default endpoint: http://localhost:11434

import os
import json
import urllib.request
from llm.base import BaseLLMClient, SYSTEM_PROMPT


class OllamaClient(BaseLLMClient):
    def __init__(self):
        base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.url   = f"{base_url}/api/chat"

    def generate_response(self, user_text: str) -> str:
        if not user_text.strip():
            return ""

        payload = json.dumps({
            "model":  self.model,
            "stream": False,
            "messages": [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_text},
            ],
        }).encode()

        req  = urllib.request.Request(
            self.url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        return data["message"]["content"].strip()
