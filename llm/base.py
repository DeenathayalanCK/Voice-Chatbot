# llm/base.py — Abstract base class for all LLM providers
# Every provider must implement generate_response(). main.py only talks to this interface.

from abc import ABC, abstractmethod

SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses concise and clear."
MAX_TOKENS    = 1024


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_response(self, user_text: str) -> str:
        """Send user_text and return the assistant reply as a plain string."""
        ...
