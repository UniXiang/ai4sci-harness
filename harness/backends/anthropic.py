"""Anthropic (Claude) LLM backend."""

import os
from typing import Optional

from .base import LLMBackend


class AnthropicBackend(LLMBackend):
    """Backend that uses the Anthropic Claude API.

    Requires: pip install anthropic
    Requires: ANTHROPIC_API_KEY environment variable (or passed explicitly).
    """

    def __init__(self, config: dict, api_key: str = None):
        self.model = config.get("model", "claude-sonnet-4-6")
        self.max_tokens_default = config.get("max_tokens", 4096)
        self.temperature_default = config.get("temperature", 0.3)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        self._client = None

    @property
    def name(self) -> str:
        return f"Anthropic/{self.model}"

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "Anthropic backend requires 'pip install anthropic'. "
                    "Or switch to backend: 'mock' in config.yaml."
                )
            if not self._api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set. "
                    "Set it or switch to backend: 'mock' in config.yaml."
                )
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens_default,
            temperature=temperature or self.temperature_default,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Collect text from all TextBlock content blocks
        # (skip ThinkingBlock etc. which don't have .text)
        text_parts = []
        for block in message.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        return '\n'.join(text_parts) if text_parts else str(message.content[0])
