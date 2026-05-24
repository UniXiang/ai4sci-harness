"""MiMo LLM backend — Anthropic-compatible API."""

import os
from typing import Optional

from .base import LLMBackend


class MiMoBackend(LLMBackend):
    """Backend that uses MiMo's Anthropic-compatible API.

    Requires: pip install anthropic
    Config: api_key and base_url from config or environment.
    """

    def __init__(self, config: dict):
        self.model = config.get("model", "mimo-v2.5-pro")
        self.max_tokens_default = config.get("max_tokens", 8192)
        self.temperature_default = config.get("temperature", 0.3)
        self._api_key = config.get("api_key", "") or os.environ.get("MIMO_API_KEY", "")
        self._base_url = config.get("base_url", "https://token-plan-cn.xiaomimimo.com/anthropic")

        self._client = None

    @property
    def name(self) -> str:
        return f"MiMo/{self.model}"

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "MiMo backend requires 'pip install anthropic'. "
                    "Or switch to backend: 'mock' in config.yaml."
                )
            if not self._api_key:
                raise ValueError(
                    "MiMo API key not set. "
                    "Set it in config.yaml under the agent's backend section, "
                    "or set MIMO_API_KEY environment variable."
                )
            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                base_url=self._base_url,
            )
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
        text_parts = []
        for block in message.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        return '\n'.join(text_parts) if text_parts else str(message.content[0])
