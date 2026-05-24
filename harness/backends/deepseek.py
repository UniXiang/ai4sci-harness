"""DeepSeek LLM backend — OpenAI-compatible API."""

import os
from typing import Optional

from .base import LLMBackend


class DeepSeekBackend(LLMBackend):
    """Backend that uses DeepSeek's OpenAI-compatible API.

    Requires: pip install openai
    Config: api_key and base_url from config or environment.
    """

    def __init__(self, config: dict):
        self.model = config.get("model", "deepseek-v4-pro")
        self.max_tokens_default = config.get("max_tokens", 8192)
        self.temperature_default = config.get("temperature", 0.3)
        self._api_key = config.get("api_key", "") or os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = config.get("base_url", "https://api.deepseek.com")

        self._client = None

    @property
    def name(self) -> str:
        return f"DeepSeek/{self.model}"

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "DeepSeek backend requires 'pip install openai'. "
                    "Or switch to backend: 'mock' in config.yaml."
                )
            if not self._api_key:
                raise ValueError(
                    "DeepSeek API key not set. "
                    "Set it in config.yaml under the agent's backend section, "
                    "or set DEEPSEEK_API_KEY environment variable."
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens_default,
            temperature=temperature or self.temperature_default,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
