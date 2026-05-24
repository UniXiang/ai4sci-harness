"""LLM Backend abstraction layer."""

from abc import ABC, abstractmethod
from typing import Optional


class LLMBackend(ABC):
    """Abstract interface for LLM backends.

    Plug in any LLM provider by implementing this interface.
    The harness only calls generate(system_prompt, user_prompt) → str.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: The system-level instruction.
            user_prompt: The user/task-level prompt.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.

        Returns:
            The model's text response.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...


def _build_backend(name: str, backend_config: dict) -> LLMBackend:
    """Instantiate a single backend by name."""
    if name == "anthropic":
        from .anthropic import AnthropicBackend
        return AnthropicBackend(backend_config)
    elif name == "deepseek":
        from .deepseek import DeepSeekBackend
        return DeepSeekBackend(backend_config)
    elif name == "mimo":
        from .mimo import MiMoBackend
        return MiMoBackend(backend_config)
    elif name == "mock":
        from .mock import MockBackend
        return MockBackend(backend_config)
    else:
        raise ValueError(
            f"Unknown backend: '{name}'. "
            f"Choose 'anthropic', 'deepseek', 'mimo', or 'mock'."
        )


def create_backend(config: dict, override: str = None) -> LLMBackend:
    """Factory: create the configured LLM backend.

    Args:
        config: Full harness configuration dict.
        override: Optional backend name override.

    Returns:
        An LLMBackend instance.
    """
    llm_config = config.get("llm", {})
    backend_name = override or llm_config.get("backend", "mock")
    return _build_backend(backend_name, llm_config)


def create_backend_for_agent(agent_type: str, config: dict) -> LLMBackend:
    """Create a backend for a specific agent type.

    Looks up agent_backend_routing in config to find which backend
    this agent should use, then creates that backend.

    Args:
        agent_type: One of 'researcher', 'planner', 'executor', 'critic', 'writer'.
        config: Full harness configuration dict.

    Returns:
        An LLMBackend instance for this agent.
    """
    routing = config.get("agent_backend_routing", {})
    agent_cfg = routing.get(agent_type, {})

    backend_name = agent_cfg.get("backend", None)
    if backend_name is None:
        # Fallback to global llm.backend
        return create_backend(config)

    return _build_backend(backend_name, agent_cfg)
