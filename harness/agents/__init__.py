"""Agent module — exports individual agent classes with pre-configured prompts."""

from .base import (
    BaseAgent,
    ResearcherAgent,
    PlannerAgent,
    ExecutorAgent,
    CriticAgent,
    WriterAgent,
)

AGENT_REGISTRY = {
    "researcher": ResearcherAgent,
    "planner": PlannerAgent,
    "executor": ExecutorAgent,
    "critic": CriticAgent,
    "writer": WriterAgent,
}


def create_agent(agent_type: str, agent_config: dict,
                 project_name: str = "") -> BaseAgent:
    """Factory: create an agent instance by type name.

    Args:
        agent_type: One of 'researcher', 'planner', 'executor', 'critic', 'writer'.
        agent_config: Agent configuration dict from config.yaml.
        project_name: Project name for context in prompts.

    Returns:
        An initialized agent instance.

    Raises:
        ValueError if agent_type is unknown.
    """
    agent_type = agent_type.lower()
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent type: '{agent_type}'. "
            f"Known types: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[agent_type](agent_config, project_name)
