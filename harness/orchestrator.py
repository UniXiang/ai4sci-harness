"""Orchestrator — main flow control for the AI4Sci Harness."""

import sys
from typing import Optional, Dict

from .config import load_config
from .task_parser import parse_tasks, Task
from .context import ContextManager
from .human_gate import HumanGate
from .sandbox import Sandbox
from .critic_loop import CriticLoop
from .backends import create_backend, create_backend_for_agent, LLMBackend
from .agents import create_agent, BaseAgent
from .pdf_reporter import PDFReporter


class Orchestrator:
    """Main orchestrator that drives the multi-agent workflow.

    Flow:
        Load tasks → For each task:
            1. Look up the assigned agent
            2. Build context from prior approved tasks
            3. Generate agent response via LLM backend
            4. If Executor: run Critic auto-correction loop
            5. Present to human for approval
            6. Save approved result to context

    Supports per-agent backend routing via agent_backend_routing in config.
    """

    def __init__(self, config: dict, tasks_file: str = "tasks.md",
                 backend_override: str = None):
        self.config = config
        self.tasks_file = tasks_file
        self.project_name = config.get("project", {}).get("name", "AI4Sci Project")

        # Initialize backends — one per agent type if routing is configured
        self._backends: Dict[str, LLMBackend] = {}
        self._backend_override = backend_override
        if backend_override:
            # Single backend override for all agents
            self.backend = create_backend(config, override=backend_override)
        else:
            # Per-agent backends (falls back to global llm.backend)
            self.backend = create_backend(config)

        # Initialize components
        self.context = ContextManager(
            config.get("output", {}).get("context_file", "outputs/context.json")
        )
        self.human_gate = HumanGate(config)
        self.sandbox = Sandbox(config)
        self.critic_loop = CriticLoop(config, self.sandbox)
        self.pdf_reporter = PDFReporter(
            config.get("output", {}).get("reports_dir", "reports")
        )

        # Load existing context
        self.context.load()

    def _get_backend(self, agent_type: str) -> LLMBackend:
        """Get the LLM backend for a specific agent type."""
        if self._backend_override:
            return self.backend
        if agent_type not in self._backends:
            self._backends[agent_type] = create_backend_for_agent(
                agent_type, self.config
            )
        return self._backends[agent_type]

    def run(self, task_filter: Optional[int] = None,
            start_from: Optional[int] = None):
        """Execute the task workflow.

        Args:
            task_filter: If set, only run this specific task number.
            start_from: If set, start execution from this task number.
        """
        tasks = parse_tasks(self.tasks_file)

        # Show backend routing
        print(f"\n{'='*70}")
        print(f"  AI4Sci Harness — {self.project_name}")
        print(f"  Tasks: {len(tasks)}")
        if self._backend_override:
            print(f"  Backend (override): {self.backend.name}")
        else:
            routing = self.config.get("agent_backend_routing", {})
            if routing:
                print(f"  Backend routing:")
                for atype in ["researcher", "planner", "executor", "critic", "writer"]:
                    bk = self._get_backend(atype)
                    print(f"    {atype:12s} → {bk.name}")
            else:
                print(f"  Backend: {self.backend.name}")
        print(f"{'='*70}")

        for task in tasks:
            # Skip if filtering
            if task_filter is not None and task.number != task_filter:
                continue
            if start_from is not None and task.number < start_from:
                continue

            # Skip already approved tasks
            if self.context.is_task_approved(task.number):
                print(f"\n  [SKIP] Task {task.number} already approved")
                continue

            backend = self._get_backend(task.agent_type)
            print(f"\n{'─'*70}")
            print(f"  Task {task.number}: {task.title}")
            print(f"  Agent: {task.agent} | Backend: {backend.name}")
            print(f"{'─'*70}")

            # Create the agent
            agent_config = self.config.get("agents", {}).get(task.agent_type, {})
            agent = create_agent(task.agent_type, agent_config, self.project_name)

            # Execute the task
            if task.agent_type == "executor":
                self._execute_with_critic(task, agent)
            elif task.agent_type == "critic":
                self._execute_critic(task, agent)
            else:
                self._execute_standard(task, agent)

    def _execute_standard(self, task: Task, agent: BaseAgent):
        """Execute a standard (non-code) task: Researcher, Planner, Writer."""
        backend = self._get_backend(task.agent_type)
        prev_context = self.context.get_context_for_task(task.number)

        print(f"  [{agent.name}] Generating via {backend.name}...")
        output = backend.generate(
            system_prompt=agent.get_system_prompt(),
            user_prompt=agent.build_task_prompt(task, prev_context),
        )

        # Human gate
        decision, feedback = self.human_gate.wait_for_decision(
            task.number, task.title, agent.name, output,
            auto_critic_passed=True,
        )

        if decision == "approved":
            self.context.save_result(
                task_id=task.number,
                title=task.title,
                agent=agent.name,
                output=output,
                approved=True,
            )
            print(f"  Task {task.number} approved and saved.")
        elif decision == "modify":
            print(f"  Re-running with feedback...")
            prev_context = self.context.get_context_for_task(task.number)
            output = backend.generate(
                system_prompt=agent.get_system_prompt(),
                user_prompt=agent.build_task_prompt(task, prev_context, feedback=feedback),
            )
            decision2, _ = self.human_gate.wait_for_decision(
                task.number, task.title, agent.name, output,
            )
            if decision2 == "approved":
                self.context.save_result(
                    task_id=task.number, title=task.title,
                    agent=agent.name, output=output, approved=True,
                )
                print(f"  Task {task.number} approved and saved.")
            else:
                print(f"  Task {task.number} not approved, skipped.")
        else:  # abort
            print(f"  Execution aborted.")
            return

    def _execute_with_critic(self, task: Task, agent: BaseAgent):
        """Execute an Executor task with the Critic auto-correction loop."""
        executor_backend = self._get_backend("executor")
        critic_backend = self._get_backend("critic")

        # Get critic agent
        critic_config = self.config.get("agents", {}).get("critic", {})
        critic_agent = create_agent("critic", critic_config, self.project_name)

        # Run the critic loop with separate backends
        approved, output, artifacts = self.critic_loop.run(
            executor_agent=agent,
            critic_agent=critic_agent,
            task=task,
            executor_backend=executor_backend,
            critic_backend=critic_backend,
            context=self.context,
        )

        # Human gate (after critic loop finishes)
        decision, feedback = self.human_gate.wait_for_decision(
            task.number, task.title, agent.name, output,
            auto_critic_passed=approved,
        )

        if decision == "approved":
            self.context.save_result(
                task_id=task.number,
                title=task.title,
                agent=agent.name,
                output=output,
                approved=True,
                artifacts=artifacts,
            )
            print(f"  Task {task.number} approved and saved.")
        elif decision == "modify":
            print(f"  Critic loop done, human feedback: {feedback}")
            self.context.save_result(
                task_id=task.number, title=task.title,
                agent=agent.name, output=output, approved=False,
                feedback=feedback,
            )
        else:  # abort
            print(f"  Execution aborted.")
            return

    def _execute_critic(self, task: Task, agent: BaseAgent):
        """Execute a standalone Critic review task (not part of auto-loop)."""
        backend = self._get_backend("critic")
        prev_context = self.context.get_context_for_task(task.number)

        print(f"  [{agent.name}] Reviewing via {backend.name}...")
        output = backend.generate(
            system_prompt=agent.get_system_prompt(),
            user_prompt=agent.build_task_prompt(task, prev_context),
        )

        decision, feedback = self.human_gate.wait_for_decision(
            task.number, task.title, agent.name, output,
            auto_critic_passed=True,
        )

        if decision == "approved":
            self.context.save_result(
                task_id=task.number, title=task.title,
                agent=agent.name, output=output, approved=True,
            )
            print(f"  Task {task.number} approved and saved.")
        elif decision == "modify":
            print(f"  Re-running with feedback...")
            output = backend.generate(
                system_prompt=agent.get_system_prompt(),
                user_prompt=agent.build_task_prompt(task, prev_context, feedback=feedback),
            )
            decision2, _ = self.human_gate.wait_for_decision(
                task.number, task.title, agent.name, output,
            )
            if decision2 == "approved":
                self.context.save_result(
                    task_id=task.number, title=task.title,
                    agent=agent.name, output=output, approved=True,
                )
                print(f"  Task {task.number} approved and saved.")
        else:
            print(f"  Execution aborted.")
