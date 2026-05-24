"""Context manager — accumulates results across tasks."""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TaskResult:
    task_id: int
    title: str
    agent: str
    approved: bool
    timestamp: str
    output: str = ""
    artifacts: List[str] = field(default_factory=list)  # Paths to generated files
    feedback: str = ""


class ContextManager:
    """Persistent context that accumulates across task executions.

    Each completed task appends its result so downstream tasks inherit
    the full history of previous work.
    """

    def __init__(self, context_file: str):
        self.context_file = Path(context_file)
        self.results: Dict[int, TaskResult] = {}

    def load(self) -> Dict[int, TaskResult]:
        """Load existing context from disk."""
        if self.context_file.exists():
            try:
                data = json.loads(self.context_file.read_text(encoding="utf-8"))
                self.results = {}
                for item in data.get("results", []):
                    tr = TaskResult(**item)
                    self.results[tr.task_id] = tr
            except (json.JSONDecodeError, TypeError):
                self.results = {}
        return self.results

    def save_result(self, task_id: int, title: str, agent: str, output: str,
                    approved: bool = False, artifacts: List[str] = None,
                    feedback: str = ""):
        """Record a task result and persist to disk."""
        tr = TaskResult(
            task_id=task_id,
            title=title,
            agent=agent,
            approved=approved,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            output=output,
            artifacts=artifacts or [],
            feedback=feedback,
        )
        self.results[task_id] = tr
        self._persist()

    def get_approved_tasks(self) -> List[TaskResult]:
        """Return all approved tasks in order."""
        return sorted(
            [r for r in self.results.values() if r.approved],
            key=lambda r: r.task_id,
        )

    def get_context_for_task(self, task_id: int) -> str:
        """Build a context string from all approved tasks before the given task_id."""
        approved = [r for r in self.get_approved_tasks() if r.task_id < task_id]
        if not approved:
            return "（无前序任务 — 这是第一个任务）"

        parts = []
        for r in approved:
            parts.append(
                f"## Task {r.task_id} 结果（{r.agent}）\n\n{r.output}"
            )
        return "\n\n---\n\n".join(parts)

    def get_all_context(self) -> str:
        """Build full context from all approved tasks."""
        approved = self.get_approved_tasks()
        if not approved:
            return ""
        parts = []
        for r in approved:
            parts.append(
                f"## Task {r.task_id} 结果（{r.agent}）\n\n{r.output}"
            )
        return "\n\n---\n\n".join(parts)

    def get_status(self) -> List[Dict[str, Any]]:
        """Return status summary of all tasks."""
        return sorted(
            [{
                "task_id": r.task_id,
                "title": r.title,
                "agent": r.agent,
                "approved": r.approved,
                "timestamp": r.timestamp,
            } for r in self.results.values()],
            key=lambda x: x["task_id"],
        )

    def is_task_approved(self, task_id: int) -> bool:
        return task_id in self.results and self.results[task_id].approved

    def _persist(self):
        """Write context to disk."""
        self.context_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "results": [asdict(r) for r in sorted(
                self.results.values(), key=lambda r: r.task_id
            )],
        }
        self.context_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
