"""Critic auto-correction loop — automatic code review and retry cycle.

Flow: Executor → Sandbox → Critic
         ↑                      │
         └── inject feedback ───┘  (on REJECTED, max 3 retries)
"""

import re
from typing import Tuple

from .sandbox import Sandbox, SandboxResult
from .context import ContextManager


class CriticLoop:
    """Manages the automatic Executor → Sandbox → Critic correction cycle.

    When a task involves code execution:
    1. Executor produces code
    2. Sandbox runs it
    3. Critic reviews output
    4. If REJECTED, feedback is injected back to Executor
    5. Repeat up to max_retries times
    """

    APPROVED_PATTERN = re.compile(r'【\s*APPROVED\s*】', re.IGNORECASE)
    REJECTED_PATTERN = re.compile(r'【\s*REJECTED\s*】', re.IGNORECASE)

    def __init__(self, config: dict, sandbox: Sandbox):
        loop_cfg = config.get("critic_loop", {})
        self.max_retries = loop_cfg.get("max_retries", 3)
        self.timeout_per_review = loop_cfg.get("timeout_per_review", 120)
        self.sandbox = sandbox

    def run(self, executor_agent, critic_agent, task, backend=None,
            context: ContextManager = None,
            executor_backend=None, critic_backend=None) -> Tuple[bool, str, list]:
        """Execute the critic auto-correction loop.

        Args:
            executor_agent: The Executor agent instance.
            critic_agent: The Critic agent instance.
            task: Task object with description and parameters.
            backend: (Legacy) Single LLM backend for both agents.
            context: ContextManager with prior task results.
            executor_backend: LLM backend for Executor (overrides backend).
            critic_backend: LLM backend for Critic (overrides backend).

        Returns:
            Tuple of (approved: bool, final_output: str, artifacts: list).
        """
        exec_bk = executor_backend or backend
        crit_bk = critic_backend or backend
        feedback = ""
        all_outputs = []
        all_artifacts = []

        for attempt in range(1, self.max_retries + 1):
            print(f"\n  [Critic 循环] 第 {attempt}/{self.max_retries} 次尝试...")

            # Step 1: Executor produces code
            print(f"  [Executor] Generating code via {exec_bk.name}...")
            prev_context = context.get_context_for_task(task.number)
            code_output = exec_bk.generate(
                system_prompt=executor_agent.get_system_prompt(),
                user_prompt=executor_agent.build_task_prompt(
                    task, prev_context, feedback=feedback
                ),
            )

            # Step 2: Extract and sandbox-execute the code
            code_blocks = self._extract_code_blocks(code_output)
            if not code_blocks:
                print(f"  [Executor] 未产生有效代码块，重试...")
                feedback = "未在输出中找到 Python 代码块（```python ... ```）。请生成包含完整可执行代码的输出。"
                all_outputs.append(code_output)
                continue

            code = code_blocks[0]  # Use the first code block
            sandbox_result = self.sandbox.run(code, task_id=task.number)

            if not sandbox_result.success:
                print(f"  [Sandbox] 执行失败 (exit={sandbox_result.exit_code})")
                feedback = (
                    f"代码执行失败。\n"
                    f"Exit code: {sandbox_result.exit_code}\n"
                    f"Stderr: {sandbox_result.stderr}\n"
                    f"Stdout: {sandbox_result.stdout}\n"
                    f"请修正代码并重新生成。"
                )
                all_outputs.append(code_output)
                continue

            print(f"  [Sandbox] 执行成功，生成 {len(sandbox_result.output_files)} 个文件")

            # Step 3: Critic reviews
            print(f"  [Critic] Reviewing via {crit_bk.name}...")
            critic_prompt = critic_agent.build_review_prompt(
                task=task,
                code=code,
                sandbox_result=sandbox_result,
                attempt=attempt,
            )
            critic_output = crit_bk.generate(
                system_prompt=critic_agent.get_system_prompt(),
                user_prompt=critic_prompt,
            )

            # Step 4: Check verdict
            if self._is_approved(critic_output):
                print(f"  [Critic] APPROVED")
                final_output = (
                    f"## Executor 输出\n\n{code_output}\n\n"
                    f"## Sandbox 执行结果\n\n"
                    f"Exit code: {sandbox_result.exit_code}\n"
                    f"生成文件: {', '.join(sandbox_result.output_files)}\n\n"
                    f"```\n{sandbox_result.stdout}\n```\n\n"
                    f"## Critic 审查报告\n\n{critic_output}"
                )
                return (True, final_output, sandbox_result.output_files)

            # REJECTED — extract feedback for next attempt
            print(f"  [Critic] REJECTED，注入反馈重试...")
            feedback = self._extract_feedback(critic_output)
            all_outputs.append(code_output)
            all_artifacts.extend(sandbox_result.output_files)

        # Exhausted retries
        print(f"\n  [Critic 循环] 已耗尽 {self.max_retries} 次重试，提交人类审查...")
        final_output = (
            f"## 警告：Critic 自动修正循环耗尽 {self.max_retries} 次重试\n\n"
            f"## 最后一次 Executor 输出\n\n{all_outputs[-1] if all_outputs else 'N/A'}\n\n"
            f"## Critic 最后反馈\n\n{feedback}"
        )
        return (False, final_output, list(set(all_artifacts)))

    def _extract_code_blocks(self, text: str) -> list:
        """Extract Python code blocks from markdown-style output."""
        pattern = r'```python\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            # Try generic code blocks
            pattern = r'```\s*\n(.*?)```'
            matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches]

    def _is_approved(self, critic_output: str) -> bool:
        """Check if critic output contains APPROVED verdict."""
        # APPROVED must appear and REJECTED must not
        has_approved = bool(self.APPROVED_PATTERN.search(critic_output))
        has_rejected = bool(self.REJECTED_PATTERN.search(critic_output))
        return has_approved and not has_rejected

    def _extract_feedback(self, critic_output: str) -> str:
        """Extract actionable feedback from critic's REJECTED output."""
        # Return the full critic output as feedback; remove APPROVED/REJECTED markers
        feedback = re.sub(r'【\s*\w+\s*】', '', critic_output)
        return feedback.strip()
