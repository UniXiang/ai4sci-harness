"""Human-in-the-loop gate — pauses for human approval after each task."""

import sys
from typing import Tuple


class HumanGate:
    """Interactive approval gate for human-in-the-loop workflow.

    After each task completes, displays results and waits for one of:
        approved  — accept and continue to next task
        modify X  — re-run with feedback X
        abort     — stop execution entirely
    """

    def __init__(self, config: dict):
        gate_cfg = config.get("human_gate", {})
        self.enabled = gate_cfg.get("enabled", True)
        self.cmd_approved = gate_cfg.get("commands", {}).get("approved", "approved")
        self.cmd_modify = gate_cfg.get("commands", {}).get("modify", "modify")
        self.cmd_abort = gate_cfg.get("commands", {}).get("abort", "abort")

    def wait_for_decision(self, task_id: int, title: str, agent: str,
                          output: str, auto_critic_passed: bool = True) -> Tuple[str, str]:
        """Display task results and wait for human decision.

        Args:
            task_id: Task number.
            title: Task title.
            agent: Agent that executed the task.
            output: The agent's output text.
            auto_critic_passed: Whether the auto critic loop approved the work.

        Returns:
            Tuple of (decision, feedback), where decision is one of
            'approved', 'modify', 'abort'.
        """
        if not self.enabled:
            return ("approved", "")

        # Display results
        print("\n" + "=" * 70)
        print(f"  Task {task_id} 完成 — {title}")
        print(f"  负责 Agent: {agent}")
        if auto_critic_passed:
            print(f"  Critic 自动审查: 通过")
        print("=" * 70)
        print()
        print("--- 任务输出 ---")
        print(output[:5000])  # Truncate very long outputs
        if len(output) > 5000:
            print(f"\n... (输出被截断，共 {len(output)} 字符)")
        print()
        print("-" * 70)
        print(f"  输入 '{self.cmd_approved}' — 确认通过，继续下一任务")
        print(f"  输入 '{self.cmd_modify} <修改意见>' — 带意见重新执行")
        print(f"  输入 '{self.cmd_abort}' — 终止执行")
        print("-" * 70)

        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n中断。")
                return ("abort", "")

            if not user_input:
                continue

            if user_input.lower() == self.cmd_abort:
                return ("abort", "")

            if user_input.lower() == self.cmd_approved:
                return ("approved", "")

            if user_input.lower().startswith(self.cmd_modify):
                feedback = user_input[len(self.cmd_modify):].strip()
                if not feedback:
                    print("请提供修改意见。例如: modify 需要修正 CP 相位的初值")
                    continue
                return ("modify", feedback)

            print(f"未识别的命令。请输入 '{self.cmd_approved}'、'{self.cmd_modify} <意见>' 或 '{self.cmd_abort}'")

    def wait_for_approval_simple(self, message: str = "") -> bool:
        """Simple yes/no confirmation."""
        prompt = f"{message}\n输入 'approved' 继续: " if message else "输入 'approved' 继续: "
        try:
            user_input = input(prompt).strip()
            return user_input.lower() == self.cmd_approved
        except (EOFError, KeyboardInterrupt):
            return False
