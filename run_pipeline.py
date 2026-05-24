#!/usr/bin/env python3
"""Batch runner for the AI4Sci Harness — runs all tasks with PDF reports.

Usage:
    .venv/bin/python3 run_pipeline.py

This script:
1. Runs all tasks through the harness (auto-approved mode)
2. Generates a PDF report for each task
3. Assembles and exports the final paper as PDF
"""

import sys
import os
import re
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from harness.config import load_config
from harness.task_parser import parse_tasks
from harness.context import ContextManager
from harness.sandbox import Sandbox
from harness.critic_loop import CriticLoop
from harness.backends import create_backend_for_agent
from harness.agents import create_agent
from harness.exporter import PaperExporter
from harness.pdf_reporter import PDFReporter

APPROVED_RE = re.compile(r'【\s*APPROVED\s*】', re.IGNORECASE)
REJECTED_RE = re.compile(r'【\s*REJECTED\s*】', re.IGNORECASE)


def parse_verdict(output: str) -> tuple:
    """Parse Critic's verdict from output. Returns (is_approved, is_rejected)."""
    has_approved = bool(APPROVED_RE.search(output))
    has_rejected = bool(REJECTED_RE.search(output))
    return (has_approved, has_rejected)


def run_pipeline():
    config = load_config("config.yaml")
    tasks = parse_tasks("tasks.md")
    project_name = config["project"]["name"]

    # Init components — per-agent backends
    backends = {}
    for atype in ["researcher", "planner", "executor", "critic", "writer"]:
        backends[atype] = create_backend_for_agent(atype, config)

    context = ContextManager(config["output"]["context_file"])
    context.load()
    sandbox = Sandbox(config)
    critic_loop = CriticLoop(config, sandbox)
    pdf_reporter = PDFReporter(config.get("output", {}).get("reports_dir", "reports"))

    print("=" * 70)
    print(f"  AI4Sci Harness — Batch Pipeline")
    print(f"  Project: {project_name}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Mode: AUTO-APPROVE (batch)")
    print(f"  Backend routing:")
    for atype, bk in backends.items():
        print(f"    {atype:12s} → {bk.name}")
    print("=" * 70)

    for task in tasks:
        if context.is_task_approved(task.number):
            print(f"\n  [SKIP] Task {task.number} already approved")
            continue

        backend = backends[task.agent_type]
        print(f"\n{'─' * 70}")
        print(f"  TASK {task.number}: {task.title}")
        print(f"  Agent: {task.agent} | Backend: {backend.name}")
        print(f"{'─' * 70}")

        agent_config = config["agents"].get(task.agent_type, {})
        agent = create_agent(task.agent_type, agent_config, project_name)
        prev_context = context.get_context_for_task(task.number)

        # Execute based on agent type
        sandbox_log = ""

        if task.agent_type == "executor":
            # Executor + Critic loop
            critic_config = config["agents"].get("critic", {})
            critic_agent = create_agent("critic", critic_config, project_name)

            approved, output, artifacts = critic_loop.run(
                executor_agent=agent,
                critic_agent=critic_agent,
                task=task,
                executor_backend=backends["executor"],
                critic_backend=backends["critic"],
                context=context,
            )
            sandbox_log = f"Critic auto-loop: {'APPROVED' if approved else 'EXHAUSTED RETRIES'}\nArtifacts: {artifacts}"

        elif task.agent_type == "critic":
            # --- Standalone Critic task with retry loop ---
            MAX_CRITIC_RETRIES = 3
            critic_approved = False
            critic_feedback = ""
            output = ""

            for attempt in range(1, MAX_CRITIC_RETRIES + 1):
                print(f"  [Critic] Reviewing (attempt {attempt}/{MAX_CRITIC_RETRIES})...")
                output = backend.generate(
                    system_prompt=agent.get_system_prompt(),
                    user_prompt=agent.build_task_prompt(task, prev_context, feedback=critic_feedback),
                )
                has_approved, has_rejected = parse_verdict(output)

                if has_approved and not has_rejected:
                    print(f"  [Critic] APPROVED")
                    critic_approved = True
                    break
                elif has_rejected:
                    print(f"  [Critic] REJECTED — injecting feedback for retry")
                    critic_feedback = REJECTED_RE.sub('', output).strip()[:3000]
                else:
                    print(f"  [Critic] No clear verdict detected, treating as PENDING")
                    critic_feedback = "Output must begin with 【APPROVED】 or 【REJECTED】."

            if not critic_approved:
                print(f"  [Critic] Exhausted {MAX_CRITIC_RETRIES} retries, marking for human review")
            approved = critic_approved
        else:
            output = backend.generate(
                system_prompt=agent.get_system_prompt(),
                user_prompt=agent.build_task_prompt(task, prev_context),
            )
            approved = True

        # Save result
        context.save_result(
            task_id=task.number,
            title=task.title,
            agent=agent.name,
            output=output,
            approved=True,
        )

        # Generate PDF report
        print(f"  [PDF] Generating report for Task {task.number}...")
        pdf_path = pdf_reporter.generate_task_report(
            task_id=task.number,
            title=task.title,
            agent=agent.name,
            output=output,
            approved=True,
            sandbox_log=sandbox_log,
        )
        print(f"  [PDF] Saved: {pdf_path}")
        print(f"  Task {task.number} complete.")

    # Export final paper
    print(f"\n{'=' * 70}")
    print(f"  EXPORTING FINAL PAPER")
    print(f"{'=' * 70}")

    exporter = PaperExporter(config)
    md_path = exporter.export_markdown()
    tex_path = exporter.export_latex()
    print(f"  Markdown: {md_path}")
    print(f"  LaTeX: {tex_path}")

    # Generate final paper PDF
    full_context = context.get_all_context()
    final_md = f"# {project_name}\n\n"
    final_md += "**AI4Sci Harness — 5-Node Multi-Agent Research Framework**\n\n"
    final_md += "---\n\n"
    final_md += full_context

    final_pdf = pdf_reporter.generate_final_paper(final_md, project_name)
    print(f"  Final PDF: {final_pdf}")

    # Summary
    approved = context.get_approved_tasks()
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Tasks completed: {len(approved)}/{len(tasks)}")
    print(f"  PDF reports: reports/")
    print(f"  Final paper: {final_pdf}")
    print(f"{'=' * 70}")

    return final_pdf


if __name__ == "__main__":
    run_pipeline()
