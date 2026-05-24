"""CLI entry point for the AI4Sci Harness."""

import argparse
import sys
from pathlib import Path

from .config import load_config
from .orchestrator import Orchestrator
from .exporter import PaperExporter


def main():
    parser = argparse.ArgumentParser(
        prog="harness",
        description="AI4Sci Harness — 5-Node Multi-Agent Research Collaboration Framework",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # init
    p_init = sub.add_parser("init", help="Initialize a new project directory")
    p_init.add_argument("--name", type=str, default="AI4Sci Project", help="Project name")
    p_init.add_argument("--dir", type=str, default=".", help="Target directory")

    # run
    p_run = sub.add_parser("run", help="Run tasks from tasks.md")
    p_run.add_argument("--task", type=int, default=None, help="Run only a specific task number")
    p_run.add_argument("--from", type=int, dest="from_task", default=None,
                       help="Start from a specific task number")
    p_run.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    p_run.add_argument("--tasks", type=str, default="tasks.md", help="Tasks file path")
    p_run.add_argument("--backend", type=str, default=None, help="Override LLM backend")

    # status
    sub.add_parser("status", help="Show task execution status")

    # export
    p_export = sub.add_parser("export", help="Export assembled paper")
    p_export.add_argument("--format", type=str, choices=["latex", "markdown", "both"],
                          default="both", help="Output format")
    p_export.add_argument("--config", type=str, default="config.yaml", help="Config file path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    project_root = Path.cwd()

    if args.command == "init":
        from .config import init_project
        init_project(args.name, Path(args.dir))
        print(f"Project initialized at {Path(args.dir).resolve()}")
        return

    if args.command == "run":
        config = load_config(args.config)
        orchestrator = Orchestrator(config, args.tasks, backend_override=args.backend)
        orchestrator.run(task_filter=args.task, start_from=args.from_task)
        return

    if args.command == "status":
        from .context import ContextManager
        config = load_config(args.config if hasattr(args, 'config') else "config.yaml")
        ctx = ContextManager(config["output"]["context_file"])
        ctx.load()
        status = ctx.get_status()
        if not status:
            print("No tasks have been executed yet.")
        else:
            print("Task Status:")
            print("-" * 60)
            for s in status:
                status_icon = "DONE" if s["approved"] else "PENDING"
                print(f"  Task {s['task_id']}: {s['title']} [{status_icon}]")
        return

    if args.command == "export":
        config = load_config(args.config)
        exporter = PaperExporter(config)
        if args.format in ("latex", "both"):
            path = exporter.export_latex()
            print(f"LaTeX paper exported to: {path}")
        if args.format in ("markdown", "both"):
            path = exporter.export_markdown()
            print(f"Markdown paper exported to: {path}")
        return


if __name__ == "__main__":
    main()
