"""Configuration loader for the AI4Sci Harness."""

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"
DEFAULT_TASKS = Path(__file__).resolve().parent.parent / "tasks.md"

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _load_dotenv(project_root: Path) -> None:
    """Load .env file into os.environ (minimal, no external dependency)."""
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} placeholders using os.environ."""
    if isinstance(obj, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def load_config(path: str = None) -> Dict[str, Any]:
    """Load YAML configuration from path, expanding ${ENV_VAR} placeholders."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _load_dotenv(config_path.parent)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _expand_env_vars(raw)


def init_project(name: str, target_dir: Path) -> None:
    """Initialize a new project directory with default config and tasks."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy default config
    config_src = DEFAULT_CONFIG
    config_dst = target_dir / "config.yaml"
    if not config_dst.exists():
        with open(config_src, "r", encoding="utf-8") as src:
            content = src.read()
        content = content.replace(
            "Neutrino RGE Stability Study", name
        )
        with open(config_dst, "w", encoding="utf-8") as dst:
            dst.write(content)

    # Copy default tasks
    tasks_dst = target_dir / "tasks.md"
    if not tasks_dst.exists():
        with open(DEFAULT_TASKS, "r", encoding="utf-8") as src:
            content = src.read()
        with open(tasks_dst, "w", encoding="utf-8") as dst:
            dst.write(content)

    # Create skeleton directories
    for d in ["sandbox", "outputs", "paper"]:
        (target_dir / d).mkdir(exist_ok=True)

    # Create harness symlink for convenience
    harness_link = target_dir / "harness"
    if not harness_link.exists():
        harness_src = Path(__file__).resolve().parent
        try:
            harness_link.symlink_to(harness_src)
        except OSError:
            pass  # symlink not supported or already exists
