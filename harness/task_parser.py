"""Task parser — extracts task definitions from tasks.md."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Task:
    number: int
    title: str
    agent: str          # Agent type: Researcher, Planner, Executor, Critic, Writer
    description: str
    expected_output: str = ""
    raw_text: str = ""

    @property
    def agent_type(self) -> str:
        return self.agent.lower()


def parse_tasks(filepath: str) -> List[Task]:
    """Parse a tasks.md file and return a list of Task objects.

    Supports these formats:
        ### Task N: Title
        ## Task N: Title
        **Task N**: Title
        # N. Title

    Each task block must contain:
        - 负责：AgentName
        - 描述：... (multiline)
        - 产出：... (optional)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {filepath}")

    text = path.read_text(encoding="utf-8")

    # Split into task blocks using various heading patterns
    task_blocks = _split_tasks(text)
    tasks = []

    for block in task_blocks:
        task = _parse_task_block(block)
        if task is not None:
            tasks.append(task)

    if not tasks:
        raise ValueError(f"No valid tasks found in {filepath}. "
                         "Ensure tasks use the format: ### Task N: Title")

    return tasks


def _split_tasks(text: str) -> List[str]:
    """Split markdown text into per-task blocks."""
    # Pattern matches various task heading styles
    pattern = r'(?:^|\n)(?:#{1,4}\s*)?(?:Task\s*)?(\d+)[\.:：]\s*[^\n]*'

    # Find all task heading positions
    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    if not matches:
        return []

    blocks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)

    return blocks


def _parse_task_block(block: str) -> Optional[Task]:
    """Parse a single task block into a Task object."""

    # Extract task number and title
    heading_match = re.match(
        r'(?:#{1,4}\s*)?(?:Task\s*)?(\d+)[\.:：]\s*(.+)',
        block.strip(), re.IGNORECASE
    )
    if not heading_match:
        return None

    number = int(heading_match.group(1))
    title = heading_match.group(2).strip()
    # Remove trailing markdown heading markers
    title = re.sub(r'\s*#+\s*$', '', title)

    # Extract agent
    agent_match = re.search(r'负责[：:]\s*(\w+)', block)
    if not agent_match:
        return None
    agent = agent_match.group(1).strip()

    # Extract description
    desc_match = re.search(r'描述[：:]\s*(.+?)(?=\n\s*(?:-|产出|$))', block, re.DOTALL)
    description = ""
    if desc_match:
        # Handle bullet-point continuation lines
        desc_lines = []
        for line in desc_match.group(1).strip().split('\n'):
            stripped = line.strip()
            if stripped:
                desc_lines.append(stripped)
        description = ' '.join(desc_lines)

    # Extract expected output
    output_match = re.search(r'产出[：:]\s*(.+?)(?=\n\s*(?:###|##|#|\Z))', block, re.DOTALL)
    expected_output = ""
    if output_match:
        expected_output = output_match.group(1).strip()

    return Task(
        number=number,
        title=title,
        agent=agent,
        description=description,
        expected_output=expected_output,
        raw_text=block,
    )
