# AI4Sci Harness

5-Node Multi-Agent Research Collaboration Framework. A reusable template for running structured, multi-agent scientific research pipelines.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai4sci-harness.git
cd ai4sci-harness

# 2. Install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env — fill in your actual API keys

# 4. Edit config.yaml — set your project name and description
# 5. Edit tasks.md  — define your research tasks (6 tasks by default)

# 6. Run the pipeline
.venv/bin/python3 run_pipeline.py
```

API keys are loaded from `.env` (git-ignored). You can also set them as environment variables directly.

## Project Structure

```
ai4sci-harness/
├── config.yaml          # Project config (agents, sandbox, LLM backend, tools)
├── .env.example         # API key template (copy to .env and fill in)
├── requirements.txt     # Python dependencies
├── tasks.md             # Task definitions (parsed by the harness)
├── run_pipeline.py      # Batch pipeline runner (auto-approve mode)
├── harness/             # Framework core
│   ├── agents/          # 5 agent types: Researcher, Planner, Executor, Critic, Writer
│   ├── backends/        # LLM backends: Anthropic, DeepSeek, MiMo, Mock
│   ├── config.py        # Config loader
│   ├── task_parser.py   # tasks.md parser
│   ├── context.py       # Persistent context across tasks
│   ├── sandbox.py       # Isolated Python code execution
│   ├── critic_loop.py   # Auto-correction loop (Executor → Sandbox → Critic)
│   ├── orchestrator.py  # Main flow control
│   ├── exporter.py      # Paper export (LaTeX + Markdown)
│   ├── pdf_reporter.py  # PDF report generation
│   ├── human_gate.py    # Human-in-the-loop approval
│   └── cli.py           # CLI entry point
├── sandbox/             # Executor outputs (code, figures)
├── outputs/             # Task context (context.json)
├── paper/               # Exported paper (paper.tex, paper.md)
└── reports/             # Per-task PDF reports + final paper
```

## CLI Usage

```bash
# Run all tasks interactively
.venv/bin/python3 -m harness run

# Run a specific task
.venv/bin/python3 -m harness run --task 3

# Start from a specific task
.venv/bin/python3 -m harness run --from 4

# Check status
.venv/bin/python3 -m harness status

# Export paper
.venv/bin/python3 -m harness export --format both

# Initialize a new project directory
.venv/bin/python3 -m harness init --name "My Project" --dir ../my-project
```

## Agent Roles

| Agent | Role | Can Do | Cannot Do |
|-------|------|--------|-----------|
| **Researcher** | Literature survey & theory | Survey, extract equations, define parameters | Compute, code, derive |
| **Planner** | Strategy & pseudocode | Plan, design, decompose, write pseudocode | Write runnable code, compute |
| **Executor** | Numerical simulation | Write code, run simulations, generate figures | Use external results as final |
| **Critic** | Code review & verification | Verify, validate, audit | Modify code, run code |
| **Writer** | Paper writing | Write, compose, integrate, draft | Compute, generate new figures |

## Multi-Model Routing

Each agent type can use a different LLM backend. Configure in `agent_backend_routing`:

| Agent | Default Backend | Model |
|-------|----------------|-------|
| Researcher | MiMo | mimo-v2.5-pro |
| Planner | DeepSeek | deepseek-v4-pro |
| Executor | MiMo | mimo-v2.5-pro |
| Critic | DeepSeek | deepseek-v4-pro |
| Writer | DeepSeek | deepseek-v4-pro |

Supported backends: `anthropic`, `deepseek`, `mimo`, `mock`

To override all agents to a single backend, use `--backend mock` on the CLI.

## Customization

### config.yaml

- `project.name` — Your project title
- `project.description` — Core research question
- `agents.*` — Agent roles, capabilities, and constraints
- `agent_backend_routing.*` — Per-agent backend, model, API key, base URL
- `sandbox.*` — Allowed/blocked Python imports, timeout, output limits
- `enhanced_tools` — External tool compliance rules (commented out by default)
- `llm.*` — Default fallback backend if agent routing is not set

### tasks.md

Each task must follow this format:

```markdown
### Task N: Title
- 负责：AgentName
- 描述：What this task should accomplish
- 产出：Expected deliverables
```

The parser expects `负责`, `描述`, and `产出` fields. Agent names map to: Researcher, Planner, Executor, Critic, Writer.

## Dependencies

The `.venv/` includes: anthropic, openai, pyyaml, weasyprint, markdown, numpy, scipy, matplotlib, torch, pandas.

To recreate:

```bash
python3 -m venv .venv
.venv/bin/pip install anthropic openai pyyaml weasyprint markdown numpy scipy matplotlib torch pandas
```
