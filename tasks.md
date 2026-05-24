# Your Project Title

> AI4Sci 5-Node Multi-Agent Research Collaboration Framework
> Core question: Replace with your research question.

---

### Task 1: Literature Survey and Theoretical Background
- 负责：Researcher
- 描述：Systematically survey the latest literature in your research domain. Identify key theoretical frameworks, experimental constraints, and open questions. Extract essential equations (LaTeX format) and define the parameter space.
- 产出：Literature summary with key citation threads + core equation list (LaTeX) + parameter space definition

### Task 2: Theoretical Framework and Simulation Strategy
- 负责：Planner
- 描述：Based on Task 1's literature survey, develop a complete analysis strategy. Define the governing equations, initial conditions, parameter scan strategy, and write pseudocode. Include enhanced tool usage planning and compliance notes if applicable.
- 产出：Complete governing equations (LaTeX) + initial condition definitions + parameter scan strategy table + complete pseudocode + enhanced tool usage plan

### Task 3: Core Numerical Simulation
- 负责：Executor
- 描述：Implement the numerical simulation code based on Task 2's strategy. The code must be self-contained Python using scipy/numpy. Generate professional figures (save to sandbox/ directory) and output numerical results tables.
- 产出：Complete executable Python code + professional figures (PNG, >=150 dpi) + numerical results tables (Markdown) + analysis summary

### Task 4: Code Review and Consistency Verification
- 负责：Critic
- 描述：Comprehensively review Task 3's code and numerical results. Check: (1) code correctness — self-contained, executable, no syntax errors; (2) scientific correctness — governing equations implemented correctly, boundary conditions properly applied; (3) enhanced tool compliance if applicable; (4) figure quality. Issue verdict: APPROVED or REJECTED with specific corrections.
- 产出：Detailed review report with checklist + per-item verdict + final conclusion

### Task 5: Analysis and Physical Discussion
- 负责：Writer
- 描述：Based on Task 3's numerical results and Task 4's review, write a complete analysis chapter. Cover: key findings, parameter dependence, comparison with existing work, physical interpretation, and experimental implications.
- 产出：Analysis chapter (Markdown + LaTeX, ~2000-3000 words) + physical conclusions summary + key comparison tables

### Task 6: Paper Integration and Final Draft
- 负责：Writer
- 描述：Integrate all prior task outputs into a complete paper draft. Structure: (1) Title; (2) Abstract (~200 words); (3) Introduction; (4) Theoretical Framework; (5) Results; (6) Discussion; (7) Conclusion; (8) References. Format: journal-appropriate style, all equations in LaTeX, all figures referenced from Task 3 outputs.
- 产出：Complete paper draft (LaTeX source + Markdown version) + compilation-ready directory structure
