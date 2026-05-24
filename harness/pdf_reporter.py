"""PDF report generator — converts task outputs to PDF reports."""

import re
import html
from pathlib import Path
from datetime import datetime


class PDFReporter:
    """Generate PDF reports from task outputs using markdown + weasyprint.

    Each task gets its own PDF report with consistent formatting,
    including task metadata, the agent output, and a cover page.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_task_report(self, task_id: int, title: str, agent: str,
                             output: str, approved: bool = False,
                             sandbox_log: str = "") -> str:
        """Generate a PDF report for a single task.

        Args:
            task_id: Task number.
            title: Task title.
            agent: Agent name.
            output: Agent's markdown output.
            approved: Whether the task was approved.
            sandbox_log: Sandbox execution log (for Executor tasks).

        Returns:
            Path to the generated PDF file.
        """
        import markdown
        from weasyprint import HTML

        # Convert markdown to HTML
        md = markdown.Markdown(extensions=['extra', 'codehilite', 'tables', 'sane_lists'])
        body_html = md.convert(output)

        # Escape sandbox log
        sandbox_html = html.escape(sandbox_log) if sandbox_log else ""

        # Build the full HTML document
        full_html = self._build_html(
            task_id=task_id,
            title=title,
            agent=agent,
            approved=approved,
            body_html=body_html,
            sandbox_log=sandbox_html,
        )

        # Generate PDF
        pdf_path = self.output_dir / f"task_{task_id:02d}_{self._slugify(title)}.pdf"
        HTML(string=full_html).write_pdf(str(pdf_path))
        return str(pdf_path)

    def generate_final_paper(self, markdown_content: str, title: str = "Final Paper") -> str:
        """Generate the final assembled paper as PDF.

        Args:
            markdown_content: The full paper in markdown format.
            title: Paper title.

        Returns:
            Path to the generated PDF file.
        """
        import markdown
        from weasyprint import HTML

        md = markdown.Markdown(extensions=['extra', 'codehilite', 'tables', 'sane_lists'])
        body_html = md.convert(markdown_content)

        full_html = self._build_paper_html(title, body_html)
        pdf_path = self.output_dir / "final_paper.pdf"
        HTML(string=full_html).write_pdf(str(pdf_path))
        return str(pdf_path)

    def _build_html(self, task_id: int, title: str, agent: str,
                    approved: bool, body_html: str, sandbox_log: str) -> str:
        status_badge = "APPROVED" if approved else "PENDING REVIEW"
        status_color = "#2e7d32" if approved else "#e65100"
        sandbox_section = ""
        if sandbox_log:
            sandbox_section = f"""
            <div class="sandbox-section">
                <h2>Sandbox Execution Log</h2>
                <pre class="sandbox-log">{sandbox_log}</pre>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Task {task_id}: {html.escape(title)}</title>
<style>
  @page {{ size: A4; margin: 2.5cm 2cm; @top-center {{ content: "AI4Sci Harness — Task Report"; font-size: 9pt; color: #888; }} @bottom-center {{ content: "Page " counter(page); font-size: 9pt; color: #888; }} }}
  body {{ font-family: "DejaVu Serif", "Linux Libertine", "Times New Roman", serif; font-size: 11pt; line-height: 1.6; color: #222; }}
  .cover {{ text-align: center; padding: 80px 0 40px 0; border-bottom: 3px double #333; margin-bottom: 40px; }}
  .cover h1 {{ font-size: 22pt; margin-bottom: 10px; }}
  .cover .meta {{ font-size: 11pt; color: #666; }}
  .cover .badge {{ display: inline-block; padding: 4px 16px; border-radius: 4px; color: #fff; font-weight: bold; font-size: 10pt; background: {status_color}; margin-top: 12px; }}
  h2 {{ font-size: 15pt; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  h3 {{ font-size: 13pt; margin-top: 22px; }}
  pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; font-size: 9pt; overflow-x: auto; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 2px; font-size: 9.5pt; }}
  pre code {{ background: none; padding: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  .math.display {{ display: block; text-align: center; margin: 12px 0; }}
  .sandbox-section {{ margin-top: 30px; border-top: 2px solid #ddd; padding-top: 20px; }}
  .sandbox-log {{ font-size: 8.5pt; max-height: 400px; overflow-y: auto; }}
  img {{ max-width: 100%; height: auto; }}
</style>
<!-- MathJax for LaTeX rendering -->
<script>
window.MathJax = {{ tex: {{ inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']] }} }};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
<div class="cover">
  <h1>Task {task_id}: {html.escape(title)}</h1>
  <div class="meta">Agent: {html.escape(agent)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  <div class="meta">Project: AI4Sci Harness</div>
  <div class="badge">{status_badge}</div>
</div>
{body_html}
{sandbox_section}
</body>
</html>"""

    def _build_paper_html(self, title: str, body_html: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  @page {{ size: A4; margin: 2.5cm 2cm; @top-center {{ content: "AI4Sci Harness — Final Paper"; font-size: 9pt; color: #888; }} @bottom-center {{ content: "Page " counter(page); font-size: 9pt; color: #888; }} }}
  body {{ font-family: "DejaVu Serif", "Linux Libertine", "Times New Roman", serif; font-size: 11pt; line-height: 1.6; color: #222; }}
  .cover {{ text-align: center; padding: 100px 0 50px 0; border-bottom: 3px double #333; margin-bottom: 40px; }}
  .cover h1 {{ font-size: 24pt; margin-bottom: 20px; }}
  .cover .meta {{ font-size: 12pt; color: #666; }}
  h2 {{ font-size: 15pt; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  h3 {{ font-size: 13pt; margin-top: 24px; }}
  pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; font-size: 9pt; overflow-x: auto; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 2px; font-size: 9.5pt; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  .abstract {{ background: #f9f9f9; padding: 16px 20px; margin: 20px 0; border-left: 4px solid #333; font-style: italic; }}
  .section {{ margin-top: 15px; }}
  img {{ max-width: 100%; height: auto; }}
</style>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']] }} }};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
<div class="cover">
  <h1>{html.escape(title)}</h1>
  <div class="meta">Generated by AI4Sci Harness — 5-Node Multi-Agent Research Framework</div>
  <div class="meta">{datetime.now().strftime('%B %d, %Y')}</div>
</div>
{body_html}
</body>
</html>"""

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert title to filename-safe slug."""
        text = re.sub(r'[^\w\s-]', '', text.lower())
        text = re.sub(r'[-\s]+', '_', text)
        return text[:60]
