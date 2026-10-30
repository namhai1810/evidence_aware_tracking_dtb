#!/usr/bin/env python3
"""Convert a research Markdown report to a PDF with tables, figures, Mermaid flowcharts, and math.

Requirements:
  pip install weasyprint graphviz matplotlib
  Ubuntu/Debian: sudo apt-get install pandoc graphviz
  macOS: brew install pandoc graphviz

Usage:
  python convert_md_to_pdf.py report.md
  python convert_md_to_pdf.py report.md report.pdf

All image paths in Markdown are resolved relative to the original .md file.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from graphviz import Digraph
from weasyprint import HTML, CSS


STYLES = """
@page {
    size: A4 landscape;
    margin: 15mm;
    @bottom-right { content: counter(page); color: #667085; font-size: 9pt; }
}
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.5pt; color: #182230; line-height: 1.43; }
h1 { color: #12345b; font-size: 21pt; margin: 0 0 12pt; }
h2 { color: #12345b; font-size: 15pt; margin: 18pt 0 7pt; border-bottom: 1px solid #d4dce8; }
h3 { color: #184772; font-size: 11.5pt; margin: 12pt 0 6pt; }
h1, h2, h3, h4 { break-after: avoid; }
p { margin: 3pt 0 7pt; }
a { color: #1767a4; text-decoration: none; }
table { border-collapse: collapse; width: 100%; margin: 9pt 0 13pt; font-size: 8pt; }
th, td { border: 1px solid #cad5e3; padding: 5pt 6pt; overflow-wrap: anywhere; vertical-align: top; }
th { background-color: #eaf1f8; font-weight: 700; }
tr:nth-child(even) td { background: #f8fafc; }
tr { break-inside: avoid; }
thead { display: table-header-group; }
img { max-width: 100%; max-height: 150mm; height: auto; object-fit: contain; }
p:has(> img) { text-align: center; break-inside: avoid; margin-top: 12pt; margin-bottom: 3pt; }
p:has(> img) + p > em { color: #48556a; font-size: 8.5pt; }
pre { white-space: pre-wrap; background: #f1f5f9; padding: 8pt; border-radius: 4pt; font-size: 8pt; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: .9em; }
blockquote { border-left: 3px solid #96adcb; padding-left: 9pt; color: #475569; }
.math { text-align: center; margin: 10pt 0; }
"""


def render_mermaid_as_svg(source: str, svg_path: Path) -> None:
    """Render the simple Mermaid flowchart syntax used by this report via Graphviz."""
    lines = [s.strip() for s in source.splitlines() if s.strip()]
    direction = 'LR' if lines[0].lower().startswith('flowchart lr') else 'TB'
    graph = Digraph('workflow', format='svg')
    graph.attr(rankdir=direction, bgcolor='white', pad='0.2', nodesep='0.32', ranksep='0.45')
    graph.attr('node', shape='box', style='rounded,filled', fillcolor='#eaf1fa', color='#8aa8cb',
               fontname='DejaVu Sans', fontsize='11', margin='0.15,0.10')
    graph.attr('edge', color='#537b9f', arrowsize='0.6')
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    for line in lines[1:]:
        for node_id, label in re.findall(r'\b([A-Za-z_]\w*)\[([^\]]+)\]', line):
            nodes[node_id] = label
        match = re.search(r'\b([A-Za-z_]\w*)(?:\[[^\]]+\])?\s*-->\s*([A-Za-z_]\w*)', line)
        if match:
            edges.append((match.group(1), match.group(2)))
        elif '-->' in line:
            raise ValueError(f'Unsupported Mermaid edge syntax: {line}')
    if not edges:
        raise ValueError('No arrows found in Mermaid diagram')
    for node_id, label in nodes.items():
        graph.node(node_id, label)
    for start, end in edges:
        graph.edge(start, end)
    svg_path.write_bytes(graph.pipe(format='svg'))


def render_latex_as_svg(source: str, svg_path: Path) -> None:
    """Render LaTeX display math to a vector figure without requiring TeX installation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 0.62))
    fig.text(0.5, 0.5, '$' + source.strip() + '$', ha='center', va='center', fontsize=15)
    fig.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.06, transparent=True)
    plt.close(fig)


def check_images(markdown_text: str, md_dir: Path) -> None:
    missing = []
    for filename in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', markdown_text):
        filename = filename.strip().split(' "', 1)[0].strip('<>')
        if '://' not in filename and not (md_dir / filename).is_file():
            missing.append(filename)
    if missing:
        listed = '\n'.join('  - ' + item for item in missing[:20])
        raise SystemExit(
            f'Missing {len(missing)} referenced figure(s) relative to {md_dir}:\n{listed}\n'
            'Copy the report_assets/ and experiment figure folders next to the .md, '
            'or fix their relative image paths before conversion.'
        )


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit('Usage: python convert_md_to_pdf.py input.md [output.pdf]')
    md_path = Path(sys.argv[1]).expanduser().resolve()
    if not md_path.is_file():
        raise SystemExit(f'Markdown file not found: {md_path}')
    pdf_path = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) == 3 else md_path.with_suffix('.pdf')
    for executable in ('pandoc', 'dot'):
        if not shutil.which(executable):
            raise SystemExit(f'Missing {executable}. Please install pandoc and graphviz.')

    raw = md_path.read_text(encoding='utf-8')
    check_images(raw, md_path.parent)
    with tempfile.TemporaryDirectory(prefix='md_pdf_') as temp_dir:
        temp_dir = Path(temp_dir)
        index = [0]

        def replace_flow(match: re.Match[str]) -> str:
            index[0] += 1
            svg = temp_dir / f'workflow_{index[0]}.svg'
            render_mermaid_as_svg(match.group(1), svg)
            return f'\n![Workflow {index[0]}]({svg.as_uri()})\n'

        converted_md = re.sub(r'(?ms)^```mermaid\s*\n(.*?)^```\s*$', replace_flow, raw)

        def replace_equation(match: re.Match[str]) -> str:
            index[0] += 1
            svg = temp_dir / f'equation_{index[0]}.svg'
            render_latex_as_svg(match.group(1), svg)
            return f'\n![]({svg.as_uri()})\n'

        converted_md = re.sub(r'(?s)\\\[(.*?)\\\]', replace_equation, converted_md)
        pandoc = subprocess.run(
            ['pandoc', '--from=markdown+tex_math_single_backslash', '--to=html5', '--mathml', '--wrap=none'],
            input=converted_md, text=True, cwd=md_path.parent,
            capture_output=True, check=True,
        )
        html = '<!doctype html><html><meta charset="utf-8"><body>' + pandoc.stdout + '</body></html>'
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html, base_url=md_path.parent.as_uri() + '/').write_pdf(
            str(pdf_path), stylesheets=[CSS(string=STYLES)]
        )
    print(f'PDF created: {pdf_path}')


if __name__ == '__main__':
    main()
