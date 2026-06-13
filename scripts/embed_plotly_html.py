"""Embed static-HTML renderings into notebook Plotly outputs.

Plotly figures displayed in a notebook are stored only as the
``application/vnd.plotly.v1+json`` MIME bundle. That bundle renders in
JupyterLab/VS Code (which ship a Plotly renderer) but shows up as
"Data type cannot be displayed: application/vnd.plotly.v1+json" in the static
HTML produced by nbsphinx/nbconvert for the documentation site.

This script walks the example notebooks and, for every output that carries a
Plotly MIME bundle but no ``text/html`` representation, reconstructs the figure
from the stored ``data``/``layout`` and adds a self-contained ``text/html``
rendering (plotly.js pulled from the CDN). The original Plotly MIME bundle is
kept, so notebooks still render natively/interactively inside Jupyter/VS Code,
while nbsphinx renders the ``text/html`` version (which it prefers).

Run from the repository root::

    python scripts/embed_plotly_html.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import plotly.graph_objects as go
import plotly.io as pio

PLOTLY_MIME = "application/vnd.plotly.v1+json"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _as_str(value: object) -> str:
    """Notebook output values may be stored as a list of lines or a string."""
    if isinstance(value, list):
        return "".join(value)
    return str(value)


def embed_html_in_output(output: nbformat.NotebookNode) -> bool:
    """Add a ``text/html`` rendering to a single output. Return True if changed."""
    data = getattr(output, "data", None)
    if not data or PLOTLY_MIME not in data:
        return False
    if "text/html" in data:
        return False

    bundle = data[PLOTLY_MIME]
    fig = go.Figure(data=bundle.get("data", []), layout=bundle.get("layout", {}))
    html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        default_width="100%",
    )
    data["text/html"] = html
    return True


def process_notebook(path: Path) -> int:
    """Convert a notebook in place. Return the number of outputs updated."""
    nb = nbformat.read(path, as_version=4)
    changed = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if embed_html_in_output(output):
                changed += 1
    if changed:
        nbformat.write(nb, path)
    return changed


def main() -> None:
    total = 0
    for path in sorted(EXAMPLES_DIR.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        updated = process_notebook(path)
        if updated:
            total += updated
            print(f"{path.relative_to(EXAMPLES_DIR.parent)}: embedded {updated} figure(s)")
    print(f"Done. Embedded HTML into {total} Plotly output(s).")


if __name__ == "__main__":
    main()
