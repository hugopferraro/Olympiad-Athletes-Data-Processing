"""Converte o script com marcadores ``# %%`` em um Notebook executável."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tratamento_atletas_olimpicos.py"
DESTINATION = ROOT / "tratamento_atletas_olimpicos.ipynb"
CELL_MARKER = re.compile(r"^# %%(?: \[(markdown)\])?\s*$")


def parse_cells(source: str) -> list[tuple[str, str]]:
    """Separa células mantendo o código-fonte exatamente como foi escrito."""
    cells: list[tuple[str, str]] = []
    cell_type: str | None = None
    lines: list[str] = []

    def append_cell() -> None:
        if cell_type is None:
            return
        content = "\n".join(lines).rstrip()
        if cell_type == "markdown":
            markdown_lines = []
            for line in lines:
                if line in {"", "#"}:
                    markdown_lines.append("")
                elif line.startswith("# "):
                    markdown_lines.append(line[2:])
                else:
                    raise ValueError(f"Linha Markdown sem prefixo '#': {line!r}")
            content = "\n".join(markdown_lines).rstrip()
        cells.append((cell_type, content))

    for line in source.splitlines():
        marker = CELL_MARKER.match(line)
        if marker:
            append_cell()
            cell_type = "markdown" if marker.group(1) else "code"
            lines = []
        else:
            lines.append(line)
    append_cell()

    if not cells or any(not content for _, content in cells):
        raise ValueError("O script gerou uma célula vazia ou nenhuma célula.")
    return cells


def build_notebook() -> nbformat.NotebookNode:
    """Cria o notebook e comprova a paridade de todas as células de código."""
    parsed = parse_cells(SOURCE.read_text(encoding="utf-8"))
    notebook_cells = [
        nbformat.v4.new_markdown_cell(content)
        if cell_type == "markdown"
        else nbformat.v4.new_code_cell(content)
        for cell_type, content in parsed
    ]
    notebook = nbformat.v4.new_notebook(
        cells=notebook_cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )

    source_code = [content for cell_type, content in parsed if cell_type == "code"]
    notebook_code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert notebook_code == source_code, "O código do Notebook divergiu do script."
    return notebook


def main() -> None:
    notebook = build_notebook()
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, DESTINATION)

    error_outputs = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not error_outputs, "O Notebook executado contém erros."
    print(f"Notebook criado, executado e validado: {DESTINATION}")


if __name__ == "__main__":
    main()
