"""Syntax gate for the figure-builder edit campaign.

Rebuilds paper1_figures.ipynb from _build_figures_nb.py and AST-parses every
emitted code cell, so a broken edit is caught in seconds instead of eight hours
into a Slurm run. Does NOT execute any cell and touches no data on ceph.

    /mnt/home/mlee1/venvs/BIND_env/bin/python _check_builder.py

Exits non-zero (and prints the offending cell + line) on any failure.
"""
import ast
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "paper1_figures.ipynb"


def main() -> int:
    r = subprocess.run([sys.executable, "_build_figures_nb.py"], cwd=HERE,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILDER FAILED TO RUN:\n" + r.stdout + r.stderr)
        return 1

    import nbformat

    nb = nbformat.read(NB, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    bad = 0
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        src = c.source
        # strip IPython line/cell magics, which are not valid Python
        clean = "\n".join("" if s.lstrip().startswith(("%", "!")) else s
                          for s in src.splitlines())
        try:
            ast.parse(clean)
        except SyntaxError as e:
            bad += 1
            print(f"SYNTAX ERROR in notebook cell index {i} (line {e.lineno}): {e.msg}")
            for n, line in enumerate(clean.splitlines()[max(0, e.lineno - 4):e.lineno + 3],
                                     start=max(1, e.lineno - 3)):
                print(f"   {n:4d}| {line}")

    saves = sorted({ln.split('"')[1] for c in code_cells for ln in c.source.splitlines()
                    if "save(fig, " in ln and '"' in ln})
    print(f"\n{len(nb.cells)} cells ({len(code_cells)} code), {bad} with syntax errors")
    print(f"{len(saves)} figures written by this builder:")
    for s in saves:
        print(f"   {s}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
