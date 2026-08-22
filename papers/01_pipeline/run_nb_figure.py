"""Render ONE figure from ``_build_figures_nb.py`` without the notebook.

The builder assembles its notebook as a list of ``(kind, source)`` cells, and a
figure cell depends on state built by earlier cells.  Hand-extracting a cell
therefore risks silently dropping a dependency and producing a figure that
differs from the shipped one.  This runner instead executes the cells IN ORDER
in one namespace -- exactly the notebook's own semantics -- with two changes:

* ``paper_style.save`` is suppressed for every stem except the target, so no
  other figure is overwritten;
* a cell that raises is reported and skipped rather than aborting, so a figure
  whose inputs do not exist for this campaign (per-halo caches, say) cannot
  block an unrelated target.  If the TARGET cell raises, that is fatal.

    BIND_CAMPAIGN=n1000 python3 run_nb_figure.py --stem fig05a_sl_response
    python3 run_nb_figure.py --list
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_tools"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", help="figure stem, e.g. fig05a_sl_response")
    ap.add_argument("--list", action="store_true", help="list every figure stem")
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first failing cell instead of skipping")
    a = ap.parse_args()

    import _build_figures_nb as B
    cells = [(k, s) for k, s in B.CELLS if k == "code"]

    stem_of: dict[str, int] = {}
    for idx, (_, src) in enumerate(cells):
        for m in re.finditer(r'save\(\s*\w+\s*,\s*"([^"]+)"', src):
            stem_of.setdefault(Path(m.group(1)).name, idx)
    if a.list:
        for s, i in sorted(stem_of.items(), key=lambda kv: kv[1]):
            print(f"  cell {i:3d}  {s}")
        return
    if a.stem not in stem_of:
        raise SystemExit(f"stem {a.stem!r} not found; --list to see them all")
    target = stem_of[a.stem]

    import paper_style
    _real_save = paper_style.save
    want = a.stem

    import os
    _tag = "_n1000" if os.environ.get("BIND_CAMPAIGN", "sci50") == "n1000" else ""

    def _save(fig, stem, *args, **kw):
        if Path(stem).name != want:
            return None
        # tag the campaign into the filename: the notebook builder writes to a
        # bare figs_v2/<stem>, so without this an n1000 render silently
        # overwrites the 50-real staging copy of the same figure.
        return _real_save(fig, f"{stem}{_tag}", *args, **kw)

    paper_style.save = _save
    def _py(src: str) -> str:
        """Drop Jupyter magics / shell escapes -- valid in a notebook, not in Python."""
        return "\n".join("" if ln.strip().startswith(("%", "!")) else ln
                          for ln in src.splitlines())

    ns: dict = {"__name__": "__nbfig__", "save": _save}
    failed = []
    for i in range(target + 1):
        try:
            exec(compile(_py(cells[i][1]), f"<cell {i}>", "exec"), ns)     # noqa: S102
        except Exception:                                             # noqa: BLE001
            if i == target or a.strict:
                print(f"\n=== TARGET cell {i} ({want}) FAILED ===", flush=True)
                traceback.print_exc()
                raise SystemExit(1) from None
            failed.append(i)
            last = traceback.format_exc().strip().splitlines()[-1]
            print(f"  [skip] cell {i}: {last}", flush=True)
        ns["save"] = _save        # later cells may rebind save from an import
    print(f"\nrendered {want} (cell {target}); {len(failed)} earlier cell(s) skipped: {failed}")


if __name__ == "__main__":
    main()
