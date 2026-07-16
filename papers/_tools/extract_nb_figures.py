#!/usr/bin/env python3
"""Extract embedded PNG outputs from a Jupyter notebook.

Usage: python3 extract_nb_figures.py NOTEBOOK.ipynb OUTDIR

Writes OUTDIR/cell{NNN}_out{K}.png for every image/png output, plus
OUTDIR/manifest.json recording, per image: the producing cell index, the most
recent markdown heading above it, and the head of the cell's source — the
provenance needed to write an accurate caption.
"""
import base64
import json
import pathlib
import sys


def main() -> None:
    nb_path, outdir = sys.argv[1], pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    nb = json.load(open(nb_path, encoding="utf-8"))
    heading = ""
    manifest = []
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            for line in src.splitlines():
                if line.startswith("#"):
                    heading = line.lstrip("# ").strip()
                    break
            continue
        for k, out in enumerate(cell.get("outputs", [])):
            data = out.get("data", {})
            if "image/png" in data:
                name = f"cell{i:03d}_out{k}.png"
                (outdir / name).write_bytes(base64.b64decode(data["image/png"]))
                manifest.append(
                    {
                        "file": name,
                        "cell": i,
                        "heading": heading,
                        "source_head": src.strip().splitlines()[:6],
                    }
                )
    json.dump(manifest, open(outdir / "manifest.json", "w"), indent=1)
    print(f"{len(manifest)} images -> {outdir}")


if __name__ == "__main__":
    main()
