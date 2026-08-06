"""Execute ONLY selected figure cells of paper1_figures.ipynb, for a fast preview.

Motivation: the full notebook needs a GPU node (~1 h via run_figures.sbatch). When only a
few figures have changed, this runs the shared setup cell plus the named figure cells on
whatever machine you are on, writing the same figs_v2/*.pdf + figs_preview/*.png that the
full run would.

    /mnt/home/mlee1/venvs/BIND_env/bin/python _run_subset.py fig00_hero fig03_halo_validation ...

Each requested figure's cell must be self-contained given the setup cell (cell 0). That is
true for the map/atlas/profile figures; it is NOT true for the emulator figures, which need
the fit performed in the fig09 cell. Those are rejected up front rather than failing late.

Reads the builder from a snapshot so a concurrent edit cannot produce a half-written cell:
the builder is copied, compiled, and only then executed.
"""
import ast
import py_compile
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE / "_build_figures_nb.py"

# figures whose cell depends on state built by an EARLIER figure cell
NEEDS_EMULATOR = {"fig09_emulator_validation", "fig09b_emulator_heads",
                  "fig09c_emulator_curves", "fig15_emulator_calibration",
                  "fig15b_ood_validation", "fig10_astro_corner",
                  "fig16_inference_robustness"}

# Cheap cross-cell dependencies, pulled in automatically. These are real: the
# notebook is executed top to bottom, so a cell may legitimately use a name bound
# by an earlier figure's cell. That is fine for the full run and invisible until
# you try to execute a cell on its own -- hence this map.
#   fig20  reads OB_OM (the cosmic baryon fraction), bound in the fig03 cell.
#   fig04b reads ell_f / f_ell / NR / paired() / KYb / KYt / ky_b (per-real, for
#   the measured survey band) from the fig04 cell.
#   fig08  reads pr (paired_perreal_fid) for the peak-count noise floor, from fig04.
#   fig04n is self-contained (no dep) -- it loads its own nu05n shards directly.
#   fig07n (2026-08-05, guarded noisy twin of fig 7) reuses fig 7's own
#   STATS/FID/response_ratio/nu_solid_drawn/color_curves rather than
#   re-deriving them, and fig 7 itself needs fig05a's STATS/Spearman-grid
#   source of truth -- so fig07n needs BOTH ancestors listed explicitly: this
#   resolver is ONE level only (not transitive/recursive), so fig05a would
#   otherwise be silently dropped if only fig07_covariation were listed here.
# 2026-08-05: fig20 split into fig20a/fig20b/fig20c -- all three saves live in
# ONE cell (any one name executes it and writes all three PDFs; requesting
# several is deduped below), and that cell still reads OB_OM from fig03's.
DEPS = {"fig20a_vandaalen_matrix": ["fig03_halo_validation"],
        "fig20b_hinge_gas": ["fig03_halo_validation"],
        "fig20c_vandaalen_plane": ["fig03_halo_validation"],
        "fig20d_budget_partition": ["fig03_halo_validation"],
        "fig20e_analytic_model": ["fig03_halo_validation"],
        "fig20f_redshift_thermal": ["fig03_halo_validation"],
        "fig20g_latent_kernels": ["fig03_halo_validation"],
        "fig20h_theta_to_latents": ["fig03_halo_validation"],
        "fig20i_latent_corner": ["fig03_halo_validation"],
        # paper compositions (2026-08-06 figure plan) -- same fig20 cell
        "pfig_s3c_hinge_plane": ["fig03_halo_validation"],
        "pfig_s4a_cv_buildup": ["fig03_halo_validation"],
        "pfig_s4b_reconstruction": ["fig03_halo_validation"],
        "pfig_s4b_generality": ["fig03_halo_validation"],
        "pfig_s4b_app_zs": ["fig03_halo_validation"],
        "pfig_s4b_app_epoch": ["fig03_halo_validation"],
        "pfig_s4b_model_curves": ["fig03_halo_validation"],
        # fig03a reuses fa/ta/logM/edges/SCT/TAIL/MLAB/PIV3 (and fg_f/fg_t for
        # the comparison print) from the fig03 cell.
        "fig03a_halo_raw": ["fig03_halo_validation"],
        "fig04b_spectra": ["fig04_field_validation"],
        "fig08_latent_pca": ["fig04_field_validation"],
        # the fig05a cell is now the single source of truth for the canonical
        # STATS table + Spearman grids (Ys/Rs/IMPg/col_order): fig05 consumes
        # them for the collapsed importance map, fig07 for STATS + IMPg.
        "fig05_param_response": ["fig05a_sl_response"],
        "fig07_covariation": ["fig05a_sl_response"],
        "fig07n_covariation_noisy": ["fig05a_sl_response", "fig07_covariation"],
        # fig23 reuses cl_relerr, defined in the fig04 cell.
        "fig23_s3_opener": ["fig04_field_validation"]}

# figures that are GUARDED on a cache that does not exist yet (2026-08-05 noisy-
# twin campaign): they compile and, if picked, execute their guard branch (which
# prints "... cache pending -- cell skipped" and saves nothing) rather than
# producing a PDF. Not a bug in this runner -- listed here so a caller checking
# for the expected *.pdf after a subset run knows why it is legitimately absent.
GUARDED_PENDING = {"fig04n_field_validation_noisy", "fig07n_covariation_noisy"}


def snapshot_builder(tmp: Path, tries: int = 10) -> Path:
    """Copy the builder somewhere stable, retrying while an agent is mid-edit."""
    dst = tmp / "_builder_snapshot.py"
    for i in range(tries):
        shutil.copy2(BUILDER, dst)
        try:
            py_compile.compile(str(dst), doraise=True, cfile=str(tmp / "x.pyc"))
            return dst
        except py_compile.PyCompileError as e:
            print(f"  builder mid-edit (attempt {i+1}/{tries}): {str(e).splitlines()[-1][:90]}")
            time.sleep(20)
    raise SystemExit("builder never compiled cleanly — is an edit agent stuck?")


def main(wanted: list[str]) -> int:
    bad = sorted(set(wanted) & NEEDS_EMULATOR)
    if bad:
        print(f"REFUSING {bad}: these cells depend on the emulator fit performed in the "
              f"fig09 cell, so they cannot run standalone. Use run_figures.sbatch.")
        return 2

    import nbformat
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        snap = snapshot_builder(tmp)
        # build into the temp dir so papers/01_pipeline/paper1_figures.ipynb is untouched
        r = subprocess.run([sys.executable, str(snap)], cwd=tmp, capture_output=True, text=True)
        if r.returncode != 0:
            print("builder failed:\n" + r.stdout + r.stderr)
            return 1
        nb = nbformat.read(tmp / "paper1_figures.ipynb", as_version=4)

        code = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
        setup = code[0][1]

        # pull in cheap prerequisites, keeping notebook order
        need = list(wanted)
        for name in wanted:
            for dep in DEPS.get(name, []):
                if dep not in need:
                    print(f"  + {dep} (prerequisite of {name})")
                    need.append(dep)

        def _order(nm):
            for j, (_, c) in enumerate(code):
                if f'save(fig, "figs_v2/{nm}")' in c.source:
                    return j
            return 1 << 30
        wanted = sorted(need, key=_order)

        picked, missing = [], []
        for name in wanted:
            hit = [c for _, c in code if f'save(fig, "figs_v2/{name}")' in c.source]
            if hit:
                picked.append((name, hit[0]))
            else:
                missing.append(name)
        if missing:
            have = sorted({ln.split('"')[1].split("/")[-1] for _, c in code
                           for ln in c.source.splitlines() if 'save(fig, "figs_v2/' in ln})
            print(f"NOT FOUND: {missing}\navailable: {have}")
            return 1

        sub = nbformat.v4.new_notebook()
        sub.metadata = nb.metadata
        # dedupe: several figure names can share one cell (fig20a/b/c) --
        # execute it once, not once per requested name
        cells, seen = [], set()
        for _, c in picked:
            if id(c) not in seen:
                seen.add(id(c))
                cells.append(c)
        sub.cells = [setup] + cells
        for name, c in picked:
            ast.parse("\n".join("" if s.lstrip().startswith(("%", "!")) else s
                                for s in c.source.splitlines()))
        print(f"executing setup + {len(cells)} figure cells for: {[n for n, _ in picked]}")

        km = KernelManager()
        km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(sub, km=km, timeout=7200,
                                resources={"metadata": {"path": str(HERE)}})
        t0 = time.time()
        try:
            client.execute()
        finally:
            nbformat.write(sub, HERE / "_subset_run.ipynb")
        print(f"\ndone in {time.time()-t0:.0f}s -> figs_v2/*.pdf + figs_preview/*.png")
        for name, _ in picked:
            p = HERE / "figs_v2" / f"{name}.pdf"
            if p.exists():
                print(f"   {name}.pdf  {p.stat().st_size/1e3:.0f} kB")
            elif name in GUARDED_PENDING:
                print(f"   {name}.pdf  ABSENT (expected -- guarded on a cache that "
                      "does not exist yet, see the cell's own printed guard message)")
            else:
                print(f"   {name}.pdf  MISSING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["fig00_hero"]))
