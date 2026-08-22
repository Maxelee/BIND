"""ONE driver for the whole imgs_1000/ directory.

Goal: ``imgs_1000/`` carries the exact figure set of ``imgs/`` (the shipped
paper directory), rebuilt on the N=1000 campaign (``ceph/bind_n1000``) wherever
a figure actually depends on realizations, and copied verbatim where it does
not.  Every figure reads only COMPUTE-ONCE saved statistics (the per-run MPI
stats products, the field/yt/nu05 caches, the assembled Sobol dataset) -- no
figure build streams raw map cubes.

    python make_imgs_1000.py                    # readiness table (default)
    python make_imgs_1000.py --build            # build everything READY
    python make_imgs_1000.py --build fig06_spectra_validation figA3_texture_debias
    python make_imgs_1000.py --assemble         # (re)assemble the Sobol dataset
                                                #  when all 256 sb35 runs have stats
    python make_imgs_1000.py --allow-partial    # with --build: render SOBOL-class
                                                #  figures from a partial dataset
                                                #  (stamped provisional in PROVENANCE)

Idempotent and re-runnable: as the campaign lands more runs, re-running
``--assemble`` + ``--build`` builds exactly what became ready.  Per-figure
build logs land in analysis/imgs1000_logs/; imgs_1000/PROVENANCE.json records,
for every figure, when it was built, by what, from which inputs (with mtimes),
and the campaign state (Sobol nodes / twobound stats) it saw.

Figure classes
  COPY      realization-independent -> copied verbatim from imgs/
  MAP       fiducial/truth/dmo arm caches (all complete today)
  TWOBOUND  needs per-run stats for specific twobound replicas
  SOBOL     needs the assembled 256-node emulator_dataset_n1000.npz
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path("/mnt/home/mlee1/BIND")
PAPER = REPO / "BINDing_the_lightcone"
HERE = PAPER / "analysis"
IMGS = PAPER / "imgs"
IMGS1000 = PAPER / "imgs_1000"
PREVIEW = PAPER / "figs_preview"
P1 = REPO / "papers/01_pipeline"
FSW = P1 / "referee/work"
ROOT_IMGS = REPO / "imgs"                     # fsfig_common.save_imgs target
N1K = Path("/mnt/home/mlee1/ceph/bind_n1000")
DS = N1K / "emulator_dataset_n1000.npz"
PY = "/mnt/home/mlee1/venvs/BIND_env/bin/python"
LOGDIR = HERE / "imgs1000_logs"
PROV = IMGS1000 / "PROVENANCE.json"
REPLICA = os.environ.get("BIND_FID_REPLICA", "tb49")
STATS3 = ("Cl_kappa.npz", "peak_counts.npz", "nongaussian_stats.npz")


# ── campaign state probes ────────────────────────────────────────────────────
def tb_stats_count() -> int:
    return sum(all((N1K / f"twobound/run_{r:04d}" / s).exists() for s in STATS3)
               for r in range(60))


def sb_stats_count() -> int:
    return sum(all((N1K / f"sb35/run_{r:04d}" / s).exists() for s in STATS3)
               for r in range(256))


def dataset_state() -> dict:
    """nodes in the assembled dataset + whether it is complete AND fresh."""
    if not DS.exists():
        return dict(exists=False, nodes=0, complete=False, fresh=False)
    with np.load(DS, allow_pickle=True) as d:
        rid = np.asarray(d["run_ids"], int)
    newest_stats = max(((N1K / f"sb35/run_{r:04d}" / "Cl_kappa.npz").stat().st_mtime
                        for r in range(256)
                        if (N1K / f"sb35/run_{r:04d}" / "Cl_kappa.npz").exists()),
                       default=0.0)
    complete = rid.tolist() == list(range(256))
    fresh = DS.stat().st_mtime >= newest_stats
    return dict(exists=True, nodes=len(rid), complete=complete, fresh=fresh)


def _ok(p: Path) -> bool:
    return p.exists()


def yt_cache_ok(side: str) -> bool:
    tag = f"_{REPLICA}" if side == "bind" else ""
    p = N1K / "field_cache" / f"yt_stats_{side}{tag}_n1000.npz"
    if not p.exists():
        return False
    try:
        with np.load(p) as z:
            return int(z["n_done"]) == int(z["yt_real"].shape[0])
    except Exception:
        return False


def core_ok() -> bool:
    need = [N1K / "field_cache/field_stats_fid_n1000.npz",
            N1K / f"nu05_shards/sci_bind_{REPLICA}_n1000.npz",
            N1K / "nu05_shards/sci_truth_n1000.npz",
            N1K / f"twobound/{ {'tb18':'run_0018','tb49':'run_0049','tb53':'run_0053'}[REPLICA] }/Cl_kappa.npz",
            N1K / "dmo/run_0000/Cl_kappa.npz",
            N1K / "truth/run_0000/Cl_kappa.npz"]
    return all(_ok(p) for p in need)


# ── build helpers ────────────────────────────────────────────────────────────
def run_script(script: Path, log_name: str, cwd: Path, extra_env: dict | None = None) -> bool:
    LOGDIR.mkdir(exist_ok=True)
    env = dict(os.environ, BIND_CAMPAIGN="n1000", BIND_FID_REPLICA=REPLICA,
               OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    env.update(extra_env or {})
    log = LOGDIR / f"{log_name}.log"
    with open(log, "w") as fh:
        r = subprocess.run([PY, "-u", str(script)] if script.suffix == ".py"
                           else [PY, "-u", *str(script).split()],
                           cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"    BUILD FAILED (see {log})")
        with open(log) as fh:
            print("    " + "    ".join(fh.readlines()[-12:]))
    return r.returncode == 0


def run_argv(argv: list[str], log_name: str, cwd: Path) -> bool:
    LOGDIR.mkdir(exist_ok=True)
    env = dict(os.environ, BIND_CAMPAIGN="n1000", BIND_FID_REPLICA=REPLICA,
               OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    log = LOGDIR / f"{log_name}.log"
    with open(log, "w") as fh:
        r = subprocess.run(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"    BUILD FAILED (see {log})")
        with open(log) as fh:
            print("    " + "    ".join(fh.readlines()[-12:]))
    return r.returncode == 0


def promote(pairs: list[tuple[Path, Path]]) -> list[str]:
    out = []
    for src, dst in pairs:
        if not src.exists():
            raise FileNotFoundError(f"promotion source missing: {src}")
        IMGS1000.mkdir(exist_ok=True)
        shutil.copy2(src, dst)
        out.append(str(dst.name))
    return out


def record(fig: str, note: str, sources: list[Path], produced: list[str],
           provisional: bool = False) -> None:
    prov = json.loads(PROV.read_text()) if PROV.exists() else {}
    st = dataset_state()
    prov[fig] = dict(
        built_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        note=note,
        provisional=provisional,
        campaign=dict(sobol_nodes=st["nodes"], sobol_complete=st["complete"],
                      twobound_stats=tb_stats_count(), replica=REPLICA),
        produced=produced,
        sources={str(p): datetime.datetime.fromtimestamp(p.stat().st_mtime)
                 .isoformat(timespec="seconds")
                 for p in sources if p.exists()},
    )
    PROV.write_text(json.dumps(prov, indent=1, sort_keys=True))


# ── figure registry ──────────────────────────────────────────────────────────
def _fsfig(script, stem, log):
    """Build one referee/work fsfig builder and promote its _fidswap_n1000 output."""
    def build():
        if not run_script(FSW / script, log, cwd=FSW):
            return None
        return promote([(ROOT_IMGS / f"{stem}_fidswap_n1000.png", IMGS1000 / f"{stem}.png"),
                        (ROOT_IMGS / f"{stem}_fidswap_n1000.pdf", IMGS1000 / f"{stem}.pdf")])
    return build


def _copy_verbatim(stem):
    def build():
        pairs = []
        for ext in ("png", "pdf"):
            src = IMGS / f"{stem}.{ext}"
            if src.exists():
                pairs.append((src, IMGS1000 / f"{stem}.{ext}"))
        if not pairs:
            raise FileNotFoundError(f"nothing to copy for {stem} in {IMGS}")
        return promote(pairs)
    return build


def _nb_cells(cells_to_paper: dict[str, str], log):
    """Run notebook cells via _run_subset.py (n1000-tagged saves) and promote."""
    def build():
        argv = [PY, "-u", str(P1 / "_run_subset.py"), *cells_to_paper.keys()]
        if not run_argv(argv, log, cwd=P1):
            return None
        pairs = []
        for cell, paper_stem in cells_to_paper.items():
            pairs.append((P1 / f"figs_v2/{cell}_n1000.pdf", IMGS1000 / f"{paper_stem}.pdf"))
            pairs.append((P1 / f"figs_preview/{cell}_n1000.png", IMGS1000 / f"{paper_stem}.png"))
        return promote(pairs)
    return build


def _p1_script(script, stems_v2_to_paper: dict[str, str], log):
    """Run a papers/01_pipeline script; promote figs_v2/<stem>_n1000 outputs."""
    def build():
        if not run_script(P1 / script, log, cwd=P1):
            return None
        pairs = []
        for v2_stem, paper_stem in stems_v2_to_paper.items():
            pairs.append((P1 / f"figs_v2/{v2_stem}_n1000.pdf", IMGS1000 / f"{paper_stem}.pdf"))
            pairs.append((P1 / f"figs_preview/{v2_stem}_n1000.png", IMGS1000 / f"{paper_stem}.png"))
        return promote(pairs)
    return build


def _analysis_script(script, direct_stems: list[str], preview_stems: list[str], log):
    """Run an analysis/ script that writes PDFs into imgs_1000/ itself; promote
    the PNG previews paper_style.save() left in figs_preview/.

    Previews written via paper_style.save(IMGS_1000/<stem>) land UNTAGGED in
    BINDing_the_lightcone/figs_preview/ -- the same filename a sci50 run uses.
    A stale sci50 preview must never be promoted as the n1000 figure, so any
    untagged source is required to be newer than this build's start time."""
    import time as _time

    def build():
        t0 = _time.time()
        if not run_script(HERE / script, log, cwd=HERE):
            return None
        produced = []
        for stem in direct_stems:
            for ext in ("pdf", "png"):
                if (IMGS1000 / f"{stem}.{ext}").exists():
                    produced.append(f"{stem}.{ext}")
        pairs = []
        for stem in preview_stems:
            src_tag = PREVIEW / f"{stem}_n1000.png"
            src_plain = PREVIEW / f"{stem}.png"
            src = src_tag if src_tag.exists() else src_plain
            if src == src_plain and src.exists() and src.stat().st_mtime < t0:
                raise RuntimeError(
                    f"{src} predates this build -- it is a stale (probably sci50) "
                    "preview; the builder did not write the expected png")
            pairs.append((src, IMGS1000 / f"{stem}.png"))
            pdf_tag = PREVIEW / f"{stem}_n1000.pdf"
            if pdf_tag.exists() and pdf_tag.stat().st_mtime >= t0:
                pairs.append((pdf_tag, IMGS1000 / f"{stem}.pdf"))
        return produced + promote(pairs)
    return build


def sobol_ready(allow_partial: bool):
    st = dataset_state()
    if st["complete"] and st["fresh"]:
        return True, "dataset complete (256/256) and fresh"
    if allow_partial and st["exists"]:
        return True, f"PARTIAL dataset ({st['nodes']}/256) -- provisional render"
    why = ("dataset missing" if not st["exists"] else
           f"dataset has {st['nodes']}/256 nodes" if not st["complete"] else
           "dataset stale vs newer per-run stats (re-run --assemble)")
    return False, why


def registry(allow_partial: bool):
    st_sobol = sobol_ready(allow_partial)
    strict_sobol = sobol_ready(False)[0]        # ignores --allow-partial
    tb_n = tb_stats_count()
    figs = []

    for stem, why in (
            ("fig01_pipeline_diagram", "single-draw showcase, no realization axis"),
            ("fig03_halo_validation", "per-halo snapshot atlas, no realization axis"),
            ("fig04_radial_profiles", "per-halo snapshot profiles, no realization axis"),
            ("figA1_fullhydro_wl", "one-off full-hydro lux trace (50-real R1 ladder)"),
            ("figA2_fullhydro_gas", "one-off full-hydro lux trace (50-real R1 ladder)")):
        figs.append(dict(name=stem, cls="COPY", ready=(True, why), stems=[stem],
                         strict_ok=True,
                         note=f"realization-independent; copied verbatim from imgs/ ({why})",
                         build=_copy_verbatim(stem), sources=[IMGS / f"{stem}.png"]))

    figs.append(dict(name="fig02_hero", cls="MAP", stems=["fig02_hero"],
                     ready=(core_ok(), "core arm caches"), strict_ok=core_ok(),
                     note="fsfig2_hero.py, realization 0 of each arm",
                     build=_fsfig("fsfig2_hero.py", "fig02_hero", "fig02"),
                     sources=[N1K / "twobound/run_0049/kappa_maps.npz"]))
    figs.append(dict(name="fig05_field_validation", cls="MAP",
                     stems=["fig05_field_validation"],
                     ready=(core_ok(), "field cache + nu05 shards"), strict_ok=core_ok(),
                     note="fsfig5_field.py on the n1000 field cache + nu05 shards",
                     build=_fsfig("fsfig5_field.py", "fig05_field_validation", "fig05"),
                     sources=[N1K / "field_cache/field_stats_fid_n1000.npz",
                              N1K / f"nu05_shards/sci_bind_{REPLICA}_n1000.npz",
                              N1K / "nu05_shards/sci_truth_n1000.npz"]))
    _yt_ok = yt_cache_ok("bind") and yt_cache_ok("truth")
    figs.append(dict(name="fig06_spectra_validation", cls="MAP",
                     stems=["fig06_spectra_validation"],
                     ready=(core_ok() and _yt_ok,
                            "field cache + yt caches" if _yt_ok else
                            "yt caches missing/incomplete (build_yt_cache_n1000.py)"),
                     strict_ok=core_ok() and _yt_ok,
                     note="fsfig6_spectra.py, fully cache-driven at N=1000 "
                          "(field cache + yt cache, cross-validated vs per-run MPI stats)",
                     build=_fsfig("fsfig6_spectra.py", "fig06_spectra_validation", "fig06"),
                     sources=[N1K / "field_cache/field_stats_fid_n1000.npz",
                              N1K / f"field_cache/yt_stats_bind_{REPLICA}_n1000.npz",
                              N1K / "field_cache/yt_stats_truth_n1000.npz"]))
    _tex = HERE / "texture_template_maps_n1000.npz"
    _figa3_ok = core_ok() and (N1K / "analysis/fig05_numbers_n1000.npz").exists()
    figs.append(dict(name="figA3_texture_debias", cls="MAP",
                     stems=["figA3_texture_debias"],
                     ready=(_figa3_ok,
                            "replica trio + fig05 numbers"
                            + ("" if _tex.exists() else
                               " (first run also BUILDS the n1000 texture template: "
                               "~12 min, streamed)")),
                     strict_ok=_figa3_ok,
                     note="make_fig_texture_debias.py; n1000-tagged texture template",
                     build=_analysis_script("make_fig_texture_debias.py",
                                            ["figA3_texture_debias"],
                                            ["figA3_texture_debias"], "figA3"),
                     sources=[_tex, N1K / "analysis/fig05_numbers_n1000.npz"]))
    _noisy = HERE / "noisy_tb49_stats_n1000.npz"
    figs.append(dict(name="fig05n_field_validation_noisy", cls="MAP",
                     stems=["fig05n_field_validation_noisy"],
                     ready=(_noisy.exists(),
                            "noisy nu-stats cache present" if _noisy.exists() else
                            "noisy cache missing; noisy_grid loads the FULL ~21 GB "
                            "cube (no streaming) -- run BIND_CAMPAIGN=n1000 python "
                            "make_fig_noisy_tb49.py on a node with >30 GB RAM "
                            "(this jupyter session's cgroup is 17.5 GB)"),
                     strict_ok=_noisy.exists(),
                     note="make_fig_noisy_tb49.py (n1000 bind arm; truth ref = "
                          "550-real sci50 noisy shard, stated on the figure)",
                     build=_analysis_script("make_fig_noisy_tb49.py",
                                            ["fig05n_field_validation_noisy"],
                                            ["fig05n_field_validation_noisy"], "fig05n"),
                     sources=[_noisy]))

    figs.append(dict(name="fig07_detectability", cls="SOBOL",
                     stems=["fig07_detectability"],
                     ready=st_sobol, strict_ok=strict_sobol,
                     note="make_fig_detectability.py (writes directly to imgs_1000)",
                     build=_analysis_script("make_fig_detectability.py",
                                            ["fig07_detectability"], [], "fig07"),
                     sources=[DS, N1K / "twobound/run_0049/Cl_kappa.npz"]))
    figs.append(dict(name="fig08_sl_response+fig09_param_response", cls="SOBOL",
                     stems=["fig08_sl_response", "fig09_param_response"],
                     ready=st_sobol, strict_ok=strict_sobol,
                     note="paper1 notebook cells fig05a_sl_response/fig05_param_response "
                          "via _run_subset.py (n1000-tagged saves), renamed on promotion",
                     build=_nb_cells({"fig05a_sl_response": "fig08_sl_response",
                                      "fig05_param_response": "fig09_param_response"},
                                     "fig08_fig09"),
                     sources=[DS]))
    figs.append(dict(name="fig10_covariation", cls="SOBOL",
                     stems=["fig10_covariation"],
                     ready=(st_sobol[0] and yt_cache_ok("bind"),
                            st_sobol[1] + ("" if yt_cache_ok("bind")
                                           else "; bind yt cache missing/incomplete")),
                     strict_ok=strict_sobol and yt_cache_ok("bind"),
                     note="fsfig10_covariation.py (cl_yt leg from the yt cache, full-N)",
                     build=_fsfig("fsfig10_covariation.py", "fig10_covariation", "fig10"),
                     sources=[DS, N1K / f"field_cache/yt_stats_bind_{REPLICA}_n1000.npz"]))
    figs.append(dict(name="pfig_fm_model_figures", cls="SOBOL",
                     stems=["pfig_fm_basis", "pfig_fm_freeamp",
                            "pfig_fm_model_curves", "pfig_fm_search_path"],
                     ready=st_sobol, strict_ok=strict_sobol,
                     note="family_model_section_figs.py: fixed shipped model "
                          "(basis/mean/latents), campaign-measured curves + re-measured "
                          "amplitudes, run_id-aligned LOO",
                     build=_p1_script("family_model_section_figs.py",
                                      {"pfig_fm_basis": "pfig_fm_basis",
                                       "pfig_fm_freeamp": "pfig_fm_freeamp",
                                       "pfig_fm_model_curves": "pfig_fm_model_curves",
                                       "pfig_fm_search_path": "pfig_fm_search_path"},
                                      "pfig_fm"),
                     sources=[DS, P1 / "figs_preview/amplitude_sets.npz",
                              P1 / "figs_preview/agnostic_lambda_results_obs.npz"]))
    figs.append(dict(name="pfig_s3b_clusters", cls="TWOBOUND+SOBOL",
                     stems=["pfig_s3b_cl_clusters", "pfig_s3b_pdf_clusters"],
                     ready=(tb_n == 60 and st_sobol[0],
                            f"twobound stats {tb_n}/60; {st_sobol[1]}"
                            + ("; pdf leg additionally needs MPI paired_stats "
                               "(V0 errors)" if tb_n == 60 else "")),
                     strict_ok=tb_n == 60 and strict_sobol,
                     note="paper_s3b_clusters.py on n1000 twobound stats "
                          "(paired_stats preferred, _fast fallback; pdf V0-S/N gate "
                          "fails loudly on NaN MF errors)",
                     build=_p1_script("paper_s3b_clusters.py",
                                      {"pfig_s3b_cl_clusters": "pfig_s3b_cl_clusters",
                                       "pfig_s3b_pdf_clusters": "pfig_s3b_pdf_clusters"},
                                      "pfig_s3b"),
                     sources=[DS]))
    figs.append(dict(name="gas_family_figures", cls="TWOBOUND+SOBOL",
                     stems=["figA4_gas_families", "pfig_fm_model_curves_gas"],
                     ready=(tb_n == 60 and st_sobol[0],
                            f"twobound stats {tb_n}/60; {st_sobol[1]}"),
                     strict_ok=tb_n == 60 and strict_sobol,
                     note="posterity_model_curves.py (runs gas_families.py inline; "
                          "figA4 written directly, gas curves promoted from figs_preview)",
                     build=_analysis_script("posterity_model_curves.py",
                                            ["figA4_gas_families"],
                                            ["figA4_gas_families",
                                             "pfig_fm_model_curves_gas"], "figA4_gas"),
                     sources=[DS]))
    return figs


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", nargs="*", metavar="FIG",
                    help="build these figures (no names = everything READY)")
    ap.add_argument("--assemble", action="store_true",
                    help="(re)assemble emulator_dataset_n1000.npz from the sb35 stats")
    ap.add_argument("--allow-partial", action="store_true",
                    help="let SOBOL-class figures render from a partial dataset "
                         "(stamped provisional)")
    args = ap.parse_args()

    if args.assemble:
        sb = sb_stats_count()
        print(f"sb35 runs with stats: {sb}/256")
        if sb < 256 and not args.allow_partial:
            raise SystemExit("refusing to assemble a partial dataset without "
                             "--allow-partial (the current dataset stays in place)")
        ok = run_argv([PY, "-m", "bind.cli.emulator_assemble",
                       "--runs_dir", str(N1K / "sb35"), "--out", str(DS),
                       "--dmo_dir", str(N1K / "dmo/run_0000")],
                      "assemble", cwd=REPO)
        if not ok:
            raise SystemExit(1)
        print(f"assembled {DS} -> {dataset_state()}")

    figs = registry(args.allow_partial)
    st = dataset_state()
    print(f"campaign state: sobol dataset {st['nodes']}/256 nodes "
          f"(complete={st['complete']}, fresh={st['fresh']}); "
          f"twobound stats {tb_stats_count()}/60; sb35 stats {sb_stats_count()}/256\n")

    if args.build is None:
        for f in figs:
            ok, why = f["ready"]
            print(f"  {'READY  ' if ok else 'WAITING'} [{f['cls']:14s}] "
                  f"{f['name']:32s} -> {', '.join(f['stems'])}\n"
                  f"          {why}")
        print("\nrun with --build to build everything READY "
              "(name a registry entry or any output stem to build one)")
        return

    # exact matching only: an entry is selected by its registry name or by any
    # of its real output stems.  Substring matching silently mis-selected
    # entries (2026-08-21 review finding) and is gone.
    want = set(args.build) if args.build else None
    consumed = set()
    for f in figs:
        if want is not None:
            hit = ({f["name"]} | set(f["stems"])) & want
            if not hit:
                continue
            consumed |= hit
        ok, why = f["ready"]
        if not ok and want is None:
            print(f"skip {f['name']} (WAITING: {why})")
            continue
        if not ok:
            print(f"{f['name']}: requested but not ready ({why}) -- attempting anyway")
        print(f"building {f['name']} ...")
        produced = f["build"]()
        if produced is None:
            print(f"  {f['name']} FAILED")
            continue
        record(f["name"], f["note"], f["sources"], produced,
               provisional=not f["strict_ok"])
        print(f"  OK -> {', '.join(produced)}")
    if want is not None and want - consumed:
        raise SystemExit(f"unknown figure name(s): {sorted(want - consumed)} -- "
                         "use a registry entry name or an output stem from --help/status")


if __name__ == "__main__":
    main()
