"""Aggregate the CAMELS L50n512/1P CylToSph measurements -> per-theta model.

Consumes the per-sim npz written by ``camels1p_cyltosph.py`` (175 sims; the
Omega0 lower-extreme box is physically empty) and produces:

- ``cyltosph_1p_summary.json``: per parameter x gate mass bin, the median
  CylToSph at each 1P step (n2, n1, 0, 1, 2 — physical values from the
  suite manifest), the fiducial-normalized log-slope over the unit cube,
  and the span across the extremes.
- ``figures/wp4_cyltosph_directions.png``: parameters ranked by
  |Delta ln CylToSph| between bounds, group bin (13.0-13.4) — directly
  comparable to the task-5 f_gas parameter directions.
- The per-theta linear model consumed by
  ``emulator.cyltosph_theta.CylToSphTheta``: slopes s_i(bin) with
  CylToSph(theta, bin) = wp2_factor(bin-independent per snap) *
  exp( sum_i s_i(bin) * (u_i - u_fid_i) ), i.e. the L50 *trend* transfers
  multiplicatively onto the wp2 absolute calibration (box-depth and
  selection differences cancel at first order).

Run (torch3 venv):
    python analysis/paper3a/scripts/camels1p_cyltosph_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "src"))

from analysis.paper3a.emulator import params_meta as pm  # noqa: E402

OUT = Path("/mnt/ceph/users/mlee1/paper3/A/wp4_emulator")
MEAS = OUT / "cyltosph_1p"
MANIFEST = Path("/mnt/ceph/users/camels/Sims/IllustrisTNG/L50n512/1P/"
                "CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt")
N_BINS_USED = 3          # 13.0-13.4 / 13.4-13.8 / 13.8-14.2 populate at z=0
STEPS = ("n2", "n1", "0", "1", "2")


def load_manifest() -> tuple[list[str], dict[str, np.ndarray]]:
    lines = MANIFEST.read_text().strip().splitlines()
    names = lines[0].lstrip("#").split()[1:-1]           # param columns, drop seed
    vals = {}
    for ln in lines[1:]:
        parts = ln.split()
        vals[parts[0]] = np.array([float(x) for x in parts[1:len(names) + 1]])
    return names, vals


def main() -> None:
    param_names, manifest = load_manifest()
    assert param_names == pm.PARAM_NAMES, "manifest ordering != SB35 CSV"

    fid = np.load(MEAS / "1P_0_snap090.npz")
    fid_med = fid["cyltosph_median_by_bin"][:N_BINS_USED]

    summary: dict = {"fiducial_median_by_bin": fid_med.tolist(),
                     "bins": ["13.0-13.4", "13.4-13.8", "13.8-14.2"],
                     "params": {}}
    slopes = np.zeros((pm.N_PARAMS, N_BINS_USED))        # d ln C2S / d u
    for i, name in enumerate(param_names, start=1):
        entry: dict = {"steps": {}}
        u_med = {}
        for step in STEPS:
            sim = f"1P_p{i}_{step}" if step != "0" else "1P_0"
            p = MEAS / f"{sim}_snap090.npz"
            if not p.exists():
                entry["steps"][step] = None
                continue
            d = np.load(p)
            med = d["cyltosph_median_by_bin"][:N_BINS_USED]
            phys = manifest.get(f"1P_p{i}_{step}", manifest["1P_p1_0"])[i - 1]
            entry["steps"][step] = {"physical": float(phys), "median_by_bin":
                                    [None if not np.isfinite(x) else float(x) for x in med]}
            u_med[step] = (phys, med)
        # fiducial-normalized log slope per bin from a least-squares line in
        # the parameter's own unit-cube coordinate
        lo, hi = pm.PARAM_MIN[i - 1], pm.PARAM_MAX[i - 1]
        log_flag = bool(pm.PARAM_LOG[i - 1])
        for b in range(N_BINS_USED):
            xs, ys = [], []
            for step, (phys, med) in u_med.items():
                if not np.isfinite(med[b]) or med[b] <= 0:
                    continue
                if log_flag:
                    x = (np.log10(phys) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
                else:
                    x = (phys - lo) / (hi - lo)
                xs.append(x)
                ys.append(np.log(med[b]))
            if len(xs) >= 3:
                slopes[i - 1, b] = np.polyfit(xs, ys, 1)[0]
        entry["dlnC2S_du_by_bin"] = slopes[i - 1].tolist()
        summary["params"][name] = entry

    astro_slopes = slopes[pm.ASTRO_IDX]
    np.savez_compressed(OUT / "cyltosph_theta_model.npz",
                        param_names=np.array(pm.ASTRO_NAMES),
                        slopes_dlnC2S_du=astro_slopes,
                        bins=np.array(summary["bins"]),
                        fiducial_l50_median=fid_med,
                        note=np.array("multiplicative trend model on the wp2 factor; "
                                      "unit-cube astro coords; z=0 (snap 090)"))
    (OUT / "cyltosph_1p_summary.json").write_text(json.dumps(summary, indent=2))

    # ranked-direction figure, group bin
    order = np.argsort(-np.abs(astro_slopes[:, 0]))
    top = order[:12]
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(top))
    ax.barh(y, astro_slopes[top, 0], color=np.where(astro_slopes[top, 0] > 0,
                                                    "#0072B2", "#D55E00"))
    ax.set_yticks(y, [pm.ASTRO_NAMES[j] for j in top], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$d\,\ln\,{\rm CylToSph}\ /\ d\,u$   (group bin $10^{13.0-13.4}$, z=0)")
    ax.set_title("CylToSph feedback dependence — CAMELS L50n512/1P")
    ax.axvline(0, color="k", lw=0.8)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "wp4_cyltosph_directions.png", dpi=150)

    print("fiducial (L50) medians:", np.round(fid_med, 3))
    print("top |slopes| (group bin):")
    for j in top[:8]:
        print(f"  {pm.ASTRO_NAMES[j]:38s} {astro_slopes[j,0]:+.4f}")
    span = np.abs(astro_slopes[:, 0]).max()
    print(f"max |dlnC2S/du| group bin: {span:.4f} "
          f"(full-cube span ~ {100*(np.exp(span)-1):.1f}%)")


if __name__ == "__main__":
    main()
