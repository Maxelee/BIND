"""WP-A6 task 5 prep: the DES-n(z)-weighted kappa x y posterior envelope.

Implements the Popeye-session-9 instruction for the Pandey comparison's
MODEL side: combine statsemu's 5 per-source-plane C_ell^{kappa y} rows
with the frozen DES Y3 source weights (`wp4_mocks/weights_desy3.npz`,
w_default; Born-level linearity of kappa_eff in the source planes), PER
POSTERIOR SAMPLE (percentiles of the weighted sum, not a weighted sum of
percentiles), at the A5 FINAL posterior (thinned-2000 convention).

What this does NOT yet do (documented next steps, not silently skipped):
- the ell -> theta Hankel transform into Pandey's measured frame
  (xi^{gamma_t y}(theta) needs the J2 kernel + curved/flat-sky choice +
  the y-map beam/pixel window conventions);
- any data comparison — the Pandey vector is only loaded and displayed
  for scale.

PRELIMINARY status carried in the output json: statsemu's cl_kappa_y
8-fold CV error is 15.4% (6.8x SEM) at n_pca=16 — the y-family re-fit
(n_pca bump) is queued and this product is regenerated after it.

Run:  python analysis/paper3a/scripts/run_a6_pandey_prep.py
Out:  wp6_propagation/a6_kappa_y_weighted.npz
      wp6_propagation/figures/a6_kappa_y_weighted.png
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
from analysis.paper3a.emulator.statsemu import WP6, StatsEmulator  # noqa: E402
from analysis.paper3a.scripts.run_a6_posterior_stats import (  # noqa: E402
    load_posterior_thin,
)

B_MOCKS = Path("/mnt/ceph/users/mlee1/paper3/B/wp4_mocks")


def main() -> None:
    emu = StatsEmulator.load()
    post = load_posterior_thin()
    u_fid = pm.astro_physical_to_unit(pm.ASTRO_FIDUCIAL)

    wz = np.load(B_MOCKS / "weights_desy3.npz", allow_pickle=True)
    w = np.asarray(wz["w_default"], float)                    # (5,)
    src_z = np.asarray(wz["source_z"], float)
    assert np.allclose(src_z, emu.source_redshifts), "source-plane mismatch"

    raw = np.load(WP6 / "sb35_stats" / "emulator_dataset.npz",
                  allow_pickle=False)
    ell = np.asarray(raw["a__cl_kappa_y__ell"], float)

    pred = emu.predict(post, "cl_kappa_y")                    # (N, 5, 724)
    weighted = np.tensordot(pred, w, axes=([1], [0]))         # (N, 724)
    fid = np.tensordot(emu.predict(u_fid, "cl_kappa_y"), w, axes=([0], [0]))
    qs = np.nanpercentile(weighted, [2.5, 16, 50, 84, 97.5], axis=0)

    out = WP6 / "a6_kappa_y_weighted.npz"
    np.savez_compressed(
        out, ell=ell, q=qs.astype(np.float32), fid=fid.astype(np.float32),
        weights=w, source_z=src_z,
        provenance=np.array(json.dumps({
            "n_posterior": len(post),
            "weights": "wp4_mocks/weights_desy3.npz w_default (NNLS, B4)",
            "status": "PRELIMINARY — cl_kappa_y CV 15.4% at n_pca=16; "
                      "regenerate after the y-family re-fit",
            "remaining_frame_gap": "ell->theta Hankel (J2) + y-map "
                                   "beam/pixel window before any Pandey "
                                   "data comparison",
        })))
    print(f"wrote {out}")

    dl = ell * (ell + 1) / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(ell, dl * qs[2], color="#0072B2", lw=1.6,
            label="posterior median (n(z)-weighted)")
    ax.fill_between(ell, dl * qs[1], dl * qs[3], color="#0072B2", alpha=0.3,
                    label="68%")
    ax.fill_between(ell, dl * qs[0], dl * qs[4], color="#0072B2", alpha=0.15)
    ax.plot(ell, dl * fid, color="#009E73", lw=1.2, ls="--",
            label="TNG fiducial")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa_{\rm eff} y}/2\pi$")
    ax.set_title("A5-posterior DES-weighted kappa x y (PRELIMINARY)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    figdir = WP6 / "figures"
    figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / "a6_kappa_y_weighted.png", dpi=150)
    print(f"wrote {figdir}/a6_kappa_y_weighted.png")

    # posterior-vs-fiducial amplitude summary at a few ells
    for l0 in (300, 1000, 3000):
        j = int(np.argmin(np.abs(ell - l0)))
        print(f"ell~{ell[j]:6.0f}: posterior/fid = "
              f"{qs[2, j] / fid[j]:.3f} [{qs[1, j] / fid[j]:.3f}, "
              f"{qs[3, j] / fid[j]:.3f}]")


if __name__ == "__main__":
    main()
