"""fig04_ejection_profiles.png -- radial profiles of the aligned quadrupole
fraction q2 = |c2|/c0 for the WL convergence residual (left) and the tSZ
Compton-y pressure (right), by mass bin (group / cluster), BIND fid
(solid) vs TNG truth (dashed). Shaded band = the ejection annulus
(0.5-2.0 r200c) used for the scalar orientation/alignment tests elsewhere
in the paper.

Data (cached, unmodified -- already the reduced per-mass-bin form; no need
to touch raw composite slabs):
    /mnt/home/mlee1/ceph/bind_science/ejection/fid_snap096.npz
    /mnt/home/mlee1/ceph/bind_science/ejection/truth_snap096.npz
  -> keys 'q2k' (2 mass bins, 15 radial bins), 'q2y' (2,15), 'rmid' (15,),
  'counts' (2,). Mass bins: 0=group [1e13,5e13), 1=cluster [5e13,5e14)
  Msun/h. rmid is in units of r200c over [0,3].

Verified: loading fid_snap096.npz directly reproduces the notebook's
printed cell-3 stdout exactly (counts=[2512,416], q2k_ann=[0.0431,0.0349],
q2y_ann=[0.1312,0.1634], align=[-0.773,-0.895]).

Original source: examples/ejection_anisotropy.ipynb, cell 5
("Figure 1 -- the radial anisotropy of kappa and y (fiducial vs TNG
truth)"), depending on reduce_run() in cell 3 (itself cache-checking; the
cache npz already exists on disk).
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
import matplotlib.pyplot as plt  # noqa: E402

from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

OUT = pathlib.Path("/mnt/home/mlee1/ceph/bind_science/ejection")
BIN_LABELS = ["group", "cluster"]
BIN_COLORS = [COLORS["bind"], COLORS["highlight"]]
ANN = (0.5, 2.0)  # ejection annulus, r/r200c


def main():
    setup()
    fid = dict(np.load(OUT / "fid_snap096.npz"))
    truth = dict(np.load(OUT / "truth_snap096.npz"))
    rmid = fid["rmid"]

    print("fid counts per mass bin:", fid["counts"])
    print("fid q2_kappa (ejection annulus):", np.round(fid["q2k_ann"], 4))
    print("fid q2_y     (ejection annulus):", np.round(fid["q2y_ann"], 4))

    fig, ax = plt.subplots(1, 2, figsize=TWO_COL)

    for b in range(len(BIN_LABELS)):
        if fid["counts"][b] < 10:
            continue
        c = BIN_COLORS[b]
        ax[0].plot(rmid, 100 * fid["q2k"][b], "-", color=c, lw=1.4, label=BIN_LABELS[b])
        ax[1].plot(rmid, 100 * fid["q2y"][b], "-", color=c, lw=1.4, label=BIN_LABELS[b])
        if np.isfinite(truth["q2k"][b]).any():
            ax[0].plot(rmid, 100 * truth["q2k"][b], "--", color=c, lw=1.1, alpha=0.7)
            ax[1].plot(rmid, 100 * truth["q2y"][b], "--", color=c, lw=1.1, alpha=0.7)

    labels = [
        r"$q_2(\kappa\!-\!\kappa_{\rm dmo})=|c_2|/c_0^{\rm dmo}$  [%]",
        r"$q_2(y)=|c_2|/c_0$  [%]",
    ]
    for i, (a, lab) in enumerate(zip(ax, labels)):
        a.axvspan(*ANN, color="0.9", zorder=0)
        a.set_xlabel(r"$r/r_{200c}$")
        a.set_ylabel(lab)
        a.set_xlim(rmid.min(), rmid.max())
        panel_label(a, f"({'ab'[i]})")

    ax[0].legend(fontsize=6.5, loc="upper right")

    fig.tight_layout()
    save(fig, "figs/fig04_ejection_profiles")


if __name__ == "__main__":
    main()
