"""fig10_kxy_driver.pdf -- which field-level probe carries the feedback signal.

Panels:
  (a) band-integrated feedback response D = median(std/|median|) over
      300<ell<8000, ranked by probe -- kappa*y, y*y, tau*tau, y*tau, kappa*kappa.
  (b) Sobol node-spread / fiducial as a function of multipole for the same
      five probes: the WL auto-spectrum (kappa*kappa) stays flat/dark to
      ell~1e4, while cross- and tSZ/kSZ probes "light up" at far lower ell.

Data: /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz (same file as the
  sibling kSZ paper's preamble) -> t__cl_kappa__value (253,5,5,724),
  t__cl_kappa_y__value (253,5,724), t__cl_yy__value / t__cl_tt__value /
  t__cl_yt__value (253,724), a__cl_kappa__ell (724,), source_redshifts (5,).

Source: ksz-desi-act/examples/paper_ksz_field.ipynb cell 14 (Sec. 6),
  reused verbatim here per dossier.md Gaps #2 (cross-paper provenance --
  produced for the sibling field-level kSZ paper). Ports the `stat()`
  accessor from that notebook's cell 2 preamble. `zi = len(ZS)-1` selects
  the last (highest) source-redshift tomographic bin for the kappa*kappa
  auto and kappa*y cross, matching the source cell exactly; yy/tt/yt are
  not tomographic in this dataset. D(arr) is closed-form (band nanmedian of
  nanstd/|nanmedian|) -- no fitting, no re-run.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import COLORS, TWO_COL, panel_label, save, setup

DS = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")
E = np.load(DS, allow_pickle=True)
ZS = E["source_redshifts"]


def stat(name):
    v = E[f"t__{name}__value"]
    ok = E[f"t__{name}__valid"]
    err = E[f"t__{name}__err"] if f"t__{name}__err" in E.files else None
    return v[ok], (err[ok] if err is not None else None), ok


ell = E["a__cl_kappa__ell"]
zi = len(ZS) - 1
band = (ell > 300) & (ell < 8000)

probes = {}
ckk, _, _ = stat("cl_kappa"); probes[r"$\kappa\kappa$"] = ckk[:, zi, zi]
cky, _, _ = stat("cl_kappa_y"); probes[r"$\kappa y$"] = cky[:, zi]
cyy, _, _ = stat("cl_yy"); probes[r"$yy$"] = cyy
ctt, _, _ = stat("cl_tt"); probes[r"$\tau\tau$"] = ctt
cyt, _, _ = stat("cl_yt"); probes[r"$y\tau$"] = cyt


def D(arr):
    a = arr[:, band]
    return np.nanmedian(np.nanstd(a, 0) / (np.abs(np.nanmedian(a, 0)) + 1e-30))


names = list(probes)
vals = [D(probes[n]) for n in names]
order = np.argsort(vals)

setup()
import matplotlib.pyplot as plt  # noqa: E402

# probe -> line color: kappa*kappa=bind(blue, the WL-auto baseline), kappa*y=secondary
# (green, matches panel (a) bar color), yy=highlight (red, the tSZ auto), tt/yt use two
# additional qualitative hues (no suite-semantic role fits a kSZ auto/cross pair) matching
# the original notebook's tab:purple/tab:brown assignment for these two probes.
line_colors = {
    r"$\kappa\kappa$": COLORS["bind"],
    r"$\kappa y$": COLORS["secondary"],
    r"$yy$": COLORS["highlight"],
    r"$\tau\tau$": "tab:purple",
    r"$y\tau$": "tab:brown",
}

fig, ax = plt.subplots(1, 2, figsize=(TWO_COL[0], TWO_COL[1]), constrained_layout=True)

# (a) band-integrated feedback response D, ranked by probe
ax[0].barh(np.arange(len(names)), [vals[i] for i in order], color=COLORS["secondary"])
ax[0].set_yticks(np.arange(len(names)))
ax[0].set_yticklabels([names[i] for i in order])
ax[0].set_xlabel(r"feedback response $D$ (band-integrated)")
panel_label(ax[0], "(a)")

# (b) node spread / fiducial vs ell, per probe
for n in names:
    a = probes[n]
    lo, hi = np.nanpercentile(a, [16, 84], 0)
    ax[1].semilogx(ell, (hi - lo) / (2 * np.abs(np.nanmedian(a, 0))),
                    color=line_colors[n], lw=1.2, label=n)
ax[1].set_xlabel(r"$\ell$")
ax[1].set_ylabel("node spread / fiducial")
ax[1].set_ylim(0, None)
ax[1].legend(ncol=2)
panel_label(ax[1], "(b)")

save(fig, "figs/fig10_kxy_driver")

print("feedback response D:", {names[i]: round(vals[i], 3) for i in order[::-1]})
