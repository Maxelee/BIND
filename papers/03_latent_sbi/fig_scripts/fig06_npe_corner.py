#!/usr/bin/env python
"""fig06_npe_corner.png -- NPE posterior P(theta | fiducial C_ell), the six
best-constrained (of 30) astro parameters.

Loads the already-materialized NPE posterior-sample cache (24000 unit-cube
samples x 30 params; no `sbi` training re-run). Maps unit-cube samples to
physical plot units (log10 for the log-flagged params, using the installed
`bind` package's parameter metadata -- not a data file), ranks parameters by
posterior-std/prior-std ("shrink"; smaller = more informed), and corner-plots
the six most-informed against the TNG fiducial (= prior centre, u=0.5).

Data (already cached, read-only):
  /mnt/home/mlee1/BIND/examples/wl_latent_sbi_figs/npe_cl.npz -> samples (24000,30)
  /mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz -> param_names (30,) [names only]
  Installed package constants (not files): bind.inference.design.ASTRO_PARAM_INDICES,
    bind.inference.design._unit_to_native, bind.params.PARAM_LOG_FLAG/PARAM_MIN/PARAM_MAX

Source: sobol-sb35/examples/wl_latent_sbi.ipynb cells 16-17 (SS3b). Ported
verbatim (unit->physical mapping + shrink ranking + corner call); no NPE
training, no emulator forward pass.

Placeholder this replaces: figs/fig06_npe_corner.png (byte-identical to
examples/wl_latent_sbi_figs/f3b_corner_cl.png).
"""
import sys
from pathlib import Path

import corner
import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import COLORS, save, setup  # noqa: E402
from param_labels import short_label  # noqa: E402

DATASET = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")
NPE_CACHE = Path("/mnt/home/mlee1/BIND/examples/wl_latent_sbi_figs/npe_cl.npz")


def main():
    setup()
    d = np.load(DATASET, allow_pickle=True)
    pn = [str(s) for s in d["param_names"]]

    samples = np.load(NPE_CACHE, allow_pickle=True)["samples"]  # (24000, 30) unit cube

    from bind.inference.design import ASTRO_PARAM_INDICES, _unit_to_native
    from bind.params import PARAM_LOG_FLAG, PARAM_MIN, PARAM_MAX
    aidx = np.asarray(ASTRO_PARAM_INDICES)
    logm = PARAM_LOG_FLAG[aidx] == 1
    plo, phi = PARAM_MIN[aidx], PARAM_MAX[aidx]

    def to_plot(u):
        v = np.atleast_2d(_unit_to_native(np.asarray(u, float), aidx)).copy()
        v[:, logm] = np.log10(v[:, logm])
        return v

    truth = np.full(30, 0.5)  # fiducial = prior centre
    pr = 1 / np.sqrt(12.0)  # flat-prior std on the unit cube
    shrink = samples.std(0) / pr
    print("posterior std / prior std  (informed = <0.9):")
    for i in np.argsort(shrink)[:8]:
        print(f"    {pn[i]:32s} {shrink[i]:.2f}")
    print(f"    mean over 30 = {shrink.mean():.2f}")

    sel = np.argsort(shrink)[:6]
    labs = [("log " if logm[i] else "") + short_label(pn[i]) for i in sel]
    rng = [((np.log10(plo[i]) if logm[i] else plo[i]), (np.log10(phi[i]) if logm[i] else phi[i]))
           for i in sel]

    fig = corner.corner(
        to_plot(samples)[:, sel], labels=labs, truths=to_plot(truth)[0][sel], range=rng,
        color=COLORS["bind"], truth_color=COLORS["highlight"], levels=(0.68, 0.95),
        plot_datapoints=False, hist_kwargs={"density": True}, label_kwargs={"fontsize": 8},
    )
    for ax in fig.axes:
        ax.tick_params(labelsize=6)

    save(fig, "figs/fig06_npe_corner")


if __name__ == "__main__":
    main()
