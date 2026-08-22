"""Demo: instant ray-traced lightcone statistics from `bind.emulator`.

Loads (or quickly trains) the emulator, then shows the headline use case — feed a
30-d feedback vector + a source redshift, get every field-level statistic in
milliseconds — and the feedback response: how the WL suppression `S(ℓ)`, the tSZ
`C_ℓ^yy`, and the Y–M relation move as a single feedback knob (wind energy) is
dialled across its prior.

    python examples/lightcone_emulator_demo.py            # uses cached bundle/dataset
    python examples/lightcone_emulator_demo.py --train    # (re)fit a quick MLP first
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np

import bind
from bind.emulator import EmulatorDataset
from bind.emulator.core import Emulator

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "figures_lightcone"
DS = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")
BUNDLE = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator/lightcone_emulator_mlp.pt")


def _get_emulator(train: bool) -> Emulator:
    if BUNDLE.exists() and not train:
        print(f"[demo] loading {BUNDLE}")
        return Emulator.load(BUNDLE)
    print("[demo] training a quick MLP emulator")
    ds = EmulatorDataset.load(DS)
    em = Emulator(backend="mlp", n_components=20,
                  backend_kwargs=dict(n_models=5, epochs=300)).fit(ds, verbose=False)
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    em.save(BUNDLE)
    return em


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--z_s", type=float, default=1.0)
    args = ap.parse_args()
    em = _get_emulator(args.train)

    # 1) one prediction, timed
    fid = bind.fiducial_params()
    t = time.perf_counter()
    out = em.predict(fid, z_s=args.z_s)
    dt = 1e3 * (time.perf_counter() - t)
    print(f"\n[demo] predict(z_s={args.z_s}) in {dt:.1f} ms — statistics:")
    for k in em.statistics:
        print(f"    {k:16s} {np.asarray(out[k]).shape}")
    print(f"    {'suppression':16s} {out['suppression'].shape} (derived S(ℓ)=C/C_DMO)")

    # 2) feedback response: sweep WindEnergyIn1e51erg across its prior
    fracs = np.linspace(0.0, 1.0, 7)
    vecs = np.array([bind.vary_param("WindEnergyIn1e51erg", fraction=f) for f in fracs])
    t = time.perf_counter()
    preds = [em.predict(v, z_s=args.z_s) for v in vecs]
    print(f"[demo] swept 7 feedback points in {1e3 * (time.perf_counter() - t):.1f} ms")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[demo] matplotlib unavailable — skipping figure")
        return
    ell = out["axes"]["ell"]
    mlb = out["axes"].get("log_mass_bins")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    cmap = plt.cm.viridis(fracs)
    for p, c, f in zip(preds, cmap, fracs):
        ax[0].semilogx(ell, p["suppression"], color=c, label=f"wind {f:.2f}")
        ax[1].loglog(ell, p["cl_yy"], color=c)
        if mlb is not None and "scaling_Y" in p:
            ax[2].semilogy(mlb, p["scaling_Y"], color=c, marker="o", ms=3)
    ax[0].axhline(1, ls=":", c="k", lw=0.8)
    ax[0].set(xlabel=r"$\ell$", ylabel=r"$S(\ell)=C_\ell/C_\ell^{\rm DMO}$",
              title=f"WL suppression ($z_s={args.z_s}$)")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set(xlabel=r"$\ell$", ylabel=r"$C_\ell^{yy}$", title="tSZ auto-power")
    ax[2].set(xlabel=r"$\log_{10}M$", ylabel=r"$Y_{500c}$", title="Y–M relation")
    fig.suptitle("bind.emulator: feedback response to WindEnergyIn1e51erg")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "emulator_demo_response.png", dpi=130)
    print(f"[demo] wrote {OUT / 'emulator_demo_response.png'}")


if __name__ == "__main__":
    main()
