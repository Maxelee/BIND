#!/usr/bin/env python
"""Experiment A: baryonify fiducial-ish halos from *observed* scaling relations.

Take a set of held-out CAMELS-TNG halos (their DMO image + their own
TNG-measured R200 observables), then re-paint each halo twice from the SAME DMO
and the SAME initial noise:

  baseline  -- condition on the halo's TNG-native observable vector  (= reconstruction)
  observed  -- condition on the same vector pushed onto the *observed* relation

The push is **ratio-anchored**: rather than plugging absolute literature numbers
into the model's idiosyncratic projected-aperture observable slots (which would
conflate a real signal with a unit/definition mismatch), we multiply each halo's
own observable by the dimensionless fractional offset the literature relation has
relative to TNG at that halo's M200.  Same DMO + same noise => every difference
in the painted field is caused purely by moving the observables onto the data.

Because the conditioning is integrated *to R200 in projection*, the quantities we
report are genuine predictions, not echoes of the inputs:
  - stacked Gas / Stars radial profiles  (shape, not just the R200 total)
  - cumulative baryon fraction f_b(<r)    at radii != R200
  - hydro-DM contraction  DM_hydro(<r) / DMO(<r)   (DM_hydro is an OUTPUT)

NOTE the circularity caveat: Mgas_200/Mstar_200/M_200 are themselves inputs, so
f_b *integrated to R200* is ~tautological -- read the inner-radius f_b(<r) and the
profile shapes, which are not conditioned.

Run on a GPU box with the torch3 kernel/venv:
    python examples/observable_paint_literature.py
Env overrides: OBS_RUN_DIR, OBS_CKPT, BIND_DATA_ROOT, N_HALOS, GAS_ONLY=1.
"""
import os
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bind.data import (
    load_file_list, AstroDataset, NormStats, THERMO_KEYS,
    OBSERVABLE_KEYS, N_OBS, PIX_MPC_H, compute_observables, r200_from_sample,
    thermo_forward,
)
from bind.train import FlowMatchingLit
from bind.inference.pipeline import _denormalize_to_physical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_DIR   = Path(os.environ.get("OBS_RUN_DIR", "/mnt/home/mlee1/ceph/fm_runs/fm_observables"))
CKPT      = os.environ.get("OBS_CKPT", "last.ckpt")
DATA_ROOT = Path(os.environ.get("BIND_DATA_ROOT", "/mnt/home/mlee1/ceph/train_data_rotated2_128_cpu"))
N_HALOS   = int(os.environ.get("N_HALOS", 64))
N_STEPS   = 50
BATCH     = 16
GAS_ONLY  = os.environ.get("GAS_ONLY", "0") == "1"
OUTDIR    = Path(os.environ.get("OBS_FIGDIR", "figures/observable_paint_literature"))
OUTDIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["DM_hydro", "Gas", "Stars"] + list(THERMO_KEYS)
IDX = {"DM": 0, "Gas": 1, "Stars": 2}

# --------------------------------------------------------------------------- #
# Literature scaling relations, expressed as the fractional offset from TNG    #
# at a given log10(M200c / [Msun/h]).  ratio = observed / TNG.  All knobs are  #
# editable; the gas deficit is the headline, the rest are secondary.           #
# --------------------------------------------------------------------------- #
def _gas_ratio(logm):
    """X-ray group gas-fraction deficit relative to TNG.

    Groups (~1e13) sit well below TNG in f_gas; the gap closes toward clusters
    (~1e14.5).  Bracketed by Sun+2009, Lovisari+2015, Eckert+2016 (XXL),
    eROSITA (Popesso+2024 / Bahar+2024).  Linear in logM, clipped to [0.45, 1].
    """
    return np.clip(0.45 + (logm - 13.0) * (1.0 - 0.45) / (14.5 - 13.0), 0.45, 1.0)


# observed / TNG ratios per observable.  Callables of logM (Msun/h).
LIT_RATIOS = {
    "Y_200":     lambda logm: 0.85,   # SZ pressure / hydrostatic-bias deficit (Planck 2013 XX; Henson+17)
    "Mgas_200":  _gas_ratio,          # headline X-ray gas deficit
    "Mstar_200": lambda logm: 0.90,   # SHMR slightly below TNG (Kravtsov+18; Behroozi+19)
    "Tx_200":    lambda logm: 0.95,   # T_X normalization (Vikhlinin+09; Lovisari+15)
    "K_200":     lambda logm: 1.15,   # group entropy excess (Sun+09; Pratt+10)
    "P_200":     lambda logm: 0.85,   # pressure, tracks Y
    "M_200":     lambda logm: 1.00,   # lensing anchor -- held fixed
}
if GAS_ONLY:  # isolate the most robust knob
    LIT_RATIOS = {k: (LIT_RATIOS[k] if k == "Mgas_200" else (lambda lm: 1.0)) for k in OBSERVABLE_KEYS}


def lit_ratio_vector(logm):
    """(N_OBS,) observed/TNG ratio vector at a halo mass."""
    return np.array([float(LIT_RATIOS[k](logm)) for k in OBSERVABLE_KEYS], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Aperture helpers (physical radii in Mpc/h)                                   #
# --------------------------------------------------------------------------- #
def cumulative_in_radii(field, radii_mpc):
    """Cumulative sum of `field` within each circular aperture radius (Mpc/h)."""
    n = field.shape[-1]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx - c, yy - c) * PIX_MPC_H
    return np.array([float(field[rr < r].sum()) for r in radii_mpc])


def load_model():
    ns = NormStats.load(RUN_DIR / "norm_stats.npz")
    assert ns.condition_observables, "run is not observable-conditioned"
    ckpt = RUN_DIR / "checkpoints" / CKPT
    model = FlowMatchingLit.load_from_checkpoint(str(ckpt), map_location=device).eval().to(device)
    ep = re.search(r"epoch(\d+)", ckpt.name)
    print(f"loaded {ckpt.name} (epoch {ep.group(1) if ep else '?'}); "
          f"predict_thermo={ns.predict_thermo} stars_two_head={ns.stars_two_head}")
    return model.fm, ns


def paint(fm, ns, cond, ls, obs_norm, seed):
    """Paint a batch with fixed noise.  obs_norm: (B, N_OBS) normalized."""
    torch.manual_seed(seed)
    with torch.no_grad():
        g = fm.sample(cond, ls, torch.from_numpy(obs_norm.astype(np.float32)).to(device),
                      n_steps=N_STEPS)
    return _denormalize_to_physical(g.float().cpu().numpy(), ns)  # (B, 3+thermo, H, W)


def main():
    fm, ns = load_model()
    files = load_file_list(str(DATA_ROOT), "test")
    ds = AstroDataset(files, ns)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(N_HALOS, len(ds)), replace=False)
    print(f"{len(idx)} halos | scenario: {'GAS_ONLY' if GAS_ONLY else 'full literature set'}")

    radii = np.linspace(0.1, 3.0, 24)          # Mpc/h apertures for f_b(<r) / contraction
    base_all, obs_all, dmo_all, r200_all = [], [], [], []

    for s in range(0, len(idx), BATCH):
        sub = idx[s:s + BATCH]
        items = [ds[i] for i in sub]
        cond = torch.stack([it["condition"] for it in items]).to(device)
        ls   = torch.stack([it["large_scale"] for it in items]).to(device)

        base_obs_norm = np.stack([it["params"].numpy() for it in items])      # already normalized TNG obs
        # Build the observed-relation conditioning: recover raw obs, scale, renormalize.
        obs_obs_norm = np.empty_like(base_obs_norm)
        dmo_maps = []
        for k, i in enumerate(sub):
            d = np.load(files[i])
            raw = compute_observables(d)                                       # physical TNG obs
            logm = np.log10(float(d["halo_mass"]))
            shifted = raw * lit_ratio_vector(logm)
            obs_obs_norm[k] = thermo_forward(shifted[None], ns.obs_mean[None], ns.obs_std[None],
                                             ns.obs_floor[None])[0]
            dmo_maps.append(d["condition"].astype(np.float32))
            r200_all.append(r200_from_sample(d))

        g_base = paint(fm, ns, cond, ls, base_obs_norm, seed=1000 + s)
        g_obs  = paint(fm, ns, cond, ls, obs_obs_norm,  seed=1000 + s)         # same seed -> same noise
        base_all.append(g_base[:, :3])
        obs_all.append(g_obs[:, :3])
        dmo_all.extend(dmo_maps)

    base = np.concatenate(base_all)   # (N,3,H,W) [DM_hydro, Gas, Stars]
    obs  = np.concatenate(obs_all)
    dmo  = np.stack(dmo_all)          # (N,H,W) DMO
    r200 = np.array(r200_all)
    print(f"painted: base {base.shape} obs {obs.shape}; median R200 = {np.median(r200):.2f} Mpc/h")

    # ---- cumulative profiles per halo ---------------------------------------
    def cum(stack, ch):
        return np.stack([cumulative_in_radii(stack[i, ch], radii) for i in range(len(stack))])

    cb = {n: cum(base, IDX[n]) for n in IDX}
    co = {n: cum(obs, IDX[n]) for n in IDX}
    cdmo = np.stack([cumulative_in_radii(dmo[i], radii) for i in range(len(dmo))])

    fb_base = (cb["Gas"] + cb["Stars"]) / (cb["DM"] + cb["Gas"] + cb["Stars"] + 1e-30)
    fb_obs  = (co["Gas"] + co["Stars"]) / (co["DM"] + co["Gas"] + co["Stars"] + 1e-30)
    contr_base = cb["DM"] / (cdmo + 1e-30)
    contr_obs  = co["DM"] / (cdmo + 1e-30)

    rmed = np.median(r200)

    # ---- figure 1: f_b(<r) --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(radii, np.median(fb_base, 0), "k-", label="baseline (TNG obs)")
    ax.fill_between(radii, *np.percentile(fb_base, [16, 84], 0), color="k", alpha=0.15)
    ax.plot(radii, np.median(fb_obs, 0), "r--", label="observed relation")
    ax.fill_between(radii, *np.percentile(fb_obs, [16, 84], 0), color="r", alpha=0.15)
    ax.axvline(rmed, ls=":", color="gray", lw=1, label=f"median R200={rmed:.2f}")
    ax.axhline(0.157, ls=":", color="C0", lw=1, label="cosmic Ωb/Ωm")
    ax.set_xlabel("r [Mpc/h]"); ax.set_ylabel("projected f_b(<r)")
    ax.set_title("Baryon fraction profile: TNG vs observed-relation painting")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUTDIR / "fb_profile.png", dpi=130); plt.close(fig)

    # ---- figure 2: DM contraction ------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(radii, np.median(contr_base, 0), "k-", label="baseline (TNG obs)")
    ax.plot(radii, np.median(contr_obs, 0), "r--", label="observed relation")
    ax.axvline(rmed, ls=":", color="gray", lw=1)
    ax.axhline(1.0, ls=":", color="gray", lw=1)
    ax.set_xlabel("r [Mpc/h]"); ax.set_ylabel("DM_hydro(<r) / DMO(<r)")
    ax.set_title("Hydro DM contraction response to observed baryons")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUTDIR / "dm_contraction.png", dpi=130); plt.close(fig)

    # ---- figure 3: gas/star profile ratio ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, n in zip(axes, ["Gas", "Stars"]):
        ratio = co[n] / (cb[n] + 1e-30)
        ax.plot(radii, np.median(ratio, 0), "C2-")
        ax.fill_between(radii, *np.percentile(ratio, [16, 84], 0), color="C2", alpha=0.2)
        ax.axhline(1.0, ls=":", color="gray"); ax.axvline(rmed, ls=":", color="gray", lw=1)
        ax.set_xlabel("r [Mpc/h]"); ax.set_ylabel(f"{n}(<r) observed / baseline")
        ax.set_title(f"{n} cumulative-mass response")
    fig.tight_layout(); fig.savefig(OUTDIR / "mass_response.png", dpi=130); plt.close(fig)

    # ---- summary printout (read these, esp. inner radii) --------------------
    def at(r):  # nearest aperture index
        return int(np.argmin(np.abs(radii - r)))
    print("\n  r[Mpc/h]   f_b base   f_b obs    Δf_b     DMcontr base/obs")
    for r in (0.2, 0.5, 1.0, rmed, 2.0):
        i = at(r)
        print(f"  {radii[i]:7.2f}  {np.median(fb_base[:, i]):8.3f}  "
              f"{np.median(fb_obs[:, i]):8.3f}  {np.median(fb_obs[:, i]-fb_base[:, i]):+7.3f}   "
              f"{np.median(contr_base[:, i]):.3f}/{np.median(contr_obs[:, i]):.3f}")
    print(f"\nfigures -> {OUTDIR}/")


if __name__ == "__main__":
    main()
