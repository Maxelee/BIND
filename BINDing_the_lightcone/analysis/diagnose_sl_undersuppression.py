"""Channel-level autopsy of the S(ell) under-suppression.

Question (2026-08-14): with the corrected fiducial, BIND under-suppresses
S(ell) by ~2.5% at the trough relative to the hydro-pasted truth, while the
integrated halo masses and the 9-bin radial profiles all validate. Where does
the excess small-scale power live?

Method: the patch level is where BIND and the truth are exactly comparable --
both runs store per-halo Sigma patches [DM, gas, stars] at IDENTICAL halos on
identical 128^2 grids (6.25 Mpc/h, 48.8 kpc/h px), and stage1 stores the DMO
conditioning patch of every halo. Stack 2D power spectra per channel over all
halos of snap_096 and form ratios. Everything that follows is a ratio of
identically produced quantities, so windowing and gridding cancel.

Decompositions:
  1. per-channel P(k):  P_BIND / P_truth  for DM / gas / stars / total
  2. stellar ablation:  the same ratio for total vs (DM+gas) only
  3. DM back-reaction:  P_DM / P_DMOcond  for truth and for BIND
  4. central stellar concentration: M_star(<2px)/M_star(patch)
  5. mass dependence: split at M200c = 10^13.5

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python diagnose_sl_undersuppression.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
B49 = CEPH / "bind_science/runs/twobound/run_0049/snap_096"
TRU = CEPH / "bind_science/runs/truth/run_0000/snap_096"
ST1 = CEPH / "bind_lightcone_tng/snap_096/stage1"
OUT = Path(__file__).resolve().parent
NPIX, LPATCH = 128, 6.25          # Mpc/h

kx = np.fft.fftfreq(NPIX, d=LPATCH / NPIX) * 2 * np.pi
kk = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
KEDGE = np.linspace(1.0, 55.0, 28)
KC = 0.5 * (KEDGE[1:] + KEDGE[:-1])
KIDX = np.digitize(kk.ravel(), KEDGE) - 1


def pk(field2d):
    """azimuthal mean |FFT|^2 of a (batch of) patches; returns (n, nk)."""
    f = np.fft.fft2(field2d, axes=(-2, -1))
    p = (f * np.conj(f)).real.reshape(len(field2d), -1)
    out = np.empty((len(field2d), len(KC)))
    for b in range(len(KC)):
        m = KIDX == b
        out[:, b] = p[:, m].mean(1) if m.any() else np.nan
    return out


acc = {c: {"P": [], "M": []} for c in
       ("bind_dm", "bind_gas", "bind_st", "bind_tot", "bind_dgs",
        "tru_dm", "tru_gas", "tru_st", "tru_tot", "tru_dgs", "dmo")}
cen = {"bind": [], "tru": [], "M": []}
yy, xx = np.mgrid[:NPIX, :NPIX]
rpix = np.sqrt((yy - NPIX // 2) ** 2 + (xx - NPIX // 2) ** 2)
CEN = rpix <= 2.0

for sl in range(4):
    b = np.load(B49 / f"composite_slab{sl:02d}.npz")
    t = np.load(TRU / f"composite_slab{sl:02d}.npz")
    s = np.load(ST1 / f"stage1_slab{sl:02d}.npz")
    assert np.allclose(b["halo_centers"], t["halo_centers"]), "halo mismatch"
    assert np.allclose(b["halo_centers"], s["halo_centers"]), "stage1 mismatch"
    gb, gt = np.asarray(b["generated_patches"], np.float64), np.asarray(t["generated_patches"], np.float64)
    cond = np.asarray(s["condition"], np.float64)
    M = np.asarray(b["halo_masses"], np.float64)
    for tag, arr in (("bind", gb), ("tru", gt)):
        dm, gas, st = arr[:, 0], arr[:, 1], arr[:, 2]
        for c, f in ((f"{tag}_dm", dm), (f"{tag}_gas", gas), (f"{tag}_st", st),
                     (f"{tag}_tot", dm + gas + st), (f"{tag}_dgs", dm + gas)):
            acc[c]["P"].append(pk(f))
            acc[c]["M"].append(M)
        tot = st.reshape(len(st), -1).sum(1)
        cen[tag].append(np.where(tot > 0, (st * CEN).reshape(len(st), -1).sum(1) / np.where(tot > 0, tot, 1), np.nan))
    acc["dmo"]["P"].append(pk(cond))
    acc["dmo"]["M"].append(M)
    cen["M"].append(M)
    print(f"slab {sl}: {len(M)} halos")

P = {c: np.concatenate(v["P"]) for c, v in acc.items()}
M = np.concatenate(acc["dmo"]["M"])
CENb, CENt = np.concatenate(cen["bind"]), np.concatenate(cen["tru"])
hi = M >= 10 ** 13.5

print(f"\ntotal halos {len(M)}, high-mass (>=10^13.5) {hi.sum()}")


def ratio(a, b, sel=None):
    sel = slice(None) if sel is None else sel
    return np.nanmean(P[a][sel], 0) / np.nanmean(P[b][sel], 0)


def show(title, pairs, sel=None, ks=(2.5, 5, 8, 12, 20, 30)):
    print(f"\n--- {title} ---")
    idx = [int(np.argmin(np.abs(KC - k))) for k in ks]
    print(f"{'ratio':34s}" + "".join(f"  k={KC[i]:5.1f}" for i in idx))
    for lab, a, b in pairs:
        r = ratio(a, b, sel)
        print(f"{lab:34s}" + "".join(f"  {r[i]:7.4f}" for i in idx))


show("per-channel P_BIND / P_truth (all halos)", [
    ("DM channel", "bind_dm", "tru_dm"),
    ("gas channel", "bind_gas", "tru_gas"),
    ("stars channel", "bind_st", "tru_st"),
    ("total (DM+gas+stars)", "bind_tot", "tru_tot"),
    ("total w/o stars (DM+gas)", "bind_dgs", "tru_dgs")])

show("per-channel P_BIND / P_truth (M >= 10^13.5)", [
    ("DM channel", "bind_dm", "tru_dm"),
    ("gas channel", "bind_gas", "tru_gas"),
    ("stars channel", "bind_st", "tru_st"),
    ("total (DM+gas+stars)", "bind_tot", "tru_tot"),
    ("total w/o stars (DM+gas)", "bind_dgs", "tru_dgs")], sel=hi)

show("back-reaction: P_DM / P_DMO-conditioning", [
    ("truth DM / DMO", "tru_dm", "dmo"),
    ("BIND DM / DMO", "bind_dm", "dmo")])

show("patch-level suppression: P_total / P_DMO", [
    ("truth (the target)", "tru_tot", "dmo"),
    ("BIND (the model)", "bind_tot", "dmo")])

print("\n--- central stellar concentration  M_star(r<2px)/M_star(patch) ---")
ok = np.isfinite(CENb) & np.isfinite(CENt)
print(f"  median  BIND {np.median(CENb[ok]):.4f}   truth {np.median(CENt[ok]):.4f}")
print(f"  mean    BIND {np.mean(CENb[ok]):.4f}   truth {np.mean(CENt[ok]):.4f}")
print(f"  high-M  BIND {np.median(CENb[ok & hi]):.4f}   truth {np.median(CENt[ok & hi]):.4f}")

np.savez(OUT / "sl_undersuppression_channels.npz",
         k=KC, M=M, hi=hi,
         **{f"P_{c}": np.nanmean(P[c], 0) for c in P},
         **{f"Phi_{c}": np.nanmean(P[c][hi], 0) for c in P},
         cen_bind=CENb, cen_tru=CENt)
print(f"\nwrote {OUT / 'sl_undersuppression_channels.npz'}")
