"""r5_truth.py -- TASK 4: the zero-generative-noise cross-provenance check.

Measure the 8 latents from the TRUTH atlas (truth_snap096.npz -- the SAME
2933 halos, but with the full-hydro TNG300 patches pasted in place of BIND's
generated ones, so ZERO generative draw noise), push them through the SHIPPED
model, and compare against
  (a) the predictions from lambda measured on BIND's own fiducial paint, and
  (b) the measured fiducial curves (BIND paint and TNG300 hydro truth).

Conventions follow _build_family_tutorial_nb.py section 3 verbatim: the
seed-paired per-realization suppression ratio band-averaged into the model
bands, and the moment-standardized kappa PDF.

Outputs -> /mnt/home/mlee1/ceph/referee_work/r5/r5_truth.npz
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline")
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline/referee/work")
from family_model import FamilyModel                              # noqa: E402
from r5_common import (CEPH, LAT_NAMES, OUT, load_fid_atlas,      # noqa: E402
                       load_truth, measure_lambda)

SCI = CEPH / "bind_science"
fm = FamilyModel()
dsn = np.load(CEPH / "bind_sb35/emulator_dataset_nu05.npz", allow_pickle=True)
coef = np.load("/mnt/home/mlee1/BIND/papers/01_pipeline/latent_model_coeffs.npz")
EDGb, _eld = coef["ell_edges"], np.asarray(dsn["a__suppression__ell"], float)

lam_fid = measure_lambda(load_fid_atlas())
lam_tru = measure_lambda(load_truth())
boot = np.load(OUT / "r5_bootstrap.npz")
rep = np.load(OUT / "r5_replicas.npz")
SIG_A, SIG_REP = boot["sigA_fid"], rep["SIG_REP"]
SIG_DES = boot["SIG_DES"]

print("lambda measured from BIND's fiducial paint vs from the HYDRO TRUTH "
      "patches (same 2933 halos)")
print(f"{'latent':>20s} {'BIND':>12s} {'TRUTH':>12s} {'Delta':>11s} "
      f"{'/sig_boot':>10s} {'/sig_rep':>9s} {'/sig_design':>12s}")
d = lam_fid - lam_tru
for i, n in enumerate(LAT_NAMES):
    print(f"{n:>20s} {lam_fid[i]:12.5f} {lam_tru[i]:12.5f} {d[i]:11.4g} "
          f"{d[i]/SIG_A[i]:10.2f} {d[i]/SIG_REP[i]:9.2f} "
          f"{d[i]/SIG_DES[i]:12.3f}")
print(f"  rms |Delta| / sigma_design = {np.sqrt((( d/SIG_DES)**2).mean()):.3f}"
      f"   max = {np.abs(d/SIG_DES).max():.3f}")

# ── measured fiducial curves (tutorial conventions) ───────────────────────
fc = np.load(SCI / "field_cache/field_stats_fid.npz")
dmo = np.load(SCI / "runs/dmo/run_0000/Cl_kappa_paired.npz")


def band_reals(cl_real):
    S = cl_real[:, 1, :] / dmo["cl_real"][:, 1, :]
    return np.stack([np.nanmean(S[:, (_eld >= EDGb[i]) & (_eld < EDGb[i + 1])],
                                1) for i in range(len(EDGb) - 1)], axis=1)


def pdf_to_nu(kb, p):
    norm = np.trapezoid(p, kb)
    mu = np.trapezoid(kb * p, kb) / norm
    sig = np.sqrt(np.trapezoid((kb - mu) ** 2 * p, kb) / norm)
    return sig * np.interp(mu + sig * fm.grid("pdf"), kb, p, left=0, right=0)


S_bind = band_reals(fc["kk_bind"]).mean(0)
S_true = band_reals(fc["kk_truth"]).mean(0)
pdf_bind = pdf_to_nu(fc["pdf_bins"], np.nanmean(fc["pdf_bind"][:, 1, :], 0))
pdf_true = pdf_to_nu(fc["pdf_bins"], np.nanmean(fc["pdf_truth"][:, 1, :], 0))
MEAS = {"clk": S_bind, "pdf": pdf_bind}
TRUE = {"clk": S_true, "pdf": pdf_true}

print("\nPREDICTIONS from the two lambda provenances, all 7 statistics")
print(f"{'stat':>5s} {'med|dpred|/sig_pred':>20s} {'max|dpred|/sig_pred':>20s}"
      f" {'med rel dpred':>14s}  (bins used)")
out = {}


def _finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


for st in fm.stats:
    pb = fm.predict(lam_fid, st)[0]
    pt = fm.predict(lam_tru, st)[0]
    sp = fm.predictive_sigma(st)
    out[st] = (pb, pt, sp)
    ok = sp > 0                       # sig_pred vanishes in empty tail bins
    with np.errstate(divide="ignore", invalid="ignore"):
        z = _finite(np.abs(pt - pb)[ok] / sp[ok])
        rel = _finite(np.abs(pt - pb)[np.abs(pb) > 0]
                      / np.abs(pb)[np.abs(pb) > 0])
    print(f"{st:>5s} {np.median(z):20.4f} {np.max(z):20.4f} "
          f"{100*np.median(rel):13.3f}%  ({ok.sum()}/{len(sp)})")

xg = fm.grid("clk")
pb, pt, sp = out["clk"]
tro = int(np.argmin(S_bind))
print(f"\nS(ell) fiducial closure (out-of-design), ell~{xg[tro]:.0f} trough "
      f"(band {tro}):")
print(f"  measured BIND fiducial paint      {S_bind[tro]:.4f}")
print(f"  measured TNG300 full hydro        {S_true[tro]:.4f}")
print(f"  model from lambda(BIND fid)       {pb[tro]:.4f}   "
      f"({abs(pb[tro]-S_bind[tro])/sp[tro]:.2f} sigma_pred)")
print(f"  model from lambda(HYDRO TRUTH)    {pt[tro]:.4f}   "
      f"({abs(pt[tro]-S_bind[tro])/sp[tro]:.2f} sigma_pred)")
print(f"  sigma_pred at the trough          {sp[tro]:.4f}")
print(f"  swapping the lambda provenance moves the trough by "
      f"{abs(pt[tro]-pb[tro]):.4f} = {abs(pt[tro]-pb[tro])/sp[tro]:.2f} "
      f"sigma_pred = {100*abs(pt[tro]-pb[tro])/abs(1-S_bind[tro]):.1f}% of "
      "the suppression signal")
for st in ("clk", "pdf"):
    pb_, pt_, _ = out[st]
    mb, mt = MEAS[st], TRUE[st]
    g = (np.abs(mb) > 0) & (np.abs(mt) > 0)      # PDF tails contain exact 0s
    db = 100 * np.median(np.abs(pb_ - mb)[g] / np.abs(mb)[g])
    dt = 100 * np.median(np.abs(pt_ - mb)[g] / np.abs(mb)[g])
    dbr = 100 * np.median(np.abs(pb_ - mt)[g] / np.abs(mt)[g])
    dtr = 100 * np.median(np.abs(pt_ - mt)[g] / np.abs(mt)[g])
    print(f"  {st}: median |model - measured| [%]   "
          f"lam_BIND->BIND {db:.2f} | lam_TRUTH->BIND {dt:.2f} | "
          f"lam_BIND->HYDRO {dbr:.2f} | lam_TRUTH->HYDRO {dtr:.2f}")

np.savez(OUT / "r5_truth.npz", lam_fid=lam_fid, lam_truth=lam_tru,
         lat_names=np.array(LAT_NAMES), S_bind=S_bind, S_true=S_true,
         clk_grid=xg, pdf_grid=fm.grid("pdf"), pdf_bind=pdf_bind,
         pdf_true=pdf_true, trough=tro,
         **{f"{st}__pred_bind": out[st][0] for st in fm.stats},
         **{f"{st}__pred_truth": out[st][1] for st in fm.stats},
         **{f"{st}__sig_pred": out[st][2] for st in fm.stats})
print(f"\nwrote {OUT/'r5_truth.npz'}")
