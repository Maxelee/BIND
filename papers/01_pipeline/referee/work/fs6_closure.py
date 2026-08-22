#!/usr/bin/env python3
"""fidswap step 4 -- family-model fiducial closure with the NEW latents.

Predicts all 7 WL statistics from lambda (family_model.FamilyModel, the shipped
figs_preview/family_model_bundle.npz) and compares against the fiducial's MEASURED
curves, for the old fiducial and for each twobound replica.

The measured-curve recipe is the one stamped into figs_preview/amplitude_sets.npz
('notes'), reproduced here and GATED: running it on the shipped caches must
reproduce the shipped <stat>__fid_measured arrays.

    clk        field_cache kk_bind / runs/dmo/run_0000 Cl_kappa_paired cl_real,
               seed-paired per realization, then band-averaged onto the model's
               24 ell bands (predict_from_latents latent_model_coeffs ell_edges)
    pdf        field_cache pdf_bind, realization mean, moment-standardized onto
               the canonical 22-bin nu grid
    pk / mn    field_cache pk_bind / min_bind (68-bin dnu=0.25), realization mean,
               aggregated exactly 2:1 onto the canonical grid
    v0/v1/v2   the nu05 shard (bind_sb35/nu05_shards/sci_bind.npz for the old
               fiducial, the replica's own nu05_stats.npz for the new one)

PROVENANCE CORRECTION found by the gate: amplitude_sets' own 'notes' string says
v0/v1/v2 came from mf_cache interpolated onto the canonical grid.  They did not --
the mf_cache path misses the shipped curves by up to 1.4% of the curve scale on
9 bins, while the nu05 shard reproduces them BIT-FOR-BIT.  (V0 agrees either way.)
The nu05 route is also the one the Sobol design targets use, so it is the correct
apples-to-apples source; the mf_cache curve is kept as a printed cross-check.

    python referee/work/fs6_closure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(P1))
from family_model import FamilyModel  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
SCI = CEPH / "bind_science"
SB35 = CEPH / "bind_sb35"
FS = CEPH / "referee_work/fidswap"
ZI = 1
STATS = ("clk", "pdf", "pk", "mn", "v0", "v1", "v2")
AGG = [(8 + 2 * j, 9 + 2 * j) for j in range(22)]

fm = FamilyModel()
coef = np.load(P1 / "latent_model_coeffs.npz")
EDGb = coef["ell_edges"]
dsn = np.load(SB35 / "emulator_dataset_nu05.npz", allow_pickle=True)
NUg = np.asarray(dsn["a__pdf__pdf_bins"], float)
_eld = np.asarray(dsn["a__suppression__ell"], float)
DMO = np.load(SCI / "runs/dmo/run_0000/Cl_kappa_paired.npz")["cl_real"]
MF_NU8 = np.linspace(-3.0, 8.0, 45)


def band_reals(cl_real):
    S = cl_real[:, ZI, :] / DMO[:, ZI, :]
    return np.stack([np.nanmean(S[:, (_eld >= EDGb[i]) & (_eld < EDGb[i + 1])], 1)
                     for i in range(len(EDGb) - 1)], axis=1)


def pdf_to_nu(kb, p):
    norm = np.trapezoid(p, kb)
    mu = np.trapezoid(kb * p, kb) / norm
    sig = np.sqrt(np.trapezoid((kb - mu) ** 2 * p, kb) / norm)
    return sig * np.interp(mu + sig * NUg, kb, p, left=0.0, right=0.0)


def measured(fcp: Path, nu05p: Path, mfp: Path | None = None):
    """the 7 measured fiducial curves from a (field_cache, nu05) pair."""
    fc = np.load(fcp)
    nu5 = np.load(nu05p)
    assert np.allclose(np.asarray(nu5["nu"], float), NUg), "nu05 grid mismatch"
    Sr = band_reals(fc["kk_bind"])
    out = {"clk": Sr.mean(0), "clk_real": Sr,
           "pdf": pdf_to_nu(fc["pdf_bins"], np.nanmean(fc["pdf_bind"][:, ZI, :], 0))}
    for st, key in (("pk", "pk_bind"), ("mn", "min_bind")):
        c = np.nanmean(fc[key][:, ZI, :], 0)
        out[st] = np.array([c[i] + c[j] for i, j in AGG])
    for st in ("v0", "v1", "v2"):
        out[st] = np.asarray(nu5[f"mf_{st}"], float)[ZI]
        if mfp is not None:
            mf = np.load(mfp)
            out[f"{st}__mfcache"] = np.interp(NUg, MF_NU8, mf[f"bind_V{st[1]}_mean"][ZI])
    return out


def report(tag, lam, meas, ref=None):
    print(f"\n{'='*78}\n{tag}\n{'='*78}")
    print(f"{'stat':>5s} {'RMS(pred-meas)':>15s} {'med|d|/rng':>11s} "
          f"{'med |d|/sig_pred':>17s} {'max|z|':>8s}")
    res = {}
    for st in STATS:
        pred = fm.predict(lam, st)[0]
        m = meas[st]
        d = pred - m
        rng = np.nanmax(m) - np.nanmin(m)
        z = np.abs(d) / fm.predictive_sigma(st)
        res[st] = dict(pred=pred, meas=m, rms=float(np.sqrt(np.nanmean(d ** 2))),
                       relrng=float(np.nanmedian(np.abs(d)) / rng),
                       medz=float(np.nanmedian(z)), maxz=float(np.nanmax(z)))
        r = res[st]
        print(f"{st:>5s} {r['rms']:15.5g} {r['relrng']:11.4f} "
              f"{r['medz']:17.3f} {r['maxz']:8.2f}")
    S, P = res["clk"]["meas"], res["clk"]["pred"]
    k = int(np.nanargmin(S))
    print(f"\nS(ell) TROUGH  band {k} (ell ~ {coef['ell'][k]:.0f}): "
          f"measured {S[k]:.4f}   predicted {P[k]:.4f}   "
          f"residual {P[k]-S[k]:+.4f}")
    k19 = 19
    print(f"S(ell) band 19 (ell ~ {coef['ell'][k19]:.0f}): "
          f"measured {S[k19]:.4f}   predicted {P[k19]:.4f}   "
          f"residual {P[k19]-S[k19]:+.4f}")
    dep = 1 - S[k]
    print(f"closure error as a fraction of the suppression depth (1-S_min = "
          f"{dep:.4f}): {abs(P[k]-S[k])/dep*100:.1f}%")
    if "clk_real" in meas:
        se = meas["clk_real"].std(0, ddof=1) / np.sqrt(len(meas["clk_real"]))
        print(f"measured trough 50-real paired SE: {se[k]:.5f}  "
              f"-> residual = {abs(P[k]-S[k])/se[k]:.1f} sigma_meas")
    return res


def main():
    lz = np.load(FS / "lam_fidswap_v2.npz")
    lam_old = lz["lam_old"]
    ids = list(lz["replica_ids"])
    canon = int(lz["canonical"])

    # ── GATE: reproduce the shipped fid_measured curves ────────────────────
    old = measured(SCI / "field_cache/field_stats_fid.npz",
                   SB35 / "nu05_shards/sci_bind.npz",
                   SCI / "mf_cache/mf_nu8_snap096.npz")
    amp = np.load(P1 / "figs_preview/amplitude_sets.npz", allow_pickle=True)
    print("GATE -- measured-curve recipe vs shipped amplitude_sets fid_measured:")
    okall = True
    for st in STATS:
        ref = np.asarray(amp[f"{st}__fid_measured"], float)
        got = old[st]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = float(np.nanmax(np.abs(np.where(ref != 0, got / ref - 1, got - ref))))
        ok = rel < 1e-9
        okall &= ok
        print(f"  {st:>4s}: max|rel diff| = {rel:.3e}  {'PASS' if ok else 'CHECK'}")
    print("GATE OVERALL:", "PASS" if okall else "see above")

    report("OLD FIDUCIAL (bind_lightcone_tng, CAMELS cosmology) -- baseline", lam_old, old)

    new = measured(FS / f"field_cache/field_stats_fidtb{canon:02d}.npz",
                   SCI / f"runs/twobound/run_{canon:04d}/nu05_stats.npz",
                   FS / f"mf_cache/mf_nu8_snap096_fidtb{canon:02d}.npz")
    rn = report(f"NEW FIDUCIAL twobound/run_{canon:04d} (TNG300 cosmology)",
                lz["lam_canonical"], new)
    for st in ("v0", "v1", "v2"):
        d = np.nanmax(np.abs(new[f"{st}__mfcache"] - new[st]))
        print(f"  cross-check {st}: |nu05 - mf_cache| max {d:.3e} "
              f"({100*d/np.nanmax(np.abs(new[st])):.2f}% of curve scale)")

    # replica spread: BOTH legs (measured S(ell) and predicted S(ell))
    k = int(np.nanargmin(new["clk"]))
    print(f"\nreplica spread at the trough band {k} (ell ~ {coef['ell'][k]:.0f}):")
    print(f"{'replica':>9s} {'measured':>10s} {'predicted':>10s} {'residual':>10s}")
    preds, meas_r = [], []
    for i, r in enumerate(ids):
        p = fm.predict(lz["lam_replicas"][i], "clk")[0]
        Sr = band_reals(np.load(FS / f"field_cache/shards/kk_bind_tb{r:02d}.npz")["cl"])
        m = Sr.mean(0)
        preds.append(p[k])
        meas_r.append(m[k])
        print(f"     tb{r:02d} {m[k]:10.4f} {p[k]:10.4f} {p[k]-m[k]:+10.4f}")
    print(f"{'sd':>9s} {np.std(meas_r, ddof=1):10.4f} {np.std(preds, ddof=1):10.4f}")
    pm = fm.predict(lz["lam_trio_mean"], "clk")[0]
    print(f"trio-mean lambda -> predicted trough {pm[k]:.4f}; "
          f"trio-mean measured trough {np.mean(meas_r):.4f}; "
          f"residual {pm[k]-np.mean(meas_r):+.4f}")
    print(f"OLD fiducial for reference: measured {old['clk'][k]:.4f}, "
          f"predicted {fm.predict(lam_old, 'clk')[0][k]:.4f}, "
          f"residual {fm.predict(lam_old, 'clk')[0][k]-old['clk'][k]:+.4f}")

    np.savez(FS / "closure_fidswap.npz",
             ell=coef["ell"], nu=NUg, lat_names=lz["lat_names"],
             **{f"old_{k2}_{s}": v for s in STATS
                for k2, v in (("meas", old[s]),)},
             **{f"new_{k2}_{s}": v for s in STATS
                for k2, v in (("meas", new[s]), ("pred", rn[s]["pred"]))},
             **{f"oldpred_{s}": fm.predict(lam_old, s)[0] for s in STATS},
             clk_real_new=new["clk_real"], clk_real_old=old["clk_real"],
             lam_old=lam_old, lam_new=lz["lam_canonical"],
             trough_pred_replicas=np.array(preds))
    print(f"\nwrote {FS/'closure_fidswap.npz'}")


if __name__ == "__main__":
    main()
