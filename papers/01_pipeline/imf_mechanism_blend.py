#!/usr/bin/env python
"""imf_mechanism_blend.py -- generalize the quick Delta-C_ell shape test:
decompose EVERY twobound parameter's bound-to-bound fractional response shape
onto a 2-template basis (AGN = BlackHoleRadiativeEfficiency, wind =
VariableWindVelFactor), for 7 statistics:

    Cl_kappa (z_s=1), Cl_yy, N_pk(nu), N_min(nu), V0, V1, V2(nu)

Per parameter p and statistic s:  shape_p = a * shape_AGN + b * shape_wind
(all shapes unit-peak normalized; a,b absorb knob direction, so a<0 means
"acts like the AGN template reversed").  Output fig21d: one (a,b) scatter per
statistic, family-colored, IMFslope highlighted; final panel = IMFslope's
(a,b) trajectory across statistics.

Data: twobound paired_stats.npz (fractional paired responses + errors; pk/min
band-integrated from per-realization cubes per the bind-paired-stats warning)
and per-run Cl_kappa_y.npz cl_yy (pre-xpkfix norm cancels in the ratio to the
fiducial's cl_yy -- ratio use only, per the stale-caches note).  Three params
(VariableWindSpecMomentum, UVBH0Deltaz, UVBHepDeltaz) have one fiducial-valued
run: their Delta is a one-sided (bound - fid) response, flagged in output.

Run from papers/01_pipeline:
    /mnt/home/mlee1/venvs/BIND_env/bin/python imf_mechanism_blend.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
from paper_style import setup, save, panel_label, COLORS  # noqa: E402
from param_labels import short_label  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
TB = CEPH / "bind_science/runs/twobound"
FID = CEPH / "bind_science/runs/bind/run_0000"
assert Path.cwd().name == "01_pipeline", "run from papers/01_pipeline/"

T_AGN, T_WIND = "BlackHoleRadiativeEfficiency", "VariableWindVelFactor"
ZI = 1                       # z_s = 1
SNR_MIN = 3.0                # peak-|Delta| / err gate for inclusion
R2_OPEN = 0.7                # below this the marker is drawn open


def family(name):
    if any(k in name for k in ("BlackHole", "Quasar", "Radio")):
        return "agn"
    if "Wind" in name or "SN" in name:
        return "wind"
    return "other"


FAMCOL = {"wind": COLORS["bind"], "agn": COLORS["highlight"], "other": "#555555"}

# ── decode all 30 pairs ──────────────────────────────────────────────────────
names35 = list(pd.read_csv(
    "/mnt/home/mlee1/BIND/src/bind/assets/SB35_param_minmax.csv")["ParamName"])
tbp = np.load(TB / "twobound_params.npy")
fidp = np.median(tbp, axis=0)
pairs = {}
for i in range(30):
    d = np.where(tbp[2 * i] != tbp[2 * i + 1])[0]
    assert len(d) == 1
    j = int(d[0])
    rr = [2 * i, 2 * i + 1]
    vv = [float(tbp[r, j]) for r in rr]
    o = np.argsort(vv)
    pairs[names35[j]] = dict(
        runs=[rr[k] for k in o], vals=[vv[k] for k in o],
        one_sided=any(np.allclose(tbp[r], fidp, rtol=1e-6, atol=0) for r in rr))
ONE_SIDED = sorted(n for n, p in pairs.items() if p["one_sided"])
print(f"decoded {len(pairs)} params; one-sided (bound vs fid): {ONE_SIDED}")

# ── load per-run products ────────────────────────────────────────────────────
P = {r: np.load(TB / f"run_{r:04d}/paired_stats.npz") for r in range(60)}
CY = {r: np.load(TB / f"run_{r:04d}/Cl_kappa_y.npz") for r in range(60)}
cy_fid = np.load(FID / "Cl_kappa_y.npz")
fidcube = np.load(FID / "paired_perreal_fid.npz")
ell, nu, mf_nu = P[0]["ell"], P[0]["nu"], P[0]["mf_nu"]

# nu-band integration for count statistics (0.5-wide bands, fid >= 2 per map)
KNU = 2


def _coarse(c):
    n = (c.shape[-1] // KNU) * KNU
    return c[..., :n].reshape(*c.shape[:-1], n // KNU, KNU).sum(-1)


nuc = nu[:(len(nu) // KNU) * KNU].reshape(-1, KNU).mean(-1)
BANDS = {}
for key in ("pk", "min"):
    fidc = _coarse(fidcube[key][:, ZI].astype(float))
    BANDS[key] = dict(fidc=fidc, mask=fidc.mean(0) >= 2.0)


def count_resp(r, key):
    fidc, mask = BANDS[key]["fidc"], BANDS[key]["mask"]
    rc = _coarse(P[r][f"{key}_real"][:, ZI].astype(float))
    n = min(len(rc), len(fidc))
    fm = fidc[:n].mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        resp = (rc[:n].mean(0) - fm) / fm
        err = (rc[:n] - fidc[:n]).std(0) / np.sqrt(n) / fm
    return nuc[mask], resp[mask], err[mask]


# ell log-binning for the spectra
NBIN = 24
EDGES = np.geomspace(100.0, 2e4, NBIN + 1)


def logbin(y, e):
    ib = np.digitize(ell, EDGES) - 1
    L, Y, E = [], [], []
    for b in range(NBIN):
        s = ib == b
        if not s.any():
            continue
        L.append(np.exp(np.log(ell[s]).mean()))
        Y.append(np.nanmean(y[s]))
        E.append(np.sqrt(np.nansum(e[s] ** 2)) / s.sum())
    return np.array(L), np.array(Y), np.array(E)


def delta(nm, stat):
    """(x, Delta, err): bound-to-bound fractional response difference."""
    rlo, rhi = pairs[nm]["runs"]
    if stat == "clk":
        d = P[rhi]["clk_resp"][ZI] - P[rlo]["clk_resp"][ZI]
        e = np.sqrt(P[rhi]["clk_err"][ZI] ** 2 + P[rlo]["clk_err"][ZI] ** 2)
        return logbin(d, e)
    if stat == "clyy":
        f = cy_fid["cl_yy"]
        d = (CY[rhi]["cl_yy"] - CY[rlo]["cl_yy"]) / f
        e = np.sqrt(CY[rhi]["cl_yy_err"] ** 2 + CY[rlo]["cl_yy_err"] ** 2) / f
        return logbin(d, e)
    if stat in ("pk", "min"):
        x, rhi_r, ehi = count_resp(rhi, stat)
        _, rlo_r, elo = count_resp(rlo, stat)
        return x, rhi_r - rlo_r, np.sqrt(ehi ** 2 + elo ** 2)
    # Minkowski functionals: resp already normalized by max|fid|
    d = P[rhi][f"{stat}_resp"][ZI] - P[rlo][f"{stat}_resp"][ZI]
    e = np.sqrt(P[rhi][f"{stat}_err"][ZI] ** 2 + P[rlo][f"{stat}_err"][ZI] ** 2)
    return mf_nu, d, e


STATS = [("clk", r"$C_\ell^{\kappa\kappa}$"), ("clyy", r"$C_\ell^{yy}$"),
         ("pk", r"$N_{\rm pk}(\nu)$"), ("min", r"$N_{\rm min}(\nu)$"),
         ("V0", r"$V_0(\nu)$"), ("V1", r"$V_1(\nu)$"), ("V2", r"$V_2(\nu)$")]


def shapenorm(y):
    k = int(np.argmax(np.abs(y)))
    return y / y[k]


# ── fit every param onto the 2-template basis, per statistic ─────────────────
RES = {}                     # stat -> {param: (a, b, r2, snr)}
TMPL_R = {}                  # stat -> template shape collinearity
for stat, _ in STATS:
    xT, dA, eA = delta(T_AGN, stat)
    _, dW, eW = delta(T_WIND, stat)
    tA, tW = shapenorm(dA), shapenorm(dW)
    snrA = np.max(np.abs(dA)) / eA[np.argmax(np.abs(dA))]
    snrW = np.max(np.abs(dW)) / eW[np.argmax(np.abs(dW))]
    A = np.column_stack([tA, tW])
    out = {}
    for nm in pairs:
        if nm in (T_AGN, T_WIND):
            continue
        _, dP, eP = delta(nm, stat)
        k = int(np.argmax(np.abs(dP)))
        snr = np.abs(dP[k]) / eP[k]
        s = shapenorm(dP)
        w, *_ = np.linalg.lstsq(A, s, rcond=None)
        pred = A @ w
        r2 = 1 - np.sum((s - pred) ** 2) / np.sum((s - s.mean()) ** 2)
        out[nm] = (float(w[0]), float(w[1]), float(r2), float(snr))
    RES[stat] = out
    tmpl_r = float(np.corrcoef(tA, tW)[0, 1])
    TMPL_R[stat] = tmpl_r
    print(f"[{stat:4s}] template S/N: AGN {snrA:.0f}, wind {snrW:.0f}; "
          f"template shape r = {tmpl_r:+.2f}; "
          f"params passing S/N>={SNR_MIN:g}: "
          f"{sum(v[3] >= SNR_MIN for v in out.values())}/{len(out)}")

# ── console table for the passing params ─────────────────────────────────────
for stat, lab in STATS:
    rows = sorted(((nm, *v) for nm, v in RES[stat].items() if v[3] >= SNR_MIN),
                  key=lambda r: -abs(r[1]) - abs(r[2]))
    print(f"\n--- {stat}: a*BHRadEff + b*VarWindVel (S/N>={SNR_MIN:g}) ---")
    for nm, a, b, r2, snr in rows:
        os_flag = " (one-sided)" if pairs[nm]["one_sided"] else ""
        print(f"  {short_label(nm):20s} a={a:+.2f} b={b:+.2f} R2={r2:.2f} "
              f"S/N={snr:.0f}{os_flag}")

# ── figure: 2x4 panels ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.3))
axes = axes.ravel()
LIM = 2.0
for k, (stat, lab) in enumerate(STATS):
    ax = axes[k]
    for nm, (a, b, r2, snr) in RES[stat].items():
        if snr < SNR_MIN:
            continue
        a, b = np.clip(a, -LIM, LIM), np.clip(b, -LIM, LIM)
        fc = FAMCOL[family(nm)]
        if nm == "IMFslope":
            ax.scatter(a, b, marker="*", s=70, color="#111111", zorder=6,
                       edgecolors="w", linewidths=0.4)
        else:
            filled = r2 >= R2_OPEN
            ax.scatter(a, b, s=13, facecolors=fc if filled else "none",
                       edgecolors=fc, linewidths=0.7, zorder=4)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.plot([-LIM, LIM], [-LIM, LIM], color="0.9", lw=0.5, ls=":", zorder=1)
    ax.set_xlim(-LIM * 1.05, LIM * 1.05)
    ax.set_ylim(-LIM * 1.05, LIM * 1.05)
    ax.set_xlabel("$a$ (AGN: BHRadEff)", fontsize=6.5)
    ax.set_ylabel("$b$ (wind: VarWindVel)", fontsize=6.5)
    ax.tick_params(labelsize=6)
    panel_label(ax, f"({'abcdefg'[k]}) {lab}")
    ax.text(0.96, 0.96, f"$r_{{\\rm templ}}={TMPL_R[stat]:+.2f}$",
            transform=ax.transAxes, fontsize=5.2, ha="right", va="top", color="0.35")

# final panel: IMFslope trajectory across statistics
ax = axes[7]
for stat, lab in STATS:
    a, b, r2, snr = RES[stat]["IMFslope"]
    ok = snr >= SNR_MIN
    ax.scatter(a, b, marker="*", s=55, color="#111111" if ok else "0.65",
               zorder=5, edgecolors="w", linewidths=0.4)
    ax.annotate(lab, (a, b), textcoords="offset points", xytext=(4, 3),
                fontsize=5.5, color="#111111" if ok else "0.6")
ax.axhline(0, color="0.8", lw=0.5)
ax.axvline(0, color="0.8", lw=0.5)
ax.plot([-LIM, LIM], [-LIM, LIM], color="0.9", lw=0.5, ls=":", zorder=1)
ax.set_xlim(-0.6, 2.0)
ax.set_ylim(-1.1, 1.6)
ax.set_xlabel("$a$ (AGN: BHRadEff)", fontsize=6.5)
ax.set_ylabel("$b$ (wind: VarWindVel)", fontsize=6.5)
ax.tick_params(labelsize=6)
panel_label(ax, "(h) IMFslope by statistic")

# family legend in panel (a)
h = [plt.Line2D([], [], marker="o", ls="", color=FAMCOL[f], ms=3.5, label=f)
     for f in ("agn", "wind", "other")]
h.append(plt.Line2D([], [], marker="*", ls="", color="#111111", ms=7, label="IMFslope"))
h.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                    color="0.4", ms=3.5, label=f"$R^2<{R2_OPEN:g}$"))
axes[0].legend(handles=h, loc="lower left", fontsize=4.6, handletextpad=0.1,
               borderpad=0.2, labelspacing=0.25)

fig.tight_layout()
save(fig, "figs_v2/fig21d_mechanism_blend")
plt.close(fig)

# ── robustness: alternate templates for IMFslope ─────────────────────────────
print("\nIMFslope blend under alternate template choices:")
for alt_agn, alt_wind in [("QuasarThreshold", T_WIND), (T_AGN, "WindEnergyIn1e51erg"),
                          ("QuasarThreshold", "WindEnergyIn1e51erg")]:
    line = []
    for stat, _ in STATS:
        xT, dA, _ = delta(alt_agn, stat)
        _, dW, _ = delta(alt_wind, stat)
        _, dP, _ = delta("IMFslope", stat)
        A = np.column_stack([shapenorm(dA), shapenorm(dW)])
        w, *_ = np.linalg.lstsq(A, shapenorm(dP), rcond=None)
        line.append(f"{stat}:({w[0]:+.2f},{w[1]:+.2f})")
    print(f"  [{short_label(alt_agn)} + {short_label(alt_wind)}]  " + "  ".join(line))
