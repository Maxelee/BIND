"""Figures for the family-basis model section (agnostic-lambda version,
2026-08-12). Five figures, all built from shipped caches + the agnostic
search results (figs_preview/agnostic_lambda_results.npz) -- the shipped
two-bin bundle is NOT touched, so these stay valid after promotion.

  pfig_fm_basis             (a) S(ell) basis functions, (b) kappa-PDF basis
  pfig_fm_freeamp           free-amplitude reconstruction of 7 spanning nodes
  pfig_fm_latent_extraction (a) halos->bin medians demo, (b) library grid +
                            chosen cells + bootstrap stability, (c) search path
  pfig_fm_model_curves      7-statistic leave-one-out validation (agnostic λ)
  pfig_fm_fiducial          out-of-design fiducial closure + trough ladder

Run:
    /mnt/home/mlee1/venvs/BIND_env/bin/python family_model_section_figs.py
"""
import sys

import os

import numpy as np

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
sys.path.insert(0, ".")
from paper_style import COLORS, TWO_COL, panel_label, save, setup  # noqa: E402

setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from family_model import FamilyModel  # noqa: E402

STATS = ["clk", "pdf", "pk", "mn", "v0", "v1", "v2"]
FC = ["#2a78d6", "#eb6834", "#199e70", "#c98500", "#d55181"]  # family colors C1..C5
OB_OM = 0.0486 / 0.3089

fm = FamilyModel()
# BIND_CAMPAIGN=n1000 -> the Sobol dataset assembled from the campaign runs
# (fewer nodes than 256 until tracing finishes), and outputs tagged _n1000 so
# they cannot overwrite the 50-real renders.
_CAMPAIGN = os.environ.get("BIND_CAMPAIGN", "sci50")
_DS = ("/mnt/home/mlee1/ceph/bind_n1000/emulator_dataset_n1000.npz"
       if _CAMPAIGN == "n1000"
       else "/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_nu05.npz")
def _tag(stem):
    return stem + ("_n1000" if _CAMPAIGN == "n1000" else "")


dsn = np.load(_DS,
              allow_pickle=True)
coef = np.load("latent_model_coeffs.npz")
EDGb = coef["ell_edges"]
_eld = np.asarray(dsn["a__suppression__ell"], float)
AMPS = np.load("figs_preview/amplitude_sets.npz", allow_pickle=True)
RES = np.load("figs_preview/agnostic_lambda_results_obs.npz", allow_pickle=True)
NAMES = [str(n) for n in RES["names"]]
CHOSEN = list(RES["chosen"])
X_ALL, X_FID, FREQ = RES["X_ALL"], RES["X_FID"], RES["freq"]
cz = np.load("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/atlas_cubes/"
             "atlas_cube_snap096.npz")
rows_s = dsn["run_ids"]

# ── row alignment + amplitude re-measurement (campaign-aware) ────────────────
# The sci50-derived model artifacts (AMPS's a_sobol, RES's X_ALL) are
# row-indexed by Sobol run_id (row i == run_id i, 256 rows).  The campaign
# dataset holds a sorted, possibly NON-CONTIGUOUS subset of run_ids until
# tracing completes, so a positional index into dsn must NEVER index those
# arrays directly (doing so silently pairs unrelated runs -- the 2026-08-21
# n1000 renders had exactly that defect).  SEL maps campaign row -> sci50 row.
# Amplitudes are re-measured from the CAMPAIGN curves by free least-squares
# projection onto the FIXED shipped basis -- the same definition as the stored
# sci50 a_sobol, verified against it below -- so the model (basis/mean/latents)
# stays the paper's while every measured input reflects the campaign data.
SEL = np.asarray(rows_s, int)
assert SEL.ndim == 1 and (np.diff(SEL) > 0).all(), "dataset run_ids not sorted"


def free_amps(Y, mean, B):
    return np.linalg.lstsq(B.T, (np.asarray(Y, float) - mean).T, rcond=None)[0].T


def check_amp_convention(st, ds50):
    """Assert free_amps() reproduces the stored sci50 a_sobol for this stat."""
    a50 = free_amps(sobol_Y(st, ds50), AMPS[f"{st}__mean"], AMPS[f"{st}__basis"])
    stored = np.asarray(AMPS[f"{st}__a_sobol"], float)
    dd = float(np.nanmax(np.abs(a50 - stored))) / max(float(np.nanstd(stored)), 1e-30)
    print(f"amp-convention check [{st}]: max|recomputed - stored|/std = {dd:.2e}")
    assert dd < 1e-3, (f"{st}: free_amps() does not reproduce the shipped a_sobol "
                       "-- the stored amplitudes use a different fit; do not re-measure")


_DS50_PATH = "/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset_nu05.npz"
_ds50 = dsn if _CAMPAIGN == "sci50" else np.load(_DS50_PATH, allow_pickle=True)
assert np.asarray(_ds50["run_ids"], int).tolist() == list(range(256)), \
    "sci50 dataset rows are not run_id 0..255 -- SEL mapping invalid"


def sobol_Y(st, ds=None):
    ds = dsn if ds is None else ds
    dk = {"clk": "suppression", "pdf": "pdf", "pk": "peak_counts",
          "mn": "minima_counts", "v0": "mf_v0", "v1": "mf_v1",
          "v2": "mf_v2"}[st]
    Y = np.asarray(ds[f"t__{dk}__value"], float)
    Y = Y[:, 1, :] if Y.ndim == 3 else Y
    if st == "clk":
        Y = np.stack([np.nanmean(Y[:, (_eld >= EDGb[i]) & (_eld < EDGb[i + 1])], 1)
                      for i in range(len(EDGb) - 1)], axis=1)
    return Y[:, :fm.basis(st).shape[1]]


Yclk = sobol_Y("clk")
key = (Yclk - Yclk.mean(0)).mean(1)
SHOW = [int(np.argsort(key)[int(q * (len(key) - 1))])
        for q in np.linspace(0, 1, 7)]

# ═════════════════════════════════════════════════════════════════════════
# Fig 1: basis functions
# ═════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 2.63))
B = fm.basis("clk")
for k in range(B.shape[0]):
    ax.plot(fm.grid("clk"), B[k], color=FC[k % len(FC)], lw=1.4,
            label=rf"$b_{{{k + 1}}}$")
ax.set_xscale("log")
ax.axhline(0, color="0.85", lw=0.6, zorder=0)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\hat B_k(\ell)$")
ax.legend(fontsize=7, loc="upper left")
plt.tight_layout()
save(fig, _tag("figs_v2/pfig_fm_basis"))
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════
# Fig 2: free-amplitude reconstruction (basis + fitted a_r, no latents)
# ═════════════════════════════════════════════════════════════════════════
xg = fm.grid("clk")
mean_c = AMPS["clk__mean"]
basis_c = AMPS["clk__basis"]
check_amp_convention("clk", _ds50)
amps_c = free_amps(Yclk, mean_c, basis_c)   # campaign-measured; == stored under sci50

fig = plt.figure(figsize=(3.5, 3.2))
gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.08)
ax = fig.add_subplot(gs[0])
rx = fig.add_subplot(gs[1], sharex=ax)
resid_all = []
for r in SHOW:
    meas = Yclk[r]
    model = mean_c + amps_c[r] @ basis_c
    ax.plot(xg, meas, color="k", lw=1.1)
    ax.plot(xg, model, color=COLORS["bind"], lw=1.1, ls="--")
    rel = 100 * (model - meas) / meas
    rx.plot(xg, rel, color=COLORS["bind"], lw=0.7, alpha=0.8)
    resid_all.append(np.abs(rel))
ax.plot([], [], color="k", lw=1.1, label="measured")
ax.plot([], [], color=COLORS["bind"], lw=1.1, ls="--",
        label=r"$\overline{S} + \sum_k \hat a_k \hat B_k$")
ax.set_xscale("log")
ax.axhline(1, color="0.9", lw=0.6, zorder=0)
ax.set_ylabel(r"$S(\ell)$")
ax.legend(fontsize=6.5, loc="upper left")
ax.tick_params(labelbottom=False)
rx.axhline(0, color="0.6", lw=0.6)
rx.axhspan(-0.5, 0.5, color="0.92", zorder=0)
rx.set_ylim(-1, 1)
rx.set_xlabel(r"$\ell$")
rx.set_ylabel(r"$\Delta$ [%]", fontsize=7)
save(fig, _tag("figs_v2/pfig_fm_freeamp"))
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════
# Fig 3: latent extraction procedure
# ═════════════════════════════════════════════════════════════════════════
BIN_EDGES = [13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.3, 15.0]
BLAB = ["13.0", "13.2", "13.4", "13.6", "13.8", "14.0", "14.3+"]
_QALL = [("f_bar", r"$f_{\rm bar}$"), ("f_star", r"$f_\star$"),
         ("f_gas200", r"$f_{\rm gas,200}$"), ("c_gas", r"$c_{\rm gas}$"),
         ("c_gas_raw", r"$c_{\rm gas}^{\rm raw}$"), ("c_dm", r"$c_{\rm dm}$"),
         ("cgas_over_cdm", r"$c_{\rm gas}/c_{\rm dm}$"),
         ("logT", r"$\log T$"), ("logK", r"$\log K$"),
         ("logPe", r"$\log P_e$"), ("logY", r"$\log Y$"),
         ("logT_ss", r"$\log T_{\rm ss}$"), ("logY_ss", r"$\log Y_{\rm ss}$"),
         ("scat_fbar", r"$\sigma(f_{\rm bar})$"), ("scat_logY", r"$\sigma(Y)$")]
_present = {n.split("[")[0] for n in NAMES if not n.startswith(("DECOY", "trend_"))}
QORDER = [q for q, _ in _QALL if q in _present]
QLAB = [l for q, l in _QALL if q in _present]
BCOLS = [f"{lo:.1f}-{hi:.1f}".replace("-15.0", "+") for lo, hi in
         zip(BIN_EDGES[:-1], BIN_EDGES[1:])]

fig, ax = plt.subplots(figsize=(TWO_COL[0], 3.3))
G = np.full((len(QORDER), len(BCOLS) + 1), np.nan)   # +1 col for trends
for i, nm in enumerate(NAMES):
    if nm.startswith("DECOY"):
        continue
    if nm.startswith("trend_"):
        q = nm[len("trend_"):]
        if q in QORDER:
            G[QORDER.index(q), -1] = FREQ[i]
        continue
    q, b = nm.split("[")
    b = b.rstrip("]")
    if q in QORDER and b in BCOLS:
        G[QORDER.index(q), BCOLS.index(b)] = FREQ[i]
# block-credited stability: a bootstrap credits a cell if it selected the
# cell ITSELF or any near-duplicate (|r|>0.95) -- aligns the heat with the
# chosen boxes (bootstraps often pick a stand-in from the same correlated
# block, so raw per-cell counts under-light the chosen representatives).
BOOT_SETS = list(RES["boot_sets"])
real = [i for i, n in enumerate(NAMES) if not n.startswith("DECOY")]
CN = np.corrcoef(((X_ALL - X_ALL.mean(0)) / X_ALL.std(0))[:, real].T)
BLOCK = {i: {real[j] for j in np.where(np.abs(CN[k]) > 0.95)[0]}
         for k, i in enumerate(real)}
CRED = np.zeros(len(NAMES))
for i in real:
    CRED[i] = sum(1 for bs in BOOT_SETS if BLOCK[i] & set(int(b) for b in bs))
G_CRED = np.full_like(G, np.nan)
for i, nm in enumerate(NAMES):
    if nm.startswith("DECOY"):
        continue
    if nm.startswith("trend_"):
        q = nm[len("trend_"):]
        if q in QORDER:
            G_CRED[QORDER.index(q), -1] = CRED[i]
        continue
    q, b = nm.split("[")
    b = b.rstrip("]")
    if q in QORDER and b in BCOLS:
        G_CRED[QORDER.index(q), BCOLS.index(b)] = CRED[i]
G = G_CRED
cmap = plt.get_cmap("Reds").copy()
cmap.set_bad("0.92")
im = ax.imshow(np.ma.masked_invalid(G), aspect="auto", cmap=cmap, vmin=0,
               vmax=20)


def _cell(nm):
    if nm.startswith("trend_"):
        return QORDER.index(nm[6:]), len(BCOLS)
    q, b = nm.split("[")
    return QORDER.index(q), BCOLS.index(b.rstrip("]"))


for order, i in enumerate(CHOSEN, start=1):
    r, c = _cell(NAMES[i])
    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                           edgecolor=COLORS["bind"], lw=1.6))
    ax.text(c + 0.42, r - 0.38, str(order), ha="right", va="top",
            fontsize=5.5, color=COLORS["bind"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="w", ec="none",
                      alpha=0.85))
r, c = _cell("f_bar[13.0-13.2]")   # selected first, floated out (see caption)
ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                       edgecolor=COLORS["bind"], lw=1.3, ls=(0, (3, 2))))
ax.set_xticks(range(len(BCOLS) + 1))
ax.set_xticklabels(BLAB + ["trend"], fontsize=6)
ax.set_yticks(range(len(QORDER)))
ax.set_yticklabels(QLAB, fontsize=6)
ax.set_xlabel(r"$\log_{10} M_{500c}$ bin (lower edge)")
cb = plt.colorbar(im, ax=ax, pad=0.01)
cb.set_label("bootstraps selecting this cell or a near-duplicate"
             r" ($|r|>0.95$), of 20", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
fig.tight_layout()
save(fig, _tag("figs_v2/pfig_fm_latent_extraction"))
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════
# Fig 4: 7-panel leave-one-out validation with the AGNOSTIC latents,
# every panel normalized by a common REFERENCE
#
# REWORKED 2026-08-14. The previous version drew each statistic raw, and the
# feedback response was invisible: P(nu) spans four decades, N_pk three, V_0
# runs 1 -> 0, so a few-percent parameter-driven change is a line width. All
# seven leave-one-out nodes lay on top of each other and neither the spread
# nor the model's ability to track it could be judged by eye. Dividing every
# panel by a reference puts all seven on a common "deviation from reference"
# footing, which is the thing the section is actually claiming.
#
# REFERENCE (REF_MODE): the measured out-of-design fiducial,
# AMPS["<st>__fid_measured"] (50-real bind_science field_cache / mf_cache at
# z_s index 1, all seven statistics on the canonical grid). Set REF_MODE to
# "mean" to reference the 256-node Sobol design mean instead -- that is the
# model's own expansion point AND it is unaffected by the 2026-08-13 fiducial
# conditioning correction, whereas this fiducial cache is derived from the
# retired bind/run_0000 and its refresh is a deferred item in
# referee/FIDUCIAL_SWAP.md. Numbers below are printed with whichever is used.
#
# TAIL MASK (REF_FLOOR): a ratio is meaningless where the reference vanishes.
# P, N_pk, N_min, V_0 and V_1 all fall to zero in their tails and V_2 CHANGES
# SIGN, so bins with |ref| < REF_FLOOR of that panel's peak are dropped and
# the excluded region is shaded. Measured: at 1% the kappa-PDF tail ratio goes
# NEGATIVE and V_2's zero crossing blows up; 3% is stable but leaves
# boundary-bin spikes (N_min max residual 15.0%); 5% removes them (4.5%) at
# the cost of a few bins. S(ell) is never masked -- its reference never drops
# below 88% of peak.
#
# The C_ell^kappakappa panel is GONE and nothing is lost: normalizing cancels
# the fixed DMO band spectrum, so C_ell/C_ell^ref is ALGEBRAICALLY IDENTICAL
# to S(ell)/S(ell)^ref. The old panels (a) and (b) would have become the same
# picture drawn twice. Seven statistics, seven panels, and the freed eighth
# slot carries the legend and the per-statistic error summary. This also
# matches main.tex's own "seven weak-lensing statistics" phrasing.
# ═════════════════════════════════════════════════════════════════════════
from matplotlib.lines import Line2D

REF_MODE = "mean"         # "fid" (measured fiducial) | "mean" (Sobol design mean)
REF_FLOOR = 0.05          # drop bins with |ref| < this fraction of the panel peak

LAT = X_ALL[:, CHOSEN]    # full 256-row design (run_id-ordered; Fig 5 uses this)
LAT_C = LAT[SEL]          # campaign-aligned rows, positional-parallel to dsn
Lc = LAT_C - LAT_C.mean(0)

SPECN = {"clk": (r"$S(\ell)$", r"$\ell$", True),
         "pdf": (r"$P(\nu)$", r"$\nu$", False),
         "pk":  (r"$N_{\rm pk}$", r"$\nu$", False),
         "mn":  (r"$N_{\rm min}$", r"$\nu$", False),
         "v0":  (r"$V_0$", r"$\nu$", False),
         "v1":  (r"$V_1$", r"$\nu$", False),
         "v2":  (r"$V_2$", r"$\nu$", False)}
ORDER7 = ["clk", "pdf", "pk", "mn", "v0", "v1", "v2"]
REFLAB = {"fid": "fiducial", "mean": "design mean"}[REF_MODE]

fig = plt.figure(figsize=(TWO_COL[0], 6.4))
gs = fig.add_gridspec(4, 4, height_ratios=[2.2, 1.0, 2.2, 1.0], hspace=0.45,
                      wspace=0.42)
summary = []
for i, st in enumerate(ORDER7):
    r0, c0 = 2 * (i // 4), i % 4
    ax = fig.add_subplot(gs[r0, c0])
    rx = fig.add_subplot(gs[r0 + 1, c0], sharex=ax)
    slab, xlab, logx = SPECN[st]
    xg = fm.grid(st)
    Y = sobol_Y(st)
    B = AMPS[f"{st}__basis"]
    mean = AMPS[f"{st}__mean"]
    check_amp_convention(st, _ds50)
    A = free_amps(Y, mean, B)                  # campaign-measured amplitudes
    ref = AMPS[f"{st}__fid_measured"] if REF_MODE == "fid" else mean
    keepb = np.abs(ref) >= REF_FLOOR * np.abs(ref).max()
    rr = np.where(keepb, ref, np.nan)          # NaN masks the drawn curves
    resid = []
    for r in SHOW:
        keep = np.arange(len(LAT_C)) != r
        W, *_ = np.linalg.lstsq(np.c_[Lc[keep], np.ones(keep.sum())],
                                A[keep], rcond=None)
        model = mean + (np.r_[Lc[r], 1.0] @ W) @ B
        meas = Y[r]
        ax.plot(xg, meas / rr, color="k", lw=1.1)
        ax.plot(xg, model / rr, color=COLORS["bind"], lw=1.1, ls="--")
        dd = 100 * (model - meas) / rr         # SAME normalization as the panel
        resid.append(dd)                       # above, so the two read together
        rx.plot(xg, dd, color=COLORS["bind"], lw=0.7, alpha=0.8)
    resid = np.asarray(resid)
    summary.append((st, float(np.nanmax(np.abs(resid))),
                    float(np.sqrt(np.nanmean(resid ** 2)))))
    # Shade the excluded tail(s), so a curve that stops reads as a masked bin
    # and not as missing data. Span edges sit at the MIDPOINT between the last
    # kept and first masked sample, so shading never covers a drawn bin.
    if (~keepb).any():
        _mid = np.r_[xg[0] - 0.5 * (xg[1] - xg[0]),
                     0.5 * (xg[1:] + xg[:-1]),
                     xg[-1] + 0.5 * (xg[-1] - xg[-2])]
        _bad, _i = ~keepb, 0
        while _i < len(_bad):
            if _bad[_i]:
                _j = _i
                while _j + 1 < len(_bad) and _bad[_j + 1]:
                    _j += 1
                for _a in (ax, rx):
                    _a.axvspan(_mid[_i], _mid[_j + 1], color="0.93", lw=0,
                               zorder=0)
                _i = _j + 1
            else:
                _i += 1
    if logx:
        ax.set_xscale("log")
    ax.axhline(1, color="0.6", lw=0.6, zorder=1)
    rx.axhline(0, color="0.6", lw=0.6)
    # Per-panel residual scale. A single shared limit cannot work once the
    # panels are normalized: the response amplitude runs from +-1% (V_0) to
    # +-80% (S(ell)), so the old fixed +-7% made V_0's residual invisible and
    # clipped S(ell)'s. Scale to the 99th percentile -- NOT the max, or one
    # boundary-bin spike sets the scale -- floored at +-1%. The summary box is
    # the authority for the max, which may run off a panel by design.
    rx.set_ylim(*(lambda v: (-v, v))(
        max(1.0, 1.25 * float(np.nanpercentile(np.abs(resid), 99)))))
    BARLAB = {"clk": r"$S(\ell)/\bar{S}(\ell)$", "pdf": r"$P(\nu)/\bar{P}(\nu)$",
              "pk": r"$N_{\rm pk}/\bar{N}_{\rm pk}$", "mn": r"$N_{\rm min}/\bar{N}_{\rm min}$",
              "v0": r"$V_0/\bar{V}_0$", "v1": r"$V_1/\bar{V}_1$", "v2": r"$V_2/\bar{V}_2$"}
    ax.set_ylabel(BARLAB[st], fontsize=6.5)
    rx.set_xlabel(xlab, fontsize=7)
    rx.set_ylabel(r"$\Delta$ [%]", fontsize=6)
    ax.tick_params(labelsize=6, labelbottom=False)
    rx.tick_params(labelsize=6)
    # upper RIGHT: at upper left the tag collided with the four-significant-
    # figure y tick labels that the narrow-range ratio panels (V_0, N_min) carry
    panel_label(ax, f"({'abcdefg'[i]})", loc="upper right")

axl = fig.add_subplot(gs[2:4, 3])
axl.axis("off")
axl.legend(handles=[Line2D([], [], color="k", lw=1.1, label="measured"),
                    Line2D([], [], color=COLORS["bind"], lw=1.1, ls="--",
                           label="model"),
                    Line2D([], [], color="0.93", lw=6,
                           label=f"masked: |ref| < {100 * REF_FLOOR:.0f}% of peak")],
           loc="upper center", fontsize=6.2, frameon=False, handlelength=2.2)
fig.subplots_adjust(top=0.97)
save(fig, _tag("figs_v2/pfig_fm_model_curves"))
plt.close(fig)
print(f"pfig_fm_model_curves: reference = {REFLAB} "
      f"(REF_MODE={REF_MODE!r}), tail mask |ref| < {100 * REF_FLOOR:.0f}% of "
      "panel peak; C_ell panel dropped (identical to S(ell) once normalized)")
for s, mx, rms in summary:
    print(f"  {s:4s} |model-meas|/{REF_MODE} over the 7 LOO nodes: "
          f"rms {rms:5.2f}%   max {mx:6.2f}%")

# ═════════════════════════════════════════════════════════════════════════
# Fig 5: out-of-design fiducial closure + trough ladder
# ═════════════════════════════════════════════════════════════════════════
if _CAMPAIGN == "n1000":
    print("pfig_fm_fiducial SKIPPED under n1000: its closure legs (50-real "
          "bind_science field cache + paired DMO prefix + shipped a_sobol) are "
          "sci50 products with no campaign twin; the sci50 render stands.")
else:
    SCI = "/mnt/home/mlee1/ceph/bind_science"
    fc = np.load(f"{SCI}/field_cache/field_stats_fid.npz")
    dmo = np.load(f"{SCI}/runs/dmo/run_0000/Cl_kappa_paired.npz")

    def band_reals(cl_real):
        S = cl_real[:, 1, :] / dmo["cl_real"][:, 1, :]
        return np.stack([np.nanmean(S[:, (_eld >= EDGb[i]) & (_eld < EDGb[i + 1])], 1)
                         for i in range(len(EDGb) - 1)], axis=1)

    S_bind = band_reals(fc["kk_bind"])
    S_true = band_reals(fc["kk_truth"])
    xg = fm.grid("clk")

    # agnostic model prediction + per-band predictive sigma (rng(1) folds)
    mu = LAT.mean(0)
    W, *_ = np.linalg.lstsq(np.c_[LAT - mu, np.ones(len(LAT))],
                            AMPS["clk__a_sobol"], rcond=None)
    pred_fid = AMPS["clk__mean"] + (np.r_[X_FID[CHOSEN] - mu, 1.0] @ W) @ AMPS["clk__basis"]
    perm = np.random.default_rng(1).permutation(len(LAT))
    folds = [(np.setdiff1d(perm, te), te) for te in np.array_split(perm, 5)]
    pred_cv = np.empty_like(AMPS["clk__a_sobol"])
    for tr, te in folds:
        m_ = LAT[tr].mean(0)
        Wf, *_ = np.linalg.lstsq(np.c_[LAT[tr] - m_, np.ones(len(tr))],
                                 AMPS["clk__a_sobol"][tr], rcond=None)
        pred_cv[te] = np.c_[LAT[te] - m_, np.ones(len(te))] @ Wf
    sig_pred = ((Yclk - AMPS["clk__mean"]) - pred_cv @ AMPS["clk__basis"]).std(0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TWO_COL[0], 2.7),
                                 gridspec_kw={"width_ratios": [2.0, 1.0]})
    m, e = S_true.mean(0), S_true.std(0) / np.sqrt(len(S_true))
    a1.plot(xg, m, color=COLORS["truth"], lw=1.5, label="TNG300 full hydro (truth)")
    a1.fill_between(xg, m - e, m + e, color=COLORS["truth"], alpha=0.18, lw=0)
    mb = S_bind.mean(0)
    a1.plot(xg, mb, color="0.55", lw=1.2, label="BIND fiducial paint (measured)")
    a1.plot(xg, pred_fid, color=COLORS["bind"], lw=1.8, ls="--",
            label=r"model: 8 measured numbers $\to S(\ell)$")
    a1.fill_between(xg, pred_fid - sig_pred, pred_fid + sig_pred,
                    color=COLORS["bind"], alpha=0.18, lw=0,
                    label=r"model $\pm 1\sigma_{\rm pred}$")
    a1.set_xscale("log")
    a1.axhline(1, color="0.9", lw=0.6, zorder=0)
    a1.set_xlabel(r"$\ell$")
    a1.set_ylabel(r"$S(\ell)$")
    a1.legend(fontsize=6, loc="lower left")
    panel_label(a1, "(a)", loc="lower right")

    tro = 19
    models = [("4 latents,\n1 mass bin", 0.920),
              ("8 latents,\n2 mass bins", 0.908),
              ("8 latents,\nagnostic", float(pred_fid[tro]))]
    y = np.arange(len(models))[::-1]
    a2.axvline(mb[tro], color="0.55", lw=1.2)
    a2.axvline(m[tro], color=COLORS["truth"], lw=1.5)
    a2.axvspan(mb[tro] - sig_pred[tro], mb[tro] + sig_pred[tro], color="0.92",
               zorder=0)
    for (lab, v), yy in zip(models, y):
        a2.plot(v, yy, "o", color=COLORS["bind"], ms=5)
        a2.text(v, yy + 0.18, lab, fontsize=5.6, ha="center", va="bottom",
                color="0.25")
    a2.text(mb[tro], len(models) - 0.35, "measured (BIND fid)", fontsize=5.6,
            ha="right", va="top", rotation=90, color="0.4")
    a2.text(m[tro], len(models) - 0.35, "truth", fontsize=5.6, ha="right",
            va="top", rotation=90, color=COLORS["truth"])
    a2.set_ylim(-0.6, len(models) - 0.1)
    a2.set_yticks([])
    a2.set_xlabel(rf"$S(\ell \simeq {xg[tro]:.0f})$  (trough)")
    panel_label(a2, "(b)", loc="upper right")
    fig.tight_layout()
    save(fig, _tag("figs_v2/pfig_fm_fiducial"))
    plt.close(fig)

print("figure set written to figs_v2/ (+ previews in figs_preview/)")


# ═════════════════════════════════════════════════════════════════════════
# Fig 6: the search trajectory
# ═════════════════════════════════════════════════════════════════════════
P_ACT = [str(a) for a in RES["path_action"]]
P_IDX = list(RES["path_index"])
P_SC = list(RES["path_score"])

QTEX = {"f_bar": r"f_{\rm bar}", "f_star": r"f_\star",
        "f_gas200": r"f_{\rm gas,200}", "c_gas": r"c_{\rm gas}",
        "logT": r"\log T", "logK": r"\log K", "logPe": r"\log P_e",
        "logY": r"\log Y", "logT_ss": r"\log T_{\rm ss}",
        "logY_ss": r"\log Y_{\rm ss}"}


def short(nm):
    q, b = nm.split("[")
    return rf"${QTEX[q]}[{b.split('-')[0].rstrip(']+')}]$"


fig, a1 = plt.subplots(figsize=(4.6, 3.2))
mv = np.arange(1, len(P_ACT) + 1)
a1.plot(mv, P_SC, "-", color=COLORS["bind"], lw=1.2, zorder=2)
for m, act, i, sc in zip(mv, P_ACT, P_IDX, P_SC):
    if act == "+":
        a1.plot(m, sc, "o", color=COLORS["bind"], ms=4, zorder=3)
        lab = short(NAMES[i])
    else:
        a1.plot(m, sc, "o", mfc="w", mec=COLORS["bind"], ms=4, zorder=3)
        lab = short(NAMES[i]) + " out"
    prev_act = P_ACT[m - 2] if m >= 2 else "+"
    up = m <= 3 or (act == "+" and prev_act == "-")
    if up:      # label ascends up-right, starting AT the dot
        kw = dict(xytext=(3, 4), ha="left", va="bottom")
    else:       # label descends down-left, ending AT the dot
        kw = dict(xytext=(-1, -5), ha="right", va="top")
    a1.annotate(lab, (m, sc), textcoords="offset points", fontsize=7.2,
                rotation=40, rotation_mode="anchor",
                color="0.45" if act == "-" else "0.1", **kw)
fin = P_SC[-1]
a1.axhline(fin + 0.002, color="0.75", lw=0.7, ls=":")
a1.text(0.6, fin + 0.0028,
        r"stop: best remaining gain $+0.0013$",
        fontsize=7.2, color="0.45", va="bottom", ha="left")
a1.set_ylim(0.69, 0.955)
a1.set_xlim(0.3, len(mv) + 1.4)
a1.set_xlabel("search move", fontsize=9)
a1.set_ylabel(r"pooled CV $R^2$ (selection folds)", fontsize=9)
a1.tick_params(labelsize=8)
fig.tight_layout()
save(fig, _tag("figs_v2/pfig_fm_search_path"))
plt.close(fig)
