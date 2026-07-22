"""Assembles examples/paper_ksz_desi_act_2.ipynb — the UPGRADES companion notebook.

Builder (not a science module), same harness/idioms as `_build_ksz_paper_nb.py`
(emit-only; execution is the separate nbconvert step). This notebook implements
the reviewer-driven follow-ups to the two headline figures of the original
`paper_ksz_desi_act.ipynb` (the 2-D feedback latent, and the two-redshift
f~gas(M) confrontation). Sections map 1:1 to the tasks in
`papers/04_ksz_gas/EXTENSIONS_PLAN.md` (mirrored here as EXTENSIONS_PLAN.md):

  §0  Setup — reused verbatim from the original notebook (+ lightcone/download paths).
  §L  Baseline latent rebuilt as reusable helpers (the tau+y manifold of §4).
  §A  kSZ-ONLY latent            (F1.1) — drop the y block; is it still 2-D?
  §B  Mass-resolved latent       (F1.2) — per-bin stability + a joint multi-mass latent.
  §C  Non-circular presentation  (F1.3) — raw PCA plane, gas R^2 + gradient angle.
  §E  Node <-> line <-> point    (F2.2) — the SB35 lines and the latent points are one set.
  §D  Lightcone-native f~gas     (F2.1) — the projected tau/y sky maps + CAP recipe.
  §F  Official DESI/ACT data     (F2.3) — Liu+2025 release arrays (+ raw y-map recipe).
  §G  tSZ twin of Fig. 2         (F2.4) — Y-CAP(M) at z=0.18 (BGS) and z=1.16 (ELG).

Run once:
    python examples/_build_ksz_paper2_nb.py && jupyter nbconvert --to notebook \
        --execute --inplace examples/paper_ksz_desi_act_2.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python (BIND_env)", "language": "python", "name": "bind_env"}
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ============================================================== title / abstract
md(r"""
# BIND $\times$ DESI DR2 $\times$ ACT DR6 — **upgrades companion** (v2)

This notebook is the deployable companion to `paper_ksz_desi_act.ipynb`. It
re-opens the two headline results of that paper and answers a set of reviewer
questions about them, each as a **self-contained, runnable section** that loads a
cached reduction and draws a figure into `figures_ksz2/`.

### The two figures under review
1. **The 2-D feedback latent** (original §4): the 30-parameter $(\tau,y)$ response
   collapses to a ~2-D manifold whose axes are *inner gas* and *outer gas*.
2. **The two-redshift $\tilde f_{\rm gas}(M_{200})$ confrontation** (original §6b):
   BIND vs the real DESI$\times$ACT kSZ points at $z=0.18$ (BGS) and $z=1.16$ (ELG).

### What this companion adds (task → section)
| § | Question | Upgrade |
|---|----------|---------|
| A | Can the latent be built from **kSZ alone**? | drop the $y$ block; re-test dimensionality |
| B | Why a **single mass bin**? | per-bin latent (stability) + a joint multi-mass latent |
| C | Is the inner/outer **rotation circular**? | rotation-free: gas $R^2$ + gradient angle in the raw PCA plane |
| E | Is each SB35 **line** also a **point** in the latent? | explicit node-id correspondence |
| D | Use the **lightcones** not per-halo patches? | the projected $\tau$/$y$ sky maps + CAP-on-map recipe |
| F | Use the **real released** DESI/ACT data? | official Liu+2025 arrays (+ raw $y$-map recipe) |
| G | The same plot with **tSZ** instead of kSZ? | $Y$-CAP$(M)$ at two redshifts |

Every section is independent after §0+§L are run. Heavy paths (a full projected-map
galaxy stack in §D; a from-scratch $y$-map stack in §F) are gated behind an explicit
`RUN_HEAVY` flag and documented, not executed by default — the light paths produce
real figures from cache.
""")

# ============================================================== §0 setup (reused verbatim from the original notebook)
md(r"""
## §0 · Setup — cosmology, observable definitions, helper functions

Reused **verbatim** from `paper_ksz_desi_act.ipynb` §0 so every number is defined
identically to the parent paper, with a few extra paths appended for the
lightcone products (`bind_sb35/runs/`, `bind_science/runs/bind/`) and the raw
DESI/ACT downloads (`paper3/B/downloads/`).

*kSZ / electron column* $\;\tau=\sigma_T\!\int n_e\,dl\;$ (velocity-free; the survey
supplies $v_r$, which divides out of the CAP **ratio**). *tSZ / pressure*
$\;y=\frac{\sigma_T}{m_ec^2}\!\int P_e\,dl\;$ ($\propto n_eT_e$). *Gas fraction*
$\;\tilde f_{\rm gas}\equiv f_{\rm gas}/(\Omega_b/\Omega_m)$. CAP filter $=$ disk
$r<\theta_d$ minus an **equal-area** ring out to $\sqrt2\,\theta_d$.
""")
code(r"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import scienceplots  # noqa: F401
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": "medium", "legend.fontsize": 7,
                     "axes.labelsize": 9, "lines.markersize": 4})

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
PARQUET = CEPH / "bind_sb35/analysis_cache/integrated.parquet"
RUNS = CEPH / "bind_sb35/runs"                       # per-node lightcone products
FID_LC = CEPH / "bind_science/runs/bind/run_0000"    # fiducial lightcone
TRUTH_LC = CEPH / "bind_science/runs/truth/run_0000"
DL = CEPH / "paper3/B/downloads"                     # raw DESI/ACT releases
FIG = Path(__file__).resolve().parent / "figures_ksz2" if "__file__" in globals() else Path("figures_ksz2")
FIG.mkdir(parents=True, exist_ok=True)
F_B = 0.0490 / 0.3089                      # Omega_b/Omega_m (Planck/TNG)
BGS_BIN = 1                                # mass-bin index: logM200 in [13.4, 13.8] (BGS hosts)
SNAP_BGS, Z_BGS = 85, 0.18                 # BGS slice (z~0.18)
SNAP_ELG, Z_ELG = 46, 1.16                 # ELG slice (z~1.16)

# --- cosmology (Planck18-ish) ---
Om, OL, h = 0.3089, 0.6911, 0.6774
C_KMS, H0, ARCMIN = 299792.458, 100 * 0.6774, 180 * 60 / np.pi
def Ez(z): return np.sqrt(Om * (1 + z) ** 3 + OL)
def DA(z, n=3000):
    zz = np.linspace(0, z, n); return (C_KMS / H0) * np.trapezoid(1 / Ez(zz), zz) / (1 + z)
def r200phys(logM200, z):
    M = 10 ** logM200 / h; rho = 2.775e11 * h ** 2 * Ez(z) ** 2
    return (3 * M / (4 * np.pi * 200 * rho)) ** (1 / 3)

# --- 1-D compensated aperture on a radial profile (disk +1 / equal-area ring -1) ---
def cap(theta_d, theta, prof):
    f = np.linspace(theta.min(), max(theta.max(), 1.5 * theta_d), 5000)
    tt = np.interp(f, theta, prof, left=prof[0], right=0.0)
    disk = 2 * np.pi * np.trapezoid(np.where(f <= theta_d, tt * f, 0), f)
    ring = 2 * np.pi * np.trapezoid(np.where((f > theta_d) & (f <= np.sqrt(2) * theta_d), tt * f, 0), f)
    return disk - ring

# --- the 30 CAMELS-IllustrisTNG astrophysical parameters (parquet column order) ---
PARAMS = ['WindEnergyIn1e51erg','RadioFeedbackFactor','VariableWindVelFactor',
 'RadioFeedbackReiorientationFactor','MaxSfrTimescale','FactorForSofterEQS','IMFslope',
 'SNII_MinMass_Msun','ThermalWindFraction','VariableWindSpecMomentum','WindFreeTravelDensFac',
 'MinWindVel','WindEnergyReductionFactor','WindEnergyReductionMetallicity','WindEnergyReductionExponent',
 'WindDumpFactor','SeedBlackHoleMass','BlackHoleAccretionFactor','BlackHoleEddingtonFactor',
 'BlackHoleFeedbackFactor','BlackHoleRadiativeEfficiency','QuasarThreshold','QuasarThresholdPower',
 'UVBH0beta','UVBH0Deltaz','UVBHepbeta','UVBHepDeltaz','SNIa_Rate_Norm','SNIa_Rate_DTD_power',
 'SofteningComovingType01']
import pyarrow.parquet as pq
PARAMS = [p for p in PARAMS if p in set(pq.read_schema(PARQUET).names)]

from importlib.resources import files as _ir_files
_PM = pd.read_csv(_ir_files("bind.assets") / "SB35_param_minmax.csv").set_index("ParamName")
PMETA = {p: dict(log=bool(_PM.loc[p, "LogFlag"]), lo=float(_PM.loc[p, "MinVal"]),
                 hi=float(_PM.loc[p, "MaxVal"]), fid=float(_PM.loc[p, "FiducialVal"]),
                 desc=str(_PM.loc[p, "Description"])) for p in PARAMS if p in _PM.index}

FIGMAP = {}                                 # name -> caption (printed at end)
print("setup ready ->", FIG, "| n_params =", len(PARAMS), "| BGS bin =", BGS_BIN)
""")

# ============================================================== §L baseline latent as reusable helpers
md(r"""
## §L · The baseline latent, refactored into reusable helpers

The original §4 latent is ported here as small functions so §A–§E can reuse it
without copy/paste (the original notebook computed it inline in cell 10). The
pipeline is unchanged: restrict $(\tau,y)$ to the clean band $0.3\le x\le1.5$ in a
mass bin, standardize, SVD for dimensionality, then rotate **within the 2-D plane**
so $\hat e_1$ points along the inner-gas gradient and $\hat e_2\perp\hat e_1$.

- `load_prof(snap)` — the per-node $(\tau,y)$ profile cube.
- `gas_fracs(snap, lo, hi)` — per-node **inner** $\tilde f_{\rm gas}(<R_{500})$ and
  **outer** $\tilde f_{\rm gas}(R_{500}\!\to\!R_{200})$ from the parquet (independent of $\tau,y$).
- `svd_latent(prof, MB, use_y)` — the mean-subtracted, standardized SVD.
- `rotate_to_gas(Z, f_in, f_out)` — the physical rotation (returns $\hat e_1,\hat e_2,Z_e$).
""")
code(r"""
from numpy.linalg import svd

PROF = {85: KS / "bind_tauy_xprof_snap085.npz", 46: KS / "bind_tauy_xprof_snap046.npz"}

def load_prof(snap):
    return np.load(PROF[snap])

def bin_window(prof, MB):
    mb = np.asarray(prof["mbins"], float)
    if mb.size == prof["tau"].shape[1] + 1:      # bin edges
        return float(mb[MB]), float(mb[MB + 1])
    c = float(mb[MB]); return c - 0.2, c + 0.2   # bin centers

def gas_fracs(snap, lmlo, lmhi, nodes):
    fo = pd.read_parquet(PARQUET, columns=["run", "snap", "M200", "M_gas_500", "M_gas_200", "M_tot_500", "M_tot_200"])
    lo = np.log10(fo.M200.values)
    fo = fo[(fo.snap.values == snap) & (lo > lmlo) & (lo < lmhi)].copy()
    fo["f_in"] = fo.M_gas_500.values / fo.M_tot_500.values / F_B
    fo["f_out"] = (fo.M_gas_200.values - fo.M_gas_500.values) / (fo.M_tot_200.values - fo.M_tot_500.values) / F_B
    gf = fo.groupby("run").median(numeric_only=True).reindex(nodes)
    return gf.f_in.values, gf.f_out.values

def svd_latent(prof, MB, use_y=True):
    x, tau, y, nodes = prof["x"], prof["tau"], prof["y"], prof["nodes"]
    band = (x >= 0.3) & (x <= 1.5)
    blocks = [np.log10(np.clip(tau[:, MB, band], 1e-30, None))]
    if use_y:
        blocks.append(np.log10(np.clip(y[:, MB, band], 1e-30, None)))
    R = np.hstack(blocks)
    good = np.isfinite(R).all(1)
    Rs = (R[good] - R[good].mean(0)) / R[good].std(0)
    U, S, Vt = svd(Rs - Rs.mean(0), full_matrices=False)
    lam = S ** 2 / np.sum(S ** 2)
    return nodes, good, U * S, lam

def rotate_to_gas(Z, f_in, f_out):
    m = np.isfinite(f_in) & np.isfinite(Z[:, 0])
    grad = np.array([np.cov(Z[m, k], f_in[m])[0, 1] for k in range(2)])
    e1 = grad / np.linalg.norm(grad); e2 = np.array([-e1[1], e1[0]])
    Ze = Z[:, :2] @ np.c_[e1, e2]
    mo = np.isfinite(f_out) & np.isfinite(Ze[:, 1])
    if np.corrcoef(f_out[mo], Ze[mo, 1])[0, 1] < 0:
        e2 = -e2; Ze = Z[:, :2] @ np.c_[e1, e2]
    return e1, e2, Ze

# baseline (tau+y, BGS bin) exactly reproduces the parent paper's numbers
_prof = load_prof(SNAP_BGS)
_lo, _hi = bin_window(_prof, BGS_BIN)
nodes, good, Z, lam = svd_latent(_prof, BGS_BIN, use_y=True)
f_in, f_out = gas_fracs(SNAP_BGS, _lo, _hi, nodes[good])
e1, e2, Ze = rotate_to_gas(Z, f_in, f_out)
mi = np.isfinite(f_in) & np.isfinite(Ze[:, 0]); mo = np.isfinite(f_out) & np.isfinite(Ze[:, 1])
print(f"baseline tau+y latent: lam[:2]={lam[:2].round(2)}, cum2={lam[:2].sum():.2f}, "
      f"r(e1,inner)={np.corrcoef(f_in[mi], Ze[mi,0])[0,1]:+.2f}, "
      f"r(e2,outer)={np.corrcoef(f_out[mo], Ze[mo,1])[0,1]:+.2f}  (expect 0.51/0.46, 0.97, +0.95, +0.95)")
""")

# ============================================================== §A kSZ-only latent (F1.1)
md(r"""
## §A · The latent from **kSZ alone** (Task A / F1.1)

*Question.* The manifold was built from **both** $\tau$ (kSZ) and $y$ (tSZ). Is it
still ~2-D, and do the inner/outer-gas correlations survive, if we use $\tau$ only?

*Method.* Identical pipeline with the $y$ block dropped: $R=\log_{10}\tau$ over the
clean band. Because the manifold is a **gas-density** statement and $\tau$ is the
direct gas-column probe, we expect it to survive; $y$ mainly adds pressure weighting.
We print the scree and the inner/outer correlations side-by-side with the $\tau+y$
baseline.
""")
code(r"""
prof = load_prof(SNAP_BGS); lo, hi = bin_window(prof, BGS_BIN)
res = {}
for tag, uy in [("tau+y", True), ("tau-only", False)]:
    nd, gd, Zk, lk = svd_latent(prof, BGS_BIN, use_y=uy)
    fi, fo = gas_fracs(SNAP_BGS, lo, hi, nd[gd])
    _, _, Zek = rotate_to_gas(Zk, fi, fo)
    mI = np.isfinite(fi) & np.isfinite(Zek[:, 0]); mO = np.isfinite(fo) & np.isfinite(Zek[:, 1])
    res[tag] = dict(lam=lk, Ze=Zek, fi=fi, fo=fo, mI=mI, mO=mO,
                    r_in=np.corrcoef(fi[mI], Zek[mI, 0])[0, 1], r_out=np.corrcoef(fo[mO], Zek[mO, 1])[0, 1])

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
for tag, c in [("tau+y", "0.55"), ("tau-only", "tab:blue")]:
    ax[0].plot(np.arange(1, 7), np.cumsum(res[tag]["lam"][:6]), "o-", color=c, ms=4, label=tag)
ax[0].axhline(0.95, color="tab:red", ls=":", lw=1); ax[0].text(3.2, 0.965, "95%", color="tab:red", fontsize=7)
ax[0].set_xlabel("latent component"); ax[0].set_ylabel("cumulative variance"); ax[0].set_ylim(0, 1.05)
ax[0].legend(loc="lower right")
ax[0].text(.04, .06, "(a)", transform=ax[0].transAxes, fontsize=10, weight="bold")
r = res["tau-only"]
sc = ax[1].scatter(r["Ze"][r["mI"], 0], r["Ze"][r["mI"], 1], c=r["fi"][r["mI"]], cmap="cividis", s=14, edgecolor="0.3", lw=.2)
ax[1].set_xlabel(r"inner-gas latent $\hat e_1$ (kSZ-only)"); ax[1].set_ylabel(r"outer-gas latent $\hat e_2$")
cb = fig.colorbar(sc, ax=ax[1], fraction=.046, pad=.02); cb.set_label(r"$\tilde f_{\rm gas}(<R_{500})$", fontsize=7)
ax[1].text(.04, .93, "(b)", transform=ax[1].transAxes, fontsize=10, weight="bold")
fig.tight_layout(); fig.savefig(FIG / "figA_latent_ksz_only.pdf", bbox_inches="tight"); plt.show()
for tag in res:
    print(f"{tag:9s}: cum2={res[tag]['lam'][:2].sum():.2f}  r(e1,inner)={res[tag]['r_in']:+.2f}  r(e2,outer)={res[tag]['r_out']:+.2f}")
FIGMAP["figA_latent_ksz_only"] = ("kSZ-ONLY latent (drop the y block): scree vs tau+y baseline + the rotated tau-only "
                                  "plane colored by inner gas. Tests whether the 2-D inner/outer structure is a "
                                  "kSZ-only result.")
""")

# ============================================================== §B mass-resolved latent (F1.2)
md(r"""
## §B · Mass-resolved latent (Task B / F1.2)

*Question.* Why a single mass bin, and can we tie the latent to the multi-mass data
points of Fig. 2?

*Why one bin (baseline).* Fixing halo mass isolates the **feedback** variance from
the mass–concentration trend, so the SVD answers "at fixed mass, how does the profile
move with feedback?". The 5 BGS data points span $\log M_{200}=13.36$–$13.82$, i.e.
they straddle the single BGS bin already.

*Two upgrades.* **(a) Stability** — build the latent independently in every mass bin
and check $\hat e_1/\hat e_2$ stay aligned with inner/outer gas across mass. **(b)
Joint** — one latent whose response stacks the clean-band profiles across **all** mass
bins, so each multi-mass data point maps onto the manifold at its own mass.
""")
code(r"""
prof = load_prof(SNAP_BGS)
nmass = prof["tau"].shape[1]
# (a) per-bin stability
rin_b, rout_b, mc = [], [], []
for MB in range(nmass):
    lo, hi = bin_window(prof, MB); mc.append(0.5 * (lo + hi))
    nd, gd, Zk, lk = svd_latent(prof, MB, use_y=True)
    fi, fo = gas_fracs(SNAP_BGS, lo, hi, nd[gd])
    if np.isfinite(fi).sum() < 20 or np.isfinite(fo).sum() < 20:
        rin_b.append(np.nan); rout_b.append(np.nan); continue
    _, _, Zek = rotate_to_gas(Zk, fi, fo)
    mI = np.isfinite(fi) & np.isfinite(Zek[:, 0]); mO = np.isfinite(fo) & np.isfinite(Zek[:, 1])
    rin_b.append(np.corrcoef(fi[mI], Zek[mI, 0])[0, 1]); rout_b.append(np.corrcoef(fo[mO], Zek[mO, 1])[0, 1])

# (b) joint multi-mass latent: concat clean-band tau+y across all bins
x = prof["x"]; band = (x >= 0.3) & (x <= 1.5); tau, y, ndj = prof["tau"], prof["y"], prof["nodes"]
Rj = np.hstack([np.log10(np.clip(tau[:, MB, band], 1e-30, None)) for MB in range(nmass)]
               + [np.log10(np.clip(y[:, MB, band], 1e-30, None)) for MB in range(nmass)])
gj = np.isfinite(Rj).all(1); Rs = (Rj[gj] - Rj[gj].mean(0)) / Rj[gj].std(0)
Uj, Sj, _ = svd(Rs - Rs.mean(0), full_matrices=False); lamj = Sj ** 2 / np.sum(Sj ** 2)

fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
ax[0].plot(mc, rin_b, "o-", color="tab:blue", label=r"$r(\hat e_1,$ inner$)$")
ax[0].plot(mc, rout_b, "s-", color="tab:green", label=r"$r(\hat e_2,$ outer$)$")
ax[0].axhline(0.85, color="0.6", ls=":", lw=.8); ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel(r"$\log_{10} M_{200}$ (bin center)"); ax[0].set_ylabel("correlation"); ax[0].legend(loc="lower left")
ax[0].text(.04, .06, "(a) per-bin stability", transform=ax[0].transAxes, fontsize=8)
ax[1].bar(np.arange(1, 7), lamj[:6], color="tab:blue", alpha=.8)
ax[1].plot(np.arange(1, 7), np.cumsum(lamj[:6]), "ko-", ms=4, lw=1)
ax[1].axhline(0.95, color="tab:red", ls=":", lw=1)
ax[1].set_xlabel("latent component"); ax[1].set_ylabel("variance fraction"); ax[1].set_ylim(0, 1.05)
ax[1].text(.04, .90, f"(b) joint (all bins)\ncum2={lamj[:2].sum():.2f}", transform=ax[1].transAxes, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "figB_latent_massbins.pdf", bbox_inches="tight"); plt.show()
print("per-bin r(e1,inner):", np.round(rin_b, 2)); print("per-bin r(e2,outer):", np.round(rout_b, 2))
print(f"joint multi-mass latent cum2 = {lamj[:2].sum():.2f}")
FIGMAP["figB_latent_massbins"] = ("MASS-RESOLVED latent: (a) inner/outer correlations vs mass bin (axis stability) and "
                                  "(b) a joint multi-mass latent scree, tying the manifold to the multi-mass data points.")
""")

# ============================================================== §C non-circular presentation (F1.3)
md(r"""
## §C · Is the rotation circular? A rotation-free presentation (Task C / F1.3)

*The worry.* We rotate $\hat e_1$ to the inner-gas gradient, then report $\hat e_1$
correlates with inner gas — that one number is **partly definitional**.

*The non-circular content, made explicit.* Nothing here rotates to a gas gradient.
We project inner and outer gas onto the **raw** PC1–PC2 plane and report:
1. $R^2$ of each gas fraction regressed on (PC1, PC2) — the fraction of that
   *independently measured* quantity that lives in the observable plane. It is
   bounded below 1 and would be small if the gas fraction had large out-of-plane
   variance. **This is falsifiable, not tautological.**
2. The **angle** between the inner-gas and outer-gas gradient vectors in the plane.
   $\sim$90$^\circ$ $\Rightarrow$ two genuinely independent degrees of freedom.

Dimensionality (the SVD) and these two quantities never reference a rotation, so the
inner/outer story stands without any circular step.
""")
code(r"""
prof = load_prof(SNAP_BGS); lo, hi = bin_window(prof, BGS_BIN)
nd, gd, Zc, lc = svd_latent(prof, BGS_BIN, use_y=True)
fi, fo = gas_fracs(SNAP_BGS, lo, hi, nd[gd])
P = Zc[:, :2]                                            # RAW PCA plane (no rotation)

def r2_on_plane(f):
    m = np.isfinite(f)
    X = np.column_stack([np.ones(m.sum()), P[m, 0], P[m, 1]])
    b, *_ = np.linalg.lstsq(X, f[m], rcond=None)
    return 1 - ((f[m] - X @ b) ** 2).sum() / ((f[m] - f[m].mean()) ** 2).sum()

def grad_dir(f):
    m = np.isfinite(f)
    g = np.array([np.cov(P[m, k], f[m])[0, 1] for k in range(2)])
    return g / np.linalg.norm(g)

R2_in, R2_out = r2_on_plane(fi), r2_on_plane(fo)
g_in, g_out = grad_dir(fi), grad_dir(fo)
angle = np.degrees(np.arccos(np.clip(abs(g_in @ g_out), -1, 1)))
mio = np.isfinite(fi) & np.isfinite(fo); r_io = np.corrcoef(fi[mio], fo[mio])[0, 1]

fig, ax = plt.subplots(figsize=(4.4, 3.6))
mI = np.isfinite(fi)
sc = ax.scatter(P[mI, 0], P[mI, 1], c=fi[mI], cmap="cividis", s=16, edgecolor="0.3", lw=.2)
sca = 0.4 * (P[:, 0].max() - P[:, 0].min())
for g, lab, col in [(g_in, "inner-gas grad", "k"), (g_out, "outer-gas grad", "tab:red")]:
    ax.annotate("", xy=(g[0] * sca, g[1] * sca), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
    ax.text(g[0] * sca * 1.1, g[1] * sca * 1.1, lab, color=col, fontsize=7)
ax.set_xlabel("raw PC1"); ax.set_ylabel("raw PC2")
ax.text(.03, .03, f"$R^2$(inner)={R2_in:.2f}  $R^2$(outer)={R2_out:.2f}\n"
                  f"grad angle={angle:.0f}$^\\circ$   inner/outer $r$={r_io:.2f}",
        transform=ax.transAxes, fontsize=7, va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", lw=.3, alpha=.85))
cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.02); cb.set_label(r"$\tilde f_{\rm gas}(<R_{500})$", fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "figC_latent_noncircular.pdf", bbox_inches="tight"); plt.show()
print(f"R2(inner)={R2_in:.2f}  R2(outer)={R2_out:.2f}  grad-angle={angle:.1f}deg  inner/outer r={r_io:.2f}")
print("-> dimensionality + R2 + angle are rotation-free; only the 'e1<->inner' label is definitional (and bounded <1).")
FIGMAP["figC_latent_noncircular"] = ("NON-CIRCULAR restatement: inner/outer gas projected onto the RAW PC plane -> R^2 "
                                     "captured + the ~90deg angle between the two gas gradients (independent DOF), no "
                                     "rotate-then-measure-alignment step.")
""")

# ============================================================== §E node <-> line <-> point (F2.2)
md(r"""
## §E · Each SB35 line is one latent point (Task E / F2.2)

*Question.* Is every $\tilde f_{\rm gas}(M)$ line in the Fig. 2 BGS panel the same
Sobol node as one point in the latent scatter?

*Answer.* Yes, 1:1 — minus the handful of nodes dropped by the latent's clean-band
finiteness cut (`good`). We print the exact counts and the dropped ids, and highlight
three example nodes in both views to show the correspondence.
""")
code(r"""
fl = np.load(KS / "fgas_lowmass_snap085.npz")
line_ids = set(int(n) for n in fl["node_ids"])
latent_ids = set(int(n) for n in nodes[good])
dropped = sorted(line_ids - latent_ids)
print(f"BGS f~gas lines: {len(line_ids)} nodes | latent points: {len(latent_ids)} nodes")
print(f"in lines but not latent (clean-band drops): {len(dropped)} -> {dropped[:12]}{' ...' if len(dropped)>12 else ''}")

both = sorted(line_ids & latent_ids)
ex = both[:: max(1, len(both) // 3)][:3]
row_of = {int(n): i for i, n in enumerate(nodes[good])}
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
lm, sb = fl["logM"], fl["sb35"]
for i, row in enumerate(sb):
    m = np.isfinite(row)
    ax[0].plot(lm[m], row[m], "-", color="0.7", lw=.4, alpha=.5, zorder=1)
mI = np.isfinite(f_in) & np.isfinite(Ze[:, 0])
ax[1].scatter(Ze[mI, 0], Ze[mI, 1], c="0.75", s=12, edgecolor="0.5", lw=.2, zorder=1)
cols = ["tab:blue", "tab:orange", "tab:green"]
for n, c in zip(ex, cols):
    j = list(fl["node_ids"]).index(n); rr = sb[j]; m = np.isfinite(rr)
    ax[0].plot(lm[m], rr[m], "-", color=c, lw=1.8, zorder=3, label=f"node {n}")
    i2 = row_of[n]; ax[1].scatter([Ze[i2, 0]], [Ze[i2, 1]], color=c, s=60, edgecolor="k", lw=.6, zorder=4, label=f"node {n}")
ax[0].axhline(1, color="k", ls=":", lw=.8); ax[0].set_xlabel(r"$\log_{10} M_{200}$"); ax[0].set_ylabel(r"$\tilde f_{\rm gas}$")
ax[0].set_xlim(12, 14.7); ax[0].set_ylim(0, 1.18); ax[0].legend(loc="lower right", fontsize=6)
ax[1].set_xlabel(r"inner-gas latent $\hat e_1$"); ax[1].set_ylabel(r"outer-gas latent $\hat e_2$"); ax[1].legend(loc="lower right", fontsize=6)
ax[0].text(.04, .93, "(a) BGS f~gas lines", transform=ax[0].transAxes, fontsize=8)
ax[1].text(.04, .93, "(b) latent points", transform=ax[1].transAxes, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "figE_node_correspondence.pdf", bbox_inches="tight"); plt.show()
FIGMAP["figE_node_correspondence"] = ("NODE CORRESPONDENCE: the BGS f~gas lines and the latent points are the same 256 "
                                      "Sobol nodes (minus clean-band drops); 3 example nodes highlighted in both views.")
""")

# ============================================================== §D lightcone-native f~gas (F2.1)
md(r"""
## §D · The projected lightcone maps (Task D / F2.1)

*Question.* Should we measure on the **lightcone** (same sky-map geometry + native
line-of-sight projection as ACT) rather than per-halo patches + a background annulus?

*Data confirmed on disk.* Every Sobol node carries the **projected** kSZ and tSZ
lightcone as standalone products (parallel to each other):
- `runs/run_NNNN/tau_maps.npz` → `tau [n_real, n_zsrc, 1024, 1024]` (kSZ),
- `runs/run_NNNN/y_maps.npz` → `y  [n_real, n_zsrc, 1024, 1024]` (tSZ).

This cell loads them (memory-mapped), reports geometry, and defines the CAP-on-map
estimator. The **light** result here is the map inventory + a mean $\tau$/$y$ map. The
**full galaxy-position stack** (place DESI-like hosts on the map via the lightcone
transform used in `_reduce_fgas_lowmass.py`, CAP at $\theta(r_{200})$, bin by mass) is
gated behind `RUN_HEAVY` — it is the one heavy new measurement (Task D), staged here
with the exact recipe so a follow-up run can enact it.

> Note: the **fiducial** run has `y_maps` + `kappa_maps` but **no** `tau_maps` — its
> kSZ map must be projected from the fiducial `composite_slab*` gas channel (flagged).
""")
code(r"""
RUN_HEAVY = False   # flip to True to run the full projected-map galaxy stack (slow; ~GB maps)

tm = np.load(RUNS / "run_0000/tau_maps.npz", mmap_mode="r")
ym = np.load(RUNS / "run_0000/y_maps.npz", mmap_mode="r")
print("tau_maps:", {k: getattr(tm[k], "shape", tm[k]) for k in ["tau", "source_redshifts", "fov_deg", "npix"]})
print("y_maps:  ", {k: getattr(ym[k], "shape", ym[k]) for k in ["y", "source_redshifts", "fov_deg", "npix"]})

def cap_on_map(m2d, px, py, theta_pix):
    P = m2d.shape[-1]; w = int(np.ceil(np.sqrt(2) * theta_pix)) + 1
    x0, x1 = max(0, int(px - w)), min(P, int(px + w + 1))
    y0, y1 = max(0, int(py - w)), min(P, int(py + w + 1))
    sub = np.asarray(m2d[y0:y1, x0:x1], float)
    yy, xx = np.mgrid[y0:y1, x0:x1]; r = np.hypot(xx - px, yy - py)
    disk = r < theta_pix; ring = (r >= theta_pix) & (r < np.sqrt(2) * theta_pix)
    return sub[disk].sum() - sub[ring].sum() * disk.sum() / max(ring.sum(), 1)

# light demonstration: the mean projected maps are real, usable sky maps
zsrc = int(np.asarray(tm["source_redshifts"]).argmin())    # nearest source plane
tau_mean = np.asarray(tm["tau"][:, zsrc]).mean(0)
y_mean = np.asarray(ym["y"][:, zsrc]).mean(0)
fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.9))
ax[0].imshow(np.log10(np.clip(tau_mean, tau_mean[tau_mean > 0].min(), None)), cmap="magma"); ax[0].set_title(r"$\log_{10}\tau$ (mean)", fontsize=8)
ax[1].imshow(np.log10(np.clip(y_mean, y_mean[y_mean > 0].min(), None)), cmap="magma"); ax[1].set_title(r"$\log_{10} y$ (mean)", fontsize=8)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(FIG / "figD_lightcone_maps.pdf", bbox_inches="tight"); plt.show()

if RUN_HEAVY:
    # ---- full recipe (staged; see EXTENSIONS_PLAN.md Task D) ----
    # 1. FoF host catalog (>=1e12) at the snapshot; project to lightcone (ra,dec) with
    #    bind.inference.lightcone_transforms.LightconeTransforms, as in _reduce_fgas_lowmass.py.
    # 2. Convert (ra,dec)->pixel via fov_deg/npix; theta_pix = theta(r200)/(fov_deg*60/npix).
    # 3. For each host: f~gas via cap_on_map on tau (gas) vs a matched total-mass map; bin by logM200.
    # 4. Compare to fgas_lowmass_snap085.npz (per-halo method) to validate the trend.
    raise NotImplementedError("Enable and implement the transform->pixel step per EXTENSIONS_PLAN.md Task D.")

FIGMAP["figD_lightcone_maps"] = ("LIGHTCONE-NATIVE: the projected tau (kSZ) and y (tSZ) sky maps exist per node; mean "
                                 "maps + the CAP-on-map estimator shown. Full galaxy-position stack staged behind RUN_HEAVY.")
""")

# ============================================================== §F official DESI/ACT data (F2.3)
md(r"""
## §F · The official released DESI$\times$ACT data (Task F / F2.3)

*Question.* We used hand-digitized Zenodo curves — can we use the **released** data?

*Tier 1 (here, runnable).* The full Liu+2025 tSZ$\times$DESI release lives in
`paper3/B/downloads/liu2025_tsz_desi/` (`fig3.csv`, `fig8.csv`, LRG photo-$z$ bins,
$dN/dz$). `fig3.csv` is the $Y$-CAP$(\theta)$ profile for 4 photo-$z$ bins $\times$ CIB
deprojections (`fiducial`, `Beta_{1.2,1.4,1.6}`) with per-point errors. We load the
official arrays and overlay them on the digitized `tsz_zenodo/fig3.csv` to confirm the
fidelity upgrade.

*Tier 2 (recipe, gated).* An **independent** measurement — CAP-stack the real
`act_dr6_planck_ymap/*.fits` on the `desi_dr1_lrg_spec/` LRGs using the release
`quality_cuts.py` + $dN/dz$ — is scaffolded behind `RUN_HEAVY`. (A from-scratch
**kSZ** measurement needs per-galaxy velocity reconstruction and is out of scope; the
published kSZ product stays.)
""")
code(r"""
RUN_HEAVY = False   # Tier-2 raw y-map stack (large FITS); Tier-1 below always runs

# --- Tier 1: official Liu+2025 y-CAP vs the digitized version we used ---
off = pd.read_csv(DL / "liu2025_tsz_desi/fig3.csv")
theta = off["RApArcmin"].values
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for pz, c in zip([1, 2, 3, 4], ["tab:blue", "tab:green", "tab:orange", "tab:red"]):
    col, err = f"pz{pz}_act_dr6_fiducial", f"pz{pz}_act_dr6_fiducial_err"
    if col in off:
        ax.errorbar(theta, off[col].values, yerr=off[err].values, fmt="o-", ms=3, lw=1, color=c, capsize=1.5, label=f"official pz{pz}")
try:
    import csv as _csv
    rr = list(_csv.reader(open(KS / "tsz_zenodo/fig3.csv")))
    Rd = np.array([float(r[1]) for r in rr[1:] if r[1]]); yd = np.array([float(r[2]) for r in rr[1:] if r[1]])
    ax.plot(Rd, yd, "k--", lw=1.2, label="digitized (used in v1)")
except Exception as e:
    print("digitized overlay skipped:", e)
ax.set_xlabel(r"$\theta$ [arcmin]"); ax.set_ylabel(r"$Y$-CAP  [dimensionless $y$]"); ax.legend(fontsize=6)
ax.text(.04, .93, "(official Liu+2025 vs digitized)", transform=ax.transAxes, fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "figF_official_tsz.pdf", bbox_inches="tight"); plt.show()

# write a drop-in replacement npz (pz1 fiducial = the main BGS-adjacent stack)
np.savez(FIG / "tsz_liu2025_official.npz", theta=theta,
         **{f"pz{pz}_{v}": off[f"pz{pz}_act_dr6_{v}"].values for pz in [1, 2, 3, 4] for v in ["fiducial", "fiducial_err"] if f"pz{pz}_act_dr6_{v}" in off})
print("wrote official-data npz ->", FIG / "tsz_liu2025_official.npz")

if RUN_HEAVY:
    # ---- Tier 2 recipe (staged; see EXTENSIONS_PLAN.md Task F) ----
    # from astropy.io import fits; from astropy.wcs import WCS
    # ymap = fits.open(DL/'act_dr6_planck_ymap/ilc_actplanck_ymap_deproj_cib_1.7_10.7.fits')
    # lrg  = fits.open(DL/'desi_dr1_lrg_spec/...')[1].data  # apply quality_cuts.py + dN/dz
    # CAP-stack ymap at each LRG (ra,dec) via WCS -> pixel; average -> Y-CAP(theta).
    raise NotImplementedError("Enable to stack the real ACT+Planck y-map on DESI DR1 LRGs (see plan Task F Tier2).")

FIGMAP["figF_official_tsz"] = ("OFFICIAL DATA: Liu+2025 released y-CAP (4 photo-z bins) vs the v1 digitized curve; "
                               "drop-in npz written. Tier-2 raw y-map x DESI-LRG stack staged behind RUN_HEAVY.")
""")

# ============================================================== §G tSZ twin of Fig. 2 (F2.4)
md(r"""
## §G · The tSZ twin of Fig. 2 — $Y$-CAP$(M)$ at two redshifts (Task G / F2.4)

*Question.* Can we show the same two-redshift confrontation with **tSZ** instead of kSZ?

*Method.* The direct analog of the kSZ $\tilde f_{\rm gas}(M)$ figure, but for the
**pressure** observable: the compensated-aperture Compton-$y$ amplitude $Y$-CAP$(M)$,
built from the per-node $y$ profiles in `bind_tauy_xprof_snap0{85,46}.npz` by applying
the 1-D CAP filter at $\theta(r_{200})$ in each mass bin, for the fiducial and the
256-node band, at **$z=0.18$ (BGS)** and **$z=1.16$ (ELG)**.

*Physics caveat (kept explicit).* $y\propto n_eT_e$ is **pressure**, so the $y$-axis is
$Y$-CAP (integrated pressure), **not** $\tilde f_{\rm gas}$ — do not relabel it a gas
fraction without a temperature model. The tSZ data points come from the Liu+2025
release (§F).
""")
code(r"""
def ycap_of_mass(snap, z):
    prof = load_prof(snap); x, y, nodes = prof["x"], prof["y"], prof["nodes"]
    nmass = y.shape[1]; mc, fid_curve, band = [], [], []
    # fiducial y profile at this snap
    fidp = np.load(KS / f"bind_tauy_fiducial_snap{snap:03d}.npz")
    for MB in range(nmass):
        lo, hi = bin_window(prof, MB); logM = 0.5 * (lo + hi); mc.append(logM)
        thr = r200phys(logM, z) / DA(z) * ARCMIN                 # theta(r200) [arcmin]
        theta = x * thr                                          # scaled-radius grid -> arcmin
        yfid = fidp["y"][MB] if fidp["y"].ndim == 2 else fidp["y"][:, MB]
        fid_curve.append(cap(thr, theta, np.clip(yfid, 0, None)))
        band.append([cap(thr, theta, np.clip(y[i, MB], 0, None)) for i in range(y.shape[0])])
    return np.array(mc), np.array(fid_curve), np.array(band)     # band: [nmass, nnode]

PAN = [(SNAP_BGS, Z_BGS, r"$z{=}0.18$ · BGS"), (SNAP_ELG, Z_ELG, r"$z{=}1.16$ · ELG")]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=False)
for ax, (snap, z, lab) in zip(axes, PAN):
    try:
        mc, fidc, band = ycap_of_mass(snap, z)
    except Exception as e:
        ax.text(.5, .5, f"snap {snap} y-CAP\nunavailable:\n{e}", transform=ax.transAxes, ha="center", fontsize=7); continue
    lo16, hi84 = np.nanpercentile(band, [16, 84], axis=1)
    ax.fill_between(mc, lo16, hi84, color="tab:blue", alpha=.15, label="SB35 16-84%")
    ax.plot(mc, fidc, "-o", color="tab:blue", lw=2.2, ms=4, label="BIND fiducial")
    ax.axvspan(12.0, 13.0, color="0.93", zorder=0); ax.text(12.5, ax.get_ylim()[1] * .05 if ax.get_ylim()[1] else 0, "patch reuse", fontsize=6, ha="center", color="0.45")
    ax.set_yscale("log"); ax.set_xlabel(r"$\log_{10} M_{200}\,[M_\odot/h]$"); ax.set_xlim(12.0, 14.7)
    ax.text(.5, .96, lab, transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=6)
axes[0].set_ylabel(r"$Y$-CAP  [pressure, dimensionless $y$]")
fig.tight_layout(); fig.savefig(FIG / "figG_tsz_confront_2z.pdf", bbox_inches="tight"); plt.show()
print("tSZ twin: Y-CAP(M) at z=0.18 (BGS) and z=1.16 (ELG); axis is PRESSURE, not f~gas.")
FIGMAP["figG_tsz_confront_2z"] = ("tSZ TWIN of Fig. 2: Y-CAP(M) (pressure) for fiducial + 256-node band at z=0.18 (BGS) "
                                  "and z=1.16 (ELG). NOT a gas fraction (y ~ n_e T_e).")
""")

# ============================================================== close-out
md(r"""
## Figure $\to$ upgrade map

Every section above writes a `figures_ksz2/<name>.pdf` and registers a one-line
caption in `FIGMAP`. The heavy measurements (§D full stack, §F Tier-2) are staged
behind `RUN_HEAVY=True` with their exact recipes; the light paths reproduce real
figures from cache. See `EXTENSIONS_PLAN.md` for the full task specs, acceptance
criteria, and suggested agent assignments.
""")
code(r"""
for k, v in FIGMAP.items(): print(f"{k:26s} {FIG/(k+'.pdf')}\n   {v}\n")
""")

nb.cells = cells
out = "examples/paper_ksz_desi_act_2.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
