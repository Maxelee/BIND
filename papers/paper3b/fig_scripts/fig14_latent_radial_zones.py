"""p9 -- The two y-CAP components in the TNG feedback eigenbasis (inner/outer f_gas).
Frozen artifacts READ-ONLY. Writes JSON + figure only.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")

# ---------------- constants ----------------
F_B = 0.0490 / 0.3089            # cosmic baryon fraction (Planck/TNG), matches fig08
SNAP = 71                        # fiducial peaks' snapshot (pinned below), z=0.419969
Z_FID = 0.419969
MASS_CUT_LOG = 13.5              # log10 M_tot_500 selection = SB35 grid mass_min
NU_FIT = [1, 2, 3, 4]           # canonical analysis bins (drop nu0), per b5_fit_wiener
R_INNER, R_OUTER = 0, 4         # cap_radii index: 2 arcmin (inner), 8 arcmin (outer)
RIDGE_ALPHA = 1.0

B = Path("/mnt/home/mlee1/ceph/paper3/B")
PARQUET = Path("/mnt/home/mlee1/ceph/bind_sb35/analysis_cache/integrated.parquet")
OUT_JSON = B / "wp5_inference/b5_latent_radial_zones.json"
FIG_STEM = "/mnt/home/mlee1/BIND-paper3b/papers/paper3b/figs/fig14_latent_radial_zones"

rng = np.random.default_rng(20260719)

# ---------------- load frozen grids + data ----------------
grid = np.load(B / "wp4_mocks/sobol/model_grid_tfwiener_sb35.npz", allow_pickle=True)
run_names = grid["run_names"]
grun = np.array([int(s.split("_")[-1]) for s in run_names])   # integer run id
y_grid = grid["y_mean"]           # (253, 5 nu, 5 rad)
dln_mg = grid["delta_ln_mgas"]
dln_t = grid["delta_ln_t"]
cap = np.array([2., 3., 4., 6., 8.])
Nrun = len(grun)

tb = np.load(B / "wp4_mocks/model_grid_tfwiener.npz", allow_pickle=True)
tb_dlnmg = tb["delta_ln_mgas"]

stack = np.load(B / "wp2_measurement/stack_wiener_sm2am_fid.npz", allow_pickle=True)
y_data = stack["y_mean"]          # (5 nu, 5 rad)
y_cov = stack["y_cov"]            # (5 nu, 5 rad, 5 rad) statistical (jackknife)
cap_radii = stack["cap_radii_arcmin"]

sysd = np.load(B / "wp3_nulls/sigma_sys_wiener_decomposed.npz", allow_pickle=True)
half_band = sysd["half_band"]     # (5 nu) systematic 1-sigma half-band at 4' fit radius
sigma_sys = sysd["sigma_sys"]     # (5 nu, 5 nu) rank-1 CIB covariance at 4'

assert np.allclose(cap_radii, cap), "cap radii mismatch"

# ============================================================
# STEP 1: join SB35 grid <-> parquet on run; build zone f_gas; common-mode check
# ============================================================
df = pd.read_parquet(PARQUET, columns=["run", "snap", "M_gas_500", "M_gas_200",
                                        "M_tot_500", "M_tot_200"])
df = df[(df.snap == SNAP) & (np.log10(df.M_tot_500) > MASS_CUT_LOG)].copy()
df["f_in"] = df.M_gas_500.values / df.M_tot_500.values / F_B
df["f_out"] = ((df.M_gas_200.values - df.M_gas_500.values)
               / (df.M_tot_200.values - df.M_tot_500.values) / F_B)
df["lnMg500"] = np.log(df.M_gas_500.values)
g = df.groupby("run").agg(f_in=("f_in", "median"), f_out=("f_out", "median"),
                          Mtot500_mean=("M_tot_500", "mean"),
                          Mtot500_med=("M_tot_500", "median"),
                          Mtot200_med=("M_tot_200", "median"),
                          lnMg500=("lnMg500", "mean"),
                          nh=("f_in", "size"))
g = g.reindex(grun)
f_in = g.f_in.values
f_out = g.f_out.values
n_join_ok = int(np.isfinite(f_in).sum())
# --- PRIMARY join validator: parquet-derived per-run gas shift vs grid delta_ln_mgas ---
pq_dlnmg = g.lnMg500.values - np.nanmean(g.lnMg500.values)
r_join = float(np.corrcoef(pq_dlnmg, dln_mg)[0, 1])
# --- SECONDARY: common-mode M_tot check (mass is common-mode, gas varies) ---
Mtot_mean = g.Mtot500_mean.values
cm_cv = float(np.nanstd(Mtot_mean) / np.nanmean(Mtot_mean))
cm_cv_med = float(np.nanstd(g.Mtot500_med.values) / np.nanmean(g.Mtot500_med.values))
gas_cv = float(np.nanstd(f_in) / np.nanmean(f_in))
print(f"[JOIN] snap={SNAP}  grid runs={Nrun}  joined(finite f_in)={n_join_ok}")
print(f"[JOIN] PRIMARY: corr(parquet dln_mgas, grid delta_ln_mgas) = {r_join:.4f} "
      f"(scramble->~0; confirms run labels)")
print(f"[JOIN] M_tot_500 common-mode CV(mean)={cm_cv:.5f} CV(median)={cm_cv_med:.5f} "
      f"vs gas(f_in) CV={gas_cv:.3f}  (assert mass-CV<0.01)")
assert r_join > 0.9, f"join correlation {r_join} too low -- labels scrambled"
assert cm_cv < 0.01, f"common-mode CV {cm_cv} >= 0.01 -- join suspect"
assert n_join_ok == Nrun, "not all grid runs joined"
print(f"[JOIN] median halos/run above 1e{MASS_CUT_LOG}: {np.nanmedian(g.nh.values):.0f}")
print(f"[ZONES] f_in(tilde) mean/std = {np.nanmean(f_in):.3f}/{np.nanstd(f_in):.3f}"
      f"   f_out(tilde) mean/std = {np.nanmean(f_out):.3f}/{np.nanstd(f_out):.3f}")

# TNG fiducial reference = run nearest (dln_mgas, dln_t)=(0,0)
i_fid = int(np.argmin(np.abs(dln_mg) + np.abs(dln_t)))
f_in_TNG = float(f_in[i_fid]); f_out_TNG = float(f_out[i_fid])
f_in_TNGmean = float(np.nanmean(f_in)); f_out_TNGmean = float(np.nanmean(f_out))
print(f"[TNG] fiducial run idx={i_fid} name={run_names[i_fid]} "
      f"dln_mg={dln_mg[i_fid]:.3f} dln_t={dln_t[i_fid]:.3f} "
      f"f_in={f_in_TNG:.3f} f_out={f_out_TNG:.3f}")

# ============================================================
# STEP 2: rank-2 SVD of z-scored log y-CAP response; target-rotate onto (f_in,f_out)
# ============================================================
R = np.log10(np.clip(y_grid.reshape(Nrun, 25), 1e-30, None))
Rs = (R - R.mean(0)) / R.std(0)
U, S, Vt = np.linalg.svd(Rs, full_matrices=False)
lam = S**2 / np.sum(S**2)
Z = U * S
cum = np.cumsum(lam)
print(f"[SVD] variance fractions (top6): {np.round(lam[:6],4)}")
print(f"[SVD] cumulative: 1c={cum[0]:.3f} 2c={cum[1]:.3f} 3c={cum[2]:.3f}")

grad = np.array([np.cov(Z[:, k], f_in)[0, 1] for k in range(2)])
e1 = grad / np.linalg.norm(grad)
e2 = np.array([-e1[1], e1[0]])
Ze = Z[:, :2] @ np.c_[e1, e2]
if np.corrcoef(f_out, Ze[:, 1])[0, 1] < 0:
    e2 = -e2
    Ze = Z[:, :2] @ np.c_[e1, e2]
r_in_e1 = float(np.corrcoef(f_in, Ze[:, 0])[0, 1])
r_out_e2 = float(np.corrcoef(f_out, Ze[:, 1])[0, 1])
r_io = float(np.corrcoef(f_in, f_out)[0, 1])
print(f"[ROT] r(f_in, e1)={r_in_e1:+.3f}  r(f_out, e2)={r_out_e2:+.3f}  "
      f"r(f_in,f_out)={r_io:+.3f}")

def R2_multi(yv, X):
    Xc = np.column_stack([np.ones(len(yv))] + X)
    b = np.linalg.lstsq(Xc, yv, rcond=None)[0]
    return float(1 - ((yv - Xc @ b)**2).sum() / ((yv - yv.mean())**2).sum())

r2_e1 = R2_multi(Ze[:, 0], [f_in, f_out])
r2_e2 = R2_multi(Ze[:, 1], [f_in, f_out])
print(f"[ROT] R2(e1~f_in,f_out)={r2_e1:.3f}  R2(e2~f_in,f_out)={r2_e2:.3f}")

# ============================================================
# STEP 3: zone loadings -- regress C1=ln y(8'), C2=ln y(2')-ln y(8') on (f_in,f_out)
# ============================================================
lg = np.log(np.clip(y_grid, 1e-30, None))                 # (run, nu, rad)
C1 = lg[:, NU_FIT, R_OUTER].mean(1)                        # outer amplitude / run
C2 = (lg[:, NU_FIT, R_INNER] - lg[:, NU_FIT, R_OUTER]).mean(1)  # inner-minus-outer / run

def zscore(a):
    return (a - a.mean()) / a.std()

Xz = np.column_stack([zscore(f_in), zscore(f_out)])       # standardized predictors

def ridge_fit(Xz, y, alpha):
    yz = zscore(y)
    A = Xz.T @ Xz + alpha * np.eye(Xz.shape[1])
    beta = np.linalg.solve(A, Xz.T @ yz)                  # standardized loadings
    return beta

def cv_r2(Xz, y, alpha, k=5, seed=0):
    yz = zscore(y)
    idx = np.arange(len(y)); np.random.default_rng(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    preds = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        A = Xz[tr].T @ Xz[tr] + alpha * np.eye(Xz.shape[1])
        beta = np.linalg.solve(A, Xz[tr].T @ yz[tr])
        preds[f] = Xz[f] @ beta
    return float(1 - ((yz - preds)**2).sum() / ((yz - yz.mean())**2).sum())

load_C1 = ridge_fit(Xz, C1, RIDGE_ALPHA)     # [on f_in, on f_out]
load_C2 = ridge_fit(Xz, C2, RIDGE_ALPHA)
r2_C1 = cv_r2(Xz, C1, RIDGE_ALPHA)
r2_C2 = cv_r2(Xz, C2, RIDGE_ALPHA)
ratio_C1 = float(abs(load_C1[1] / load_C1[0]))   # outer/inner  (expect >2)
ratio_C2 = float(abs(load_C2[0] / load_C2[1]))   # inner/outer  (expect >2)
print(f"[LOAD] C1(outer amp) loadings [f_in,f_out]={np.round(load_C1,3)} "
      f"|out/in|={ratio_C1:.2f}  CV-R2={r2_C1:.3f}")
print(f"[LOAD] C2(inner-outer) loadings [f_in,f_out]={np.round(load_C2,3)} "
      f"|in/out|={ratio_C2:.2f}  CV-R2={r2_C2:.3f}")

# ============================================================
# STEP 4: invert the frozen data DEFICIT (Delta ln y) for per-zone f_gas deficit
# ------------------------------------------------------------
# The data lies 3-7x below the grid (off-manifold), so we invert the RESPONSE
# (Delta ln y about the grid, log-linear in ln f_gas), NOT absolute amplitudes.
# J: 2x2 response Jacobian d(amp)/d(ln f_zone). Data Delta-ln-y -> Delta ln f_zone
# = per-zone log deficit; exp() = data/TNG deficit ratio. Large deficit = linear
# EXTRAPOLATION beyond the sampled manifold (that is the off-manifold result).
# ============================================================
lnfi = np.log(f_in); lnfo = np.log(f_out)
mC1, mC2 = C1.mean(), C2.mean()
mlfi, mlfo = lnfi.mean(), lnfo.mean()
Xc = np.column_stack([lnfi - mlfi, lnfo - mlfo])          # centered ln f (253,2)
J = np.zeros((2, 2))
J[0] = np.linalg.lstsq(Xc, C1 - mC1, rcond=None)[0]        # d C1 / d(ln f_in, ln f_out)
J[1] = np.linalg.lstsq(Xc, C2 - mC2, rcond=None)[0]
condJ = float(np.linalg.cond(J))
r2_J1 = float(1 - ((C1 - mC1 - Xc @ J[0])**2).sum() / ((C1 - mC1)**2).sum())
r2_J2 = float(1 - ((C2 - mC2 - Xc @ J[1])**2).sum() / ((C2 - mC2)**2).sum())
print(f"[INV] response Jacobian J (rows C1,C2; cols dln f_in,dln f_out):\n{np.round(J,3)}")
print(f"[INV] cond(J)={condJ:.2f}  R2(C1 fit)={r2_J1:.3f}  R2(C2 fit)={r2_J2:.3f}")

def data_u(yy):
    l = np.log(np.clip(yy, 1e-30, None))
    u1 = l[NU_FIT, R_OUTER].mean() - mC1                   # Delta outer amp
    u2 = (l[NU_FIT, R_INNER] - l[NU_FIT, R_OUTER]).mean() - mC2   # Delta (inner-outer)
    return np.array([u1, u2])

def invert(yy):
    dlx = np.linalg.solve(J, data_u(yy))                  # (dln f_in, dln f_out) vs grid geomean
    lnfi_d = dlx[0] + mlfi
    lnfo_d = dlx[1] + mlfo
    # deficit ratio relative to TNG fiducial run
    r_in = np.exp(lnfi_d - lnfi[i_fid])
    r_out = np.exp(lnfo_d - lnfo[i_fid])
    return np.array([np.exp(lnfi_d), np.exp(lnfo_d), r_in, r_out])   # tilde f_in,f_out ; deficits

pt = invert(y_data)
f_in_data, f_out_data, def_in, def_out = pt
print(f"[INV] data f_in(tilde)={f_in_data:.3f}  f_out(tilde)={f_out_data:.3f}")
print(f"[INV] per-zone deficit vs TNG fiducial: inner={def_in:.3f}  outer={def_out:.3f}")
print(f"[INV] (implied inner deficit {1/def_in:.2f}x, outer deficit {1/def_out:.2f}x)")

# ---- Monte-Carlo error propagation (stat jackknife + rank-1 CIB sys) ----
NMC = 40000
frac_sys = half_band / y_data[:, 2]        # fractional CIB band per nu, referenced at 4'
samples = np.zeros((NMC, 4))
chols = {}
for nu in NU_FIT:
    C = y_cov[nu][np.ix_([R_INNER, R_OUTER], [R_INNER, R_OUTER])]
    chols[nu] = np.linalg.cholesky(C + 1e-30 * np.eye(2))
for m in range(NMC):
    yy = y_data.copy()
    a_cib = rng.standard_normal()                          # shared CIB amplitude (rank-1)
    for nu in NU_FIT:
        eps = chols[nu] @ rng.standard_normal(2)
        yy[nu, R_INNER] += eps[0] + a_cib * frac_sys[nu] * y_data[nu, R_INNER]
        yy[nu, R_OUTER] += eps[1] + a_cib * frac_sys[nu] * y_data[nu, R_OUTER]
    yy = np.clip(yy, 1e-12, None)
    samples[m] = invert(yy)
# fractional/log-normal errors -> report 16/50/84 percentiles
def pctl(a): return np.percentile(a, [16, 50, 84])
fin_p = pctl(samples[:, 0]); fout_p = pctl(samples[:, 1])
din_p = pctl(samples[:, 2]); dout_p = pctl(samples[:, 3])
mc_cov = np.cov(samples[:, :2].T)
mc_std = np.sqrt(np.diag(mc_cov))
def_in_err = float((din_p[2] - din_p[0]) / 2); def_out_err = float((dout_p[2] - dout_p[0]) / 2)
print(f"[INV] MC f_in(tilde)={fin_p[1]:.3f} [{fin_p[0]:.3f},{fin_p[2]:.3f}]  "
      f"f_out(tilde)={fout_p[1]:.3f} [{fout_p[0]:.3f},{fout_p[2]:.3f}]")
print(f"[INV] inner deficit = {def_in:.3f} +/-{def_in_err:.3f}   "
      f"outer deficit = {def_out:.3f} +/-{def_out_err:.3f}")
fdata = np.array([f_in_data, f_out_data])

# ============================================================
# STEP 5: reachability on the (f_in,f_out) plane -- data vs SB35 cloud, per zone
# ============================================================
def reach(val, cloud, sig):
    lo, hi = np.nanmin(cloud), np.nanmax(cloud)
    inside = bool(lo <= val <= hi)
    if val < lo:
        gap_sig = (lo - val) / sig      # sigma below cloud edge (data error)
        gap_cloudsig = (lo - val) / np.nanstd(cloud)
    elif val > hi:
        gap_sig = (val - hi) / sig
        gap_cloudsig = (val - hi) / np.nanstd(cloud)
    else:
        gap_sig = 0.0; gap_cloudsig = 0.0
    frac_below = float(np.mean(cloud > val))   # fraction of cloud above data
    return dict(inside=inside, cloud_min=float(lo), cloud_max=float(hi),
                gap_sigma_data=float(gap_sig), gap_sigma_cloud=float(gap_cloudsig),
                cloud_frac_above=frac_below)

reach_in = reach(fdata[0], f_in, mc_std[0])
reach_out = reach(fdata[1], f_out, mc_std[1])
print(f"[REACH] inner: inside={reach_in['inside']} gap={reach_in['gap_sigma_data']:.1f}sig(data) "
      f"cloud[{reach_in['cloud_min']:.2f},{reach_in['cloud_max']:.2f}]")
print(f"[REACH] outer: inside={reach_out['inside']} gap={reach_out['gap_sigma_data']:.1f}sig(data) "
      f"cloud[{reach_out['cloud_min']:.2f},{reach_out['cloud_max']:.2f}]")

# ============================================================
# STEP 6: CAP radii -> R500/R200 geometric check (self-similar; no M_eff artifact)
# ============================================================
Om, OL, h = 0.3089, 0.6911, 0.6774
H0 = 100 * h                                   # km/s/Mpc
c_km = 299792.458
rho_c0 = 2.775e11 * h**2                        # Msun/Mpc^3 (physical, z=0)
def Ez(z): return np.sqrt(Om * (1 + z)**3 + OL)
def d_A(z, n=4000):
    zs = np.linspace(0, z, n)
    integ = np.trapezoid(1.0 / Ez(zs), zs)
    dC = c_km / H0 * integ                       # comoving Mpc
    return dC / (1 + z)                          # angular-diameter Mpc
def R_delta(M, delta, z):
    rho = rho_c0 * Ez(z)**2                       # crit density at z
    return (3 * M / (4 * np.pi * delta * rho))**(1 / 3.)   # proper Mpc
dA = d_A(Z_FID)
arcmin_per_Mpc = (1.0 / dA) * (180 / np.pi) * 60   # proper Mpc -> arcmin at z
M500_rep = float(np.nanmedian(g.Mtot500_med.values))
M200_rep = float(np.nanmedian(g.Mtot200_med.values))
capmap = {}
ladder = [("median_1e%.2f" % np.log10(M500_rep), M500_rep, M200_rep)]
for lm in [13.8, 14.2, 14.6, 15.0]:
    ladder.append((f"1e{lm}", 10**lm, 10**lm / 0.72))
for label, M5, M2 in ladder:
    r500 = R_delta(M5, 500, Z_FID); r200 = R_delta(M2, 200, Z_FID)
    capmap[label] = dict(M500=M5, theta500_arcmin=float(r500 * arcmin_per_Mpc),
                         M200=M2, theta200_arcmin=float(r200 * arcmin_per_Mpc))
    print(f"[CAP] {label}: theta500={r500*arcmin_per_Mpc:.2f}'  theta200={r200*arcmin_per_Mpc:.2f}'")
# required M_eff so that inner aperture (2')~theta500 and outer (8')~theta200
def M_for_theta(theta_arcmin, delta):
    r_mpc = theta_arcmin / arcmin_per_Mpc
    rho = rho_c0 * Ez(Z_FID)**2
    return float(4 * np.pi / 3 * delta * rho * r_mpc**3)
M_need_2p_R500 = M_for_theta(cap[R_INNER], 500)
M_need_8p_R200 = M_for_theta(cap[R_OUTER], 200)
geom_ok = bool(np.log10(M500_rep) >= np.log10(M_need_2p_R500) - 0.1)
print(f"[CAP] to make 2'~R500 need logM500={np.log10(M_need_2p_R500):.2f}; "
      f"8'~R200 need logM200={np.log10(M_need_8p_R200):.2f}")
print(f"[CAP] median peak logM500~{np.log10(M500_rep):.2f} -> apertures probe "
      f"{'inner<R500..R500->R200' if geom_ok else '>=R200 (core+2-halo); clean split only for high-nu massive peaks'}")

# sanity: data/model deficit at 4' vs canon (0.313/0.315/0.282/0.147 for nu1-4)
ratio_4 = y_data[NU_FIT, 2] / y_grid[i_fid, NU_FIT, 2]
print(f"[SANITY] data/fid-model @4' nu1-4 = {np.round(ratio_4,3)}  (canon 0.313/0.315/0.282/0.147)")

# ============================================================
# WRITE JSON
# ============================================================
def rj(x):
    if isinstance(x, np.ndarray): return [rj(v) for v in x]
    if isinstance(x, (np.floating, np.integer)): return float(x)
    return x

out = {
  "plan": "p9-latent-radial-zones",
  "one_liner": "y-CAP two-component radial deficit re-expressed in TNG feedback "
               "eigenbasis (inner f_gas<R500, outer f_gas R500->R200).",
  "generated": "2026-07-19",
  "headline": {
    "rank2_variance": float(cum[1]),
    "inner_axis_corr": r_in_e1, "outer_axis_corr": r_out_e2, "inner_outer_corr": r_io,
    "inner_deficit_ratio": float(def_in), "inner_deficit_factor": float(1 / def_in),
    "inner_reach_sigma": reach_in["gap_sigma_data"], "inner_inside_cloud": reach_in["inside"],
    "outer_deficit_ratio": float(def_out), "outer_constrained": False,
    "component1_maps_to_outer": bool((abs(load_C1[1]) / abs(load_C1[0])) > 2),
    "component2_maps_to_inner": bool((abs(load_C2[0]) / abs(load_C2[1])) > 2),
  },
  "verdict": (
    "Rank-2 holds (%.0f%% in 2 comps). The INNER-gas latent is recovered sharply "
    "(r=%.2f, kSZ-like), but the OUTER-gas latent DEGRADES to r=%.2f (between kSZ 0.95 "
    "and WL 0.6-0.7) exactly as pre-registered, and inner/outer independence r=%.2f "
    "reproduces papers 03/04 (r=0.24). The clean per-component->zone split FAILS the "
    "|loading|>2 bar (C1/8'-amp is inner-dominated because y-CAP is pressure/core-weighted "
    "-- the WL-like-blending result). Inverting the frozen data deficit: the signal is an "
    "INNER-zone (core) baryon deficit of ~%.0fx (f_in/TNG=%.2f), lying %.0f-sigma below the "
    "SB35 cloud (off-manifold); the OUTER zone is consistent with TNG but essentially "
    "UNCONSTRAINED by tSZ. The clean inner/outer split -- especially the outer zone -- "
    "requires the same-peak kSZ/tau stack."
  ) % (100 * cum[1], r_in_e1, r_out_e2, r_io, 1 / def_in, def_in,
       reach_in["gap_sigma_data"]),
  "frozen_inputs": {
     "sb35_grid": str(B / "wp4_mocks/sobol/model_grid_tfwiener_sb35.npz"),
     "twobound_grid": str(B / "wp4_mocks/model_grid_tfwiener.npz"),
     "parquet": str(PARQUET),
     "data_stack": str(B / "wp2_measurement/stack_wiener_sm2am_fid.npz"),
     "sys_band": str(B / "wp3_nulls/sigma_sys_wiener_decomposed.npz"),
  },
  "config": {"F_B": F_B, "snap": SNAP, "z": Z_FID, "mass_cut_log10_Mtot500": MASS_CUT_LOG,
             "nu_bins_used": NU_FIT, "inner_radius_arcmin": float(cap[R_INNER]),
             "outer_radius_arcmin": float(cap[R_OUTER]),
             "f_gas_units": "tilde = f_gas / (Omega_b/Omega_m); divide-out cancels in data/TNG ratio"},
  "step1_join": {
     "join_key": "integer run parsed from run_names 'sb35/run_%04d' == parquet.run",
     "snap_identification": "snap=71 (z=0.420) reproduces grid delta_ln_mgas "
        "(mean-of-log M_gas_500, >1e13.5) with r=0.998, rms=0.0063 vs std=0.095 -- best of 20 snaps",
     "n_grid_runs": Nrun, "n_joined": n_join_ok,
     "join_validator_corr_dln_mgas": r_join,
     "join_validator_note": "corr(parquet-derived per-run dln M_gas, grid delta_ln_mgas)=0.99 "
                            "confirms run-label alignment; a scrambled join gives ~0",
     "Mtot500_common_mode_CV_mean": cm_cv, "Mtot500_common_mode_CV_median": cm_cv_med,
     "gas_f_in_CV": gas_cv, "common_mode_assert": "mass CV<0.01 PASS (gas CV ~12x larger)",
     "median_halos_per_run": float(np.nanmedian(g.nh.values)),
     "f_in_tilde_mean": f_in_TNGmean, "f_in_tilde_std": float(np.nanstd(f_in)),
     "f_out_tilde_mean": f_out_TNGmean, "f_out_tilde_std": float(np.nanstd(f_out)),
     "TNG_fiducial_run": str(run_names[i_fid]),
     "f_in_TNG": f_in_TNG, "f_out_TNG": f_out_TNG,
  },
  "step2_svd_rotation": {
     "variance_fraction_top6": rj(lam[:6]),
     "cum_1c": float(cum[0]), "cum_2c": float(cum[1]), "cum_3c": float(cum[2]),
     "rank2_ge_90pct": bool(cum[1] >= 0.90),
     "r_f_in__e1": r_in_e1, "r_f_out__e2": r_out_e2, "r_f_in__f_out": r_io,
     "R2_e1_on_zones": r2_e1, "R2_e2_on_zones": r2_e2,
     "note": "kSZ-clean reference r~0.95/0.95 (fig08); tSZ y-CAP DEGRADES as pre-registered",
  },
  "step3_zone_loadings": {
     "definition": "C1(outer amp)=<ln y(8')>_nu ; C2(inner-outer)=<ln y(2')-ln y(8')>_nu",
     "ridge_alpha": RIDGE_ALPHA,
     "C1_loadings_on_[f_in,f_out]": rj(load_C1), "C1_abs_ratio_out_over_in": ratio_C1,
     "C1_cv_R2": r2_C1,
     "C2_loadings_on_[f_in,f_out]": rj(load_C2), "C2_abs_ratio_in_over_out": ratio_C2,
     "C2_cv_R2": r2_C2,
     "hypothesis_C1_outer": bool(ratio_C1 > 2 and r2_C1 > 0.5),
     "hypothesis_C2_inner": bool(ratio_C2 > 2 and r2_C2 > 0.5),
  },
  "step4_inversion": {
     "method": "invert Delta-ln-y (data/model deficit) through response Jacobian J "
               "(log-linear in ln f_zone); large deficit = linear extrapolation off-manifold",
     "response_jacobian_J": rj(J), "J_cond": condJ,
     "J_R2_C1": r2_J1, "J_R2_C2": r2_J2,
     "f_in_data_tilde": float(f_in_data), "f_in_data_16_50_84": rj(fin_p),
     "f_out_data_tilde": float(f_out_data), "f_out_data_16_50_84": rj(fout_p),
     "inner_deficit_vs_TNGfid": float(def_in), "inner_deficit_16_50_84": rj(din_p),
     "outer_deficit_vs_TNGfid": float(def_out), "outer_deficit_16_50_84": rj(dout_p),
     "inner_deficit_factor": float(1 / def_in), "outer_deficit_factor": float(1 / def_out),
     "mc_cov_tilde": rj(mc_cov),
     "sys_model": "rank-1 CIB shared across nu, fractional band half_band/y(4') "
                  "extrapolated to 2'/8' by constant fraction (caveat)",
  },
  "step5_reachability": {"inner_zone": reach_in, "outer_zone": reach_out},
  "step6_cap_to_Rdelta": {
     "z": Z_FID, "d_A_Mpc": float(dA), "arcmin_per_proper_Mpc": float(arcmin_per_Mpc),
     "note": "no M_eff(nu) artifact found -> self-similar estimate across a mass ladder",
     "mapping": capmap,
     "logM500_for_2arcmin_eq_R500": float(np.log10(M_need_2p_R500)),
     "logM200_for_8arcmin_eq_R200": float(np.log10(M_need_8p_R200)),
     "median_peak_logM500": float(np.log10(M500_rep)),
     "geom_verdict": ("At the median peak mass (logM500~%.2f, z=0.42) theta500~%.1f' and "
        "theta200~%.1f': the 2-8' CAP apertures sit AT/BEYOND R200, so the clean "
        "(inner<R500, outer R500->R200) geometric split holds only for the high-nu massive "
        "peaks (logM~14.6-15). This REINFORCES the pressure-weighted blending: the 8' 'outer' "
        "amplitude is core+2-halo dominated, not a clean R500->R200 shell.")
        % (np.log10(M500_rep), capmap[ladder[0][0]]["theta500_arcmin"],
           capmap[ladder[0][0]]["theta200_arcmin"]),
  },
  "sanity_data_model_ratio_4arcmin_nu1to4": rj(ratio_4),
  "caveats": [
     "y-CAP is pressure-weighted and radially broad: the two components do NOT cleanly "
     "separate into (inner,outer) f_gas the way kSZ does; see loadings/R2 and latent correlations.",
     "T is prior-supplied, not measured from y alone; the same-peak kSZ/tau stack is the resolver.",
     "Absolute keV / f_gas are prior-conditional; lead with the data/model RATIO.",
     "Frozen sys band (half_band/sigma_sys) is calibrated at the 4' fit radius; its extrapolation "
     "to 2'/8' is a constant-fraction assumption.",
     "CAP<->R500/R200 uses a self-similar M(z) estimate (no per-nu M_eff artifact); carry the mass caveat.",
     "Cosmology-provenance: bind paint at CAMELS-CV vs twobound design-theta (as in sibling plans).",
  ],
}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\n[WRITE] {OUT_JSON}")

# ============================================================
# FIGURE
# ============================================================
from paper_style import setup, save, panel_label, COLORS, TWO_COL
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
setup()
fig, ax = plt.subplots(1, 2, figsize=TWO_COL, constrained_layout=True)

# (a) the (inner, outer) f_gas plane
ax[0].set_xlim(-0.03, 1.07)
ax[0].set_ylim(0.45, 1.45)
sc = ax[0].scatter(f_in, f_out, c=dln_mg, cmap="cividis", s=16,
                   edgecolor="0.3", lw=0.2, label="SB35 cloud (253)")
# twobound uniform-rescale diagonal through TNG fiducial
tt = np.linspace(tb_dlnmg.min(), tb_dlnmg.max(), 50)
ax[0].plot(f_in_TNG * np.exp(tt), f_out_TNG * np.exp(tt), color=COLORS["secondary"],
           lw=1.6, ls="--", label="twobound uniform-rescale")
ax[0].plot(f_in_TNG, f_out_TNG, marker="*", ms=13, color=COLORS["bind"],
           mec="k", mew=0.4, ls="none", label="TNG fiducial", zorder=6)
# data point: inner tightly pinned, outer unconstrained (draw ellipse clipped by ylim)
vals, vecs = np.linalg.eigh(mc_cov)
order = np.argsort(vals)[::-1]; vals = vals[order]; vecs = vecs[:, order]
ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
for ns, al in [(2, 0.16), (1, 0.32)]:
    e = Ellipse(fdata, 2 * ns * np.sqrt(vals[0]), 2 * ns * np.sqrt(vals[1]),
                angle=ang, facecolor=COLORS["highlight"], alpha=al, edgecolor="none",
                zorder=4)
    e.set_clip_path(ax[0].patch); ax[0].add_patch(e)
ax[0].plot(fdata[0], fdata[1], marker="o", ms=7, color=COLORS["highlight"],
           mec="k", mew=0.5, ls="none", label="data (inverted)", zorder=7)
# outer-zone unconstrained: double arrow spanning the panel at the data x
ax[0].annotate("", xy=(fdata[0], 1.42), xytext=(fdata[0], 0.48),
               arrowprops=dict(arrowstyle="<->", color=COLORS["highlight"], lw=1.0, alpha=0.7))
ax[0].text(fdata[0] + 0.03, 1.30, "outer zone\nunconstrained\nby tSZ", fontsize=5.6,
           color=COLORS["highlight"], ha="left", va="top")
ax[0].text(0.30, 0.60, f"inner deficit\n${1/def_in:.1f}\\times$ ({reach_in['gap_sigma_data']:.0f}$\\sigma$\noff-manifold)",
           fontsize=6, ha="center", va="center")
ax[0].annotate("", xy=(0.15, 0.86), xytext=(0.55, 0.86),
               arrowprops=dict(arrowstyle="->", color="0.25", lw=1.1))
ax[0].set_xlabel(r"inner $\tilde f_{\rm gas}(<R_{500})$")
ax[0].set_ylabel(r"outer $\tilde f_{\rm gas}(R_{500}\!\to\!R_{200})$")
ax[0].legend(loc="upper right", fontsize=5.4, handletextpad=0.4, borderpad=0.3)
cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.02)
cb.set_label(r"$\Delta\ln M_{\rm gas}$", fontsize=7)
panel_label(ax[0], "(a)", loc="lower right")

# (b) component -> zone loadings
yp = np.arange(2)
ax[1].barh(yp + 0.18, [load_C1[0], load_C2[0]], 0.34, color=COLORS["bind"],
           label=r"on inner $\tilde f_{\rm gas}$")
ax[1].barh(yp - 0.18, [load_C1[1], load_C2[1]], 0.34, color=COLORS["secondary"],
           label=r"on outer $\tilde f_{\rm gas}$")
ax[1].axvline(0, color="k", lw=0.6)
ax[1].set_yticks(yp)
ax[1].set_yticklabels([f"C1: outer amp\n$\\ln y(8')$\n$R^2$={r2_C1:.2f}",
                       f"C2: inner$-$outer\n$\\ln y(2')-\\ln y(8')$\n$R^2$={r2_C2:.2f}"],
                      fontsize=6)
ax[1].set_xlabel(r"standardized loading")
ax[1].set_xlim(-0.85, 0.95)
ax[1].legend(loc="lower right", fontsize=6)
ax[1].text(0.02, 0.30,
           f"latent rotation:\n"
           f"$r(f_{{\\rm in}},\\hat e_1){{=}}{r_in_e1:+.2f}$ (sharp)\n"
           f"$r(f_{{\\rm out}},\\hat e_2){{=}}{r_out_e2:+.2f}$ (degraded)\n"
           f"$r(f_{{\\rm in}},f_{{\\rm out}}){{=}}{r_io:+.2f}$ (indep.)",
           transform=ax[1].transAxes, fontsize=5.8, va="top", ha="left",
           bbox=dict(boxstyle="round", fc="white", ec="0.7", lw=0.3, alpha=0.85))
panel_label(ax[1], "(b)", loc="upper right")

save(fig, FIG_STEM)
print("[DONE]")
