"""Assembles examples/paper_ksz_lightcone.ipynb — the P4b map-level companion
notebook (docs/ksz_lightcone_map_plan.md §5).

Builder (not a science module), same harness/idioms as `_build_ksz_paper2_nb.py`
/ `_build_lightcone_emulator_nb.py` (emit-only; execution is a separate
nbconvert step). Sections mirror the plan's phases P0-P6:

  §0  What this is — map-level P4b vs the per-halo capstone, link to the plan.
  §1  Geometry chain — the box->plane->map chain (§1.2), V0/V1/V1b/V1c/V2.
  §2  Mock catalogs — BGS/ELG/LRG selections, V3/V3b.
  §3  Closure + noise budget — map-vs-patch CAP agreement, V4, P4's noise panel.
  §4  Money plots — M1 (f~gas vs DESI kSZ, headline), M2 (LRG y-CAP vs Liu,
      regenerated), M4 (ELG z-shell), M3 (kappa mass anchor), M5 (map-level
      f~gas vs logM200, the mass-binned recreation of the per-halo headline
      figure), M6 (SB35 astro-param constraints from the kSZ-consistent node
      sets — a selection test, not a posterior) — each REGENERATED in-notebook
      from the small merged KS/lightcone/*.npz products (no raw
      {tau,y,kappa}_maps.npz loads; a single optional raw-map reproducibility
      cell is gated behind RUN_HEAVY).
  §5  Caveats — patch reuse, shared-box realization covariance, Born+smearing,
      beam, 253/256 nodes, empty-sample runs, Liu CSV bug.
  §6  P6b verdict, verbatim.

Every section ends with a 1-paragraph "what would look wrong" note. All
cells load only KS/lightcone/{*.npz,figs/*.png,verdicts/*.json} (~100MB of
npz -- M5's 506 per-node massbin shards are the largest single contributor,
~75MB, since no merged per-node product exists yet for the massbin sample --
+ a few MB of PNGs) — well under the plan's <300MB budget.

Run once:
    python examples/_build_ksz_lightcone_nb.py && jupyter nbconvert --to notebook \
        --execute --inplace examples/paper_ksz_lightcone.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python (BIND_env)", "language": "python", "name": "bind_env"}
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ============================================================== title
md(r"""
# BIND $\times$ DESI $\times$ ACT — **map-level lightcone confrontation** (P4b)

This notebook is the companion to `docs/ksz_lightcone_map_plan.md` (phases
P0-P6, all complete). It repeats the per-halo capstone's headline result —
*which TNG feedback regimes are consistent with the DESI$\times$ACT kSZ
data* — but on the **ray-traced lightcone maps** ($\tau$/$y$/$\kappa$, 50
realizations $\times$ 25 deg$^2$ per Sobol node) instead of individually
painted halo patches, adding real line-of-sight projection, the 2-halo term,
correlated LSS "noise" in the stack, and a realization-based covariance —
i.e. the analysis performed the way observers actually perform it.

**Relation to the per-halo capstone** (`examples/paper_ksz_desi_act_2.ipynb`,
`examples/_build_ksz_paper2_nb.py`): stacked CAP $\tilde f_{\rm gas}$ from
painted patches finds 49/256 SB35 nodes consistent with Ried Guachalla+25 BGS
kSZ (fiducial TNG $\sim$1.4-1.8$\times$ too gas-rich); the tSZ $y$-CAP leg was
computed on individual halos and was **not trusted** (shape/aperture
systematics, wrong LRG host mass). This notebook's map-level $\tilde f_{\rm
gas}$ closes to the per-halo result at small aperture (P4, KG2) and its
mass-matched LRG $y$-CAP (P6b) **replaces** the distrusted per-halo tSZ
figure. It also finds the map-level kSZ-consistent fraction is **much
larger** than the per-halo one — a real, diagnosed, explained finding (P6b,
§5 below), not a bug.

Consumes only the reduced products under `KS/lightcone/` (`KS =
/mnt/home/mlee1/ceph/bind_science/ksz_confront`) — no raw multi-GB
`{tau,y,kappa}_maps.npz` cubes are loaded by default (`RUN_HEAVY=False`).
""")

code(r"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/src")
sys.path.insert(0, "/mnt/home/mlee1/BIND-ksz2/examples")
warnings.filterwarnings("ignore", category=RuntimeWarning)

CEPH = Path("/mnt/home/mlee1/ceph")
KS = CEPH / "bind_science/ksz_confront"
LIGHTCONE = KS / "lightcone"
FIGS = LIGHTCONE / "figs"
VERDICTS = LIGHTCONE / "verdicts"

RUN_HEAVY = False   # flip True to re-derive one shard from the raw ~2.9GB tau_maps.npz (see the end of S3)

def show(name, width=920):
    p = FIGS / name
    if not p.exists():
        print(f"[missing] {p}"); return
    display(Image(str(p), width=width))

def verdict(phase):
    p = VERDICTS / f"{phase}.json"
    with open(p) as f:
        return json.load(f)

def pverdict(phase, keys=None):
    d = verdict(phase)
    print(f"[{d['phase']}] pass={d['pass']}  next={d.get('next')}")
    m = d.get("metrics", {})
    if keys:
        for k in keys:
            print(f"  {k}: {m.get(k)}")
    return d
""")

# ============================================================== §0
md(r"""
## $\S$0 · What this is

**Goal.** Confirm the per-halo kSZ/tSZ result survives (or is refined by)
going from "one painted 6.25 Mpc/h halo patch" to "a full ray-traced 25
deg$^2$ lightcone with 2-halo term, LOS projection, and realization noise" —
and quantify how the map-level DESI-mock $\tilde f_{\rm gas}(\theta)$ and
$y$-CAP$(\theta)$ compare to the real Ried Guachalla+25 (kSZ) and Liu+2025
(tSZ) data. See `docs/ksz_lightcone_map_plan.md` $\S$0 for the full framing
and $\S$1 for the geometry chain this notebook cites but does not re-derive.

**What would look wrong (this section):** if the verdict JSONs loaded below
don't all report `pass=True`, or if the plan file / verdict paths don't
exist, the pipeline this notebook depends on is incomplete — stop and check
`docs/ksz_lightcone_map_plan.md` $\S$3 before trusting anything downstream.
""")
code(r"""
for phase in ["P0","P1","P2","P3","P3prep","P3lrg","P4","P6a"]:
    try:
        d = verdict(phase)
        print(f"{phase:8s} pass={d['pass']!s:6s} next={d.get('next','')[:70]}")
    except FileNotFoundError:
        print(f"{phase:8s} MISSING")
print()
print("P6b (this phase):")
d = verdict("P6b")
print(f"  pass={d['pass']}  next={d['next']}")
""")

# ============================================================== §1 geometry
md(r"""
## $\S$1 · Geometry chain — box $\to$ plane $\to$ ray-traced map pixel

Four deterministic steps turn a halo's 3-D box position into the pixel(s) it
occupies in every traced map (plan $\S$1.2, implemented in
`src/bind/inference/lux_geometry.py`, validated in P0-P2):

- **A** — per-snapshot box transform (`LightconeTransforms`, seed 2020).
- **B** — slab assignment (`slab = floor(LOS/51.25) `) $\to$ global plane
  `p = 4*snap_idx + slab + 1`.
- **C** — plane pixelization: a literal `[51:51+4096]` center-crop of the
  native 4198px stage-1 grid (not a resample) — pins the ~4.8% area loss off
  the plane crop.
- **D** — per-realization rotation/displacement scatter (seed `1992+7*r`,
  identical across all 256 Sobol nodes) + Born ray geometry, including
  periodic tiling at high z.

P1 additionally measured the **Born deflection budget**: stacked centroid
offset grows from ~0.3px (z=0.18, BGS) to ~1.7px (z$\gtrsim$1) — real ray
deflection, not a bug — which is why P2 co-traced a haloplane (gold-standard
positions incl. deflection) and, for ELG (z=1.16), this notebook's $\tau$-CAP
stacks include an extra $\sigma$=0.222$'$ Gaussian smearing on top of the
Born-predicted positions (P2's fallback recipe).

Figures: **V0** (the $\chi(z)$ geometry table), **V1/V1b/V1c** (the P1
validation ladder: halo-pixel recovery, plane overlay, map-stack SNR),
**V2** (the P2 haloplane/massplane co-trace: bit-exact kappa identity over
47/50 realizations, the deflection budget above).
""")
code(r"""
pverdict("P0", ["n_records","rot_ok","disp_ok","chi_monotonic","source_z_ok"])
print()
pverdict("P1", [])
print("  check3 (map-stack SNR/centroid) — see figure for the full per-(snap,realization) table")
print()
pverdict("P2", ["kappa_identity_pass","deflection","realizations_complete_range"])
""")
code(r"""
show("V0_geometry.png")
""")
code(r"""
show("V1_halo_pixels.png")
show("V1b_plane_gallery.png")
show("V1c_map_stacks.png")
""")
code(r"""
show("V2_cotrace.png")
""")
md(r"""
**What would look wrong (this section):** any V1 panel showing a
peak/random SNR $\lesssim$5 or a stacked centroid $>$1px would mean halo
positions are not reliably recovered on the raw maps — everything downstream
(catalogs, CAP stacks, money plots) would be untrustworthy (plan kill-gate
KG1). V2's kappa-identity check failing to be bit-exact would mean the P2
co-trace does not actually share the fiducial's geometry, invalidating the
massplane denominator used in $\S$4 below.
""")

# ============================================================== §2 catalogs
md(r"""
## $\S$2 · Mock DESI catalogs

Three samples, one row per (halo, snapshot-shell), built by
`examples/lightcone_desi_catalog.py` (E2):

- **BGS** ($M_\star{>}10^{11.0},10^{11.25}$, painted central $M_\star$ within
  50 kpc/h, snap085 z$\approx$0.26): 2813 $\geq10^{13}M_\odot/h$ halos, mean
  $\log M_{200}=$13.63/13.81 (matches the per-halo reference 13.60/13.82 to
  0.03/0.01 dex).
- **ELG** (mass-proxy $\log M_{200}\in[12.0,12.6]$, snap046 z=1.16, **below
  BIND's painted 1e13 floor** $\to$ patch-reuse regime): 19189/23883 pass
  selection, `in_patch` fraction $\approx$29.4% (physical only where a
  low-mass halo happens to fall inside an existing $\geq10^{13}$ patch).
- **LRG** (mass-proxy window on snap067 z=0.503, tuned so mean
  $\log M_{200}=13.181$, matching Liu+2025's ACT-CMB-lensing DESI-LRG host
  mass 13.18 — Sailer+24 — to 0.001 dex; new this phase's predecessor,
  P3-LRG): 1769 selected (1691 after the plane-crop cut).

Figures: **V3** (dN/dz, mass distributions, one realization's BGS/ELG sky
scatter over the $\tau$ map), **V3b** (LRG mass histogram + sky scatter).
""")
code(r"""
pverdict("P3", ["n_bgs_halos","n_elg_halos","n_elg_sel","elg_in_patch_frac",
                "mean_logM200_cut11.0","mean_logM200_cut11.25"])
print()
pverdict("P3lrg", ["n_selected","n_selected_in_crop","mean_logM200","target"])
""")
code(r"""
show("V3_desi_mock.png")
""")
code(r"""
show("V3b_lrg_mock.png")
""")
md(r"""
**What would look wrong (this section):** BGS mean $\log M_{200}$ drifting
$>$0.1 dex from the per-halo reference (13.60/13.82) would mean the
map-level and per-halo samples are not actually comparable populations. An
`in_patch` fraction near 0% or 100% for ELG would mean the reuse-capture
geometry (halo positions vs. painted-patch positions) is broken, not just
noisy.
""")

# ============================================================== §3 closure
md(r"""
## $\S$3 · Closure — map-level CAP vs the painted-patch CAP (P4)

Before trusting a 256-node sweep, P4 checked that the SAME fiducial sample,
stacked on the ray-traced maps with the CAP filter, agrees with the
per-halo painted-patch CAP at small aperture (kill-gate KG2). Result:
**agreement at the 5-20% level for $\theta\lesssim0.25\,\theta_{200}$**,
growing into a smooth, monotonic **2-halo/LOS excess** at larger aperture —
the expected, documented *feature* of the map-level analysis, not a bug (a
discontinuous or sign-flipped excess would have been the failure mode to
watch for).

The **noise budget** (P4's `noise_panel`) compares two error estimates on
the fiducial's own CAP(theta): the 50-realization scatter (LSS + finite
per-realization sample) vs. a galaxy-bootstrap standard error (finite-N
resampling within one stack) — both grow with aperture and are
comparable in size, confirming the realization scatter is not a
numerical artifact.
""")
code(r"""
d4 = pverdict("P4", [])
gate = d4["metrics"]["gate"]
print("\nP4 small-aperture gate (per map,sample):")
for k, v in gate.items():
    print(f"  {k:14s} smallest<=0.25: {v['smallest_aperture_le0.25']!s:6s}  "
          f"monotone_nondecreasing: {v['monotone_nondecreasing']}")
np_ = d4["metrics"]["noise_panel"]
tv = np.array(np_["theta_value"])[:5]
rs = np.array(np_["realization_std"])[:5]
gb = np.array(np_["galaxy_bootstrap_se"])[:5]
print("\nnoise panel (first 5 xb points): theta_value / realization_std / galaxy_bootstrap_se")
for a,b,c in zip(tv, rs, gb):
    print(f"  {a:.3f}   {b:.4f}   {c:.4f}")
""")
code(r"""
show("V4_cap_closure.png")
""")
md(r"""
**What would look wrong (this section):** if the map/patch ratio were wild
(order-of-magnitude, not the observed smooth few-tens-of-percent growth) at
ALL apertures including the smallest, that would signal a units bug in the
$\tau/(\sigma_T x_e)$ normalization or the mass conversion — P4 explicitly
ruled this out by cross-checking the native-pixel-size $\tau$-per-gas-pixel
constant against the full-slab pipeline call (agreement to $<$0.02%).
""")

# ============================================================== §4 money plots
md(r"""
## $\S$4 · Money plots — M1 (headline), M2, M4, M3

Regenerated **in this notebook**, directly from the small merged products
(`KS/lightcone/{taucap,ycap,ycap_beam,kcap,capmat,elg*,lrg*}_lightcone.npz`,
$\approx$20MB total) — no raw `{tau,y,kappa}_maps.npz` cubes are loaded here.

- **M1** (`lightcone_m1_fgas_desiact.py`) — **the headline**: map-level
  $\tilde f_{\rm gas}(\theta)={\rm CAP}_\tau({\rm gas})/{\rm CAP}_{\rm mat}/F_B$
  (the massplane-denominator $\tilde f_{\rm gas}$, new this phase) for both
  BGS cuts, 253-node SB35 envelope + fiducial vs. the real Ried Guachalla+25
  points. **CAP_mat is per-node** (`lightcone_capmat_merge.py`, bug-fixed
  2026-07-24: each Sobol node's stellar-mass cut selects a different halo
  subset, so the denominator must be stacked at THAT node's own selected
  positions, not the fiducial's — an earlier version divided every node's
  gas column by the fiducial's matter column, inflating $\tilde f_{\rm gas}$
  to unphysical values for mass-mismatched extreme-feedback nodes). $\chi^2$
  consistency is reported in **two variants**: (a) survey-variance
  (`cov_ksz` $\oplus$ this node's own single-realization covariance — a weak
  test, since one 25 deg$^2$ realization has only ~40-90 galaxies) and (b)
  **DESI-precision** (`cov_ksz` $\oplus$ covariance of the realization
  *mean*, i.e. `cov_map/n_real` — the apples-to-apples number vs. the
  per-halo 49/256, 65/256, and the plotted/headline classification).
- **M2** (`lightcone_m2_ycap_liu.py`) — beamed $y$-CAP vs. Liu+2025: PRIMARY
  panel is the mass-matched LRG comparison (new this phase); SECONDARY panel
  keeps the mass/z-mismatched BGS-vs-Liu comparison from P6a for reference.
- **M4** (`lightcone_m4_elg_zshell.py`) — ELG (z=1.16) $\tau$-CAP and
  $\tilde f_{\rm gas}$ in the patch-reuse regime, plus the deflection-smearing
  sensitivity check.
- **M3** (`lightcone_m3_kappa_anchor.py`, P6a, unchanged) — the $\kappa$
  mass anchor: feedback-blind sanity check that the mock samples have the
  expected relative masses.
- **M5** (`lightcone_m5_fgas_mass.py`) — the map-level recreation of the
  per-halo headline figure ($\tilde f_{\rm gas}$ vs $\log M_{200}$, not vs
  $\theta$): 7 fixed FoF-$M_{200}$ bins at $z{=}0.18$ (BGS) and $z{=}1.16$
  (ELG), full 253-node envelope. See its own subsection below.
- **M6** (`lightcone_m6_params.py`) — turns M1's node SETS around: what do the
  31/253 & 48/251 kSZ-consistent nodes say about the 30 SB35 astro params
  (relative to their Sobol prior)? A selection test, not a posterior. See its
  own subsection below.
""")
code(r"""
import lightcone_m1_fgas_desiact as M1
import lightcone_m2_ycap_liu as M2
import lightcone_m3_kappa_anchor as M3
import lightcone_m4_elg_zshell as M4

m1_metrics = M1.main()
plt.close("all")
""")
code(r"""
show("M1_fgas_vs_desiact.png", width=1100)
""")
code(r"""
print("map-level consistent counts (this notebook's own recompute, per-node CAP_mat):")
for s in ["bgs110","bgs1125"]:
    c = m1_metrics["consistent_counts"][s]
    sv, dp = c["survey_variance"], c["desi_precision"]
    perhalo = m1_metrics["perhalo_consistent"][s]
    print(f"  {s}:")
    print(f"    DESI-precision (headline) : {dp['n_consistent']}/{dp['n_valid_nodes']}"
          f"  (cf. per-halo {perhalo[0]}/{perhalo[1]})")
    print(f"    survey-variance (side note): {sv['n_consistent']}/{sv['n_valid_nodes']}")
    print(f"    sb35 f~gas envelope range  : {c['sb35_envelope_min_max']}")
print("\nf~gas at theta(r200) (fiducial vs data):", m1_metrics["fgas_at_theta200"])
print("\ncosmic-crossing check (is f~gas>1 anywhere within the 1-halo regime?):")
print(m1_metrics["cosmic_crossing_check"])
""")
code(r"""
m2_metrics = M2.main()
plt.close("all")
""")
code(r"""
show("M2_ycap_vs_liu.png", width=1100)
""")
code(r"""
print("LRG (mass-matched) fiducial/Liu ratio range:", m2_metrics["lrg_ycap_fid_over_liu_range"]["min"],
      "-", m2_metrics["lrg_ycap_fid_over_liu_range"]["max"])
print("(for reference) BGS (mass-MISmatched) fiducial/Liu ratio ranges:",
      {s: (v["min"], v["max"]) for s, v in m2_metrics["bgs_vs_liu_ratio_range_mismatched"].items()})
""")
code(r"""
m4_metrics = M4.main()
plt.close("all")
""")
code(r"""
show("M4_elg_zshell.png", width=1100)
""")
code(r"""
print("ELG in_patch fraction:", m4_metrics["in_patch_fraction"])
print("ELG deflection-smearing effect: mean |dtau/tau| = %.1f%%, max = %.1f%%" % (
    m4_metrics["elg_smearing_effect_pct"]["mean_abs_pct"], m4_metrics["elg_smearing_effect_pct"]["max_abs_pct"]))
""")
code(r"""
m3_metrics = M3.main()
plt.close("all")
""")
code(r"""
show("M3_kappa_anchor.png", width=1100)
""")
md(r"""
### M5 — map-level $\tilde f_{\rm gas}$ vs $\log M_{200}$ (the headline figure, mass-binned)

M1 above regenerates the per-halo headline as a function of aperture
$\theta$ at **fixed** stellar-mass cuts. M5 regenerates the *other* headline
axis: $\tilde f_{\rm gas}$ vs **$\log M_{200}$** at fixed aperture, in 7
FoF-$M_{200}$ bins (`lightcone_cap_stack.py --sample massbin85/massbin46`,
`lightcone_capmat_merge.py --massbin`) — the direct map-level analogue of
`examples/_build_ksz_paper_nb.py` $\S\S$6b's `f6b_lowmass` panel. Left panel:
BGS shell (snap085, $z{=}0.18$) at $1.0\,\theta_{200}$. Right panel: ELG
shell (snap046, $z{=}1.16$) at the data's own fixed 1.0$'$ aperture. Both
panels carry the full 253-node SB35 envelope (`_p5_massbin_tasks.disbatch`,
506 tasks, complete).

**Headline closure.** At $\log M_{200}\ge13$ (the 4 highest bins, where BIND
actually paints halos as centrals) the map-level fiducial curve agrees with
the per-halo reference to **1-6%** — comfortably inside the P4 closure
gate's $\lesssim$10-15% — confirming the mass-binned stacking machinery
(per-bin positions, the CAP formula, the shared massplane denominator,
$F_B$) reproduces the trusted per-halo result.

**The reuse-regime story (sub-$10^{13}$ bins, shaded grey).** Below BIND's
painted mass floor, the pooled "all halos" curve is **diluted**: roughly
half the ELG/BGS-shell low-mass sample sits outside any painted
$\ge10^{13}$ patch, so those pixels are pure unpainted background/2-halo
signal with no real gas physics, pulling the pooled curve well below the
per-halo reuse reference. The dashed **in\_patch-only** substack (halos that
do land inside a painted patch) is the honest read of this regime — it
still doesn't match the per-halo curve one-for-one (the per-halo pipeline
reads the CAP off the local 6.25 Mpc/h patch only, whereas this map's
$\tau$ is the full line-of-sight integral to $z_s{=}2.44$, so a correlated
2-halo/LOS term dilutes even the in\_patch subset), but it is the correct
apples-to-apples comparison. Concretely, at the ELG data's closest bin
(bin1, $\log M_{200}\approx12.5$): the in\_patch fiducial sits at
**0.578**, $\approx$1.3$\times$ the ELG data ($\approx$0.41-0.46) — same
direction ("BIND too gas-rich") as the per-halo headline, but a smaller
excess than the per-halo pipeline's $\approx$1.9$\times$ at 2.8$\theta_{200}$
(the pooled all-halo curve, by contrast, sits *below* the data — the
opposite sign, purely from the reuse dilution above).
""")
code(r"""
import lightcone_m5_fgas_mass as M5

m5_metrics = M5.main()
plt.close("all")
""")
code(r"""
show("M5_fgas_mass.png", width=1100)
""")
code(r"""
print("M5 >=1e13 closure (map fiducial / per-halo reference, plan's headline sanity gate):")
d5 = verdict("M5")
print(" ", d5["metrics"]["fid_fgas_per_bin_snap085"]["map_over_per_halo_ratio_ge1e13_bins"])
print()
print("SB35 envelope range (max-min across 253 nodes) per bin, BGS panel (massbin85):")
for c, lo, hi in zip(m5_metrics["massbin85_bin_centers"],
                      m5_metrics["massbin85_envelope_min"],
                      m5_metrics["massbin85_envelope_max"]):
    print(f"  logM200={c:.2f}: [{lo:.3f}, {hi:.3f}]  range={hi-lo:.3f}")
print()
print("BGS-data 'touch' count (>=1e13 bins only -- all 5 BGS points live at logM200=13.36-13.82):")
print("  per M*-cut, out of", m5_metrics["bgs_touch_n_nodes_total"], "nodes:",
      m5_metrics["bgs_touch_counts_per_cut"])
print("  nodes touching >=1 of the 5 BGS points:", m5_metrics["bgs_touch_n_nodes_any"])
print("  nodes touching ALL 5 BGS points        :", m5_metrics["bgs_touch_n_nodes_all5"])
""")
md(r"""
**What would look wrong (M5):** the pooled ("all halos") and in\_patch-only
curves are expected to diverge from each other and from the per-halo
reference **only inside the shaded sub-$10^{13}$ patch-reuse region** — any
such divergence appearing **outside** that shading (i.e. at $\log
M_{200}\ge13$) would mean the reuse-dilution story is leaking into the
regime BIND actually paints, which should not happen. Separately, the
$\ge10^{13}$ bins disagreeing with the per-halo reference by **more than
$\sim$15%** (vs. the 1-6% actually observed) would violate the P4 closure
gate (KG2) and mean the mass-binned stacker itself — not just the
low-mass reuse physics — has a bug; neither failure mode is present here.
""")
md(r"""
### M6 — SB35 astro-param constraints from the kSZ-consistent node sets

M1-M5 above are all in the *observable* space (aperture, mass). M6 flips the
question around: **given the kSZ-consistent node SETS** (`lightcone_m1_
fgas_desiact.py`'s DESI-precision classification, saved as
`KS/lightcone/ksz_consistent_nodes.npz` — 31/253 for BGS $M_\star{>}10^{11.0}$,
48/251 for $M_\star{>}10^{11.25}$), **what do they say about the 30 SB35
astro parameters** relative to their Sobol prior box?

**This is a SELECTION test, not a posterior** — node membership is a hard
$\chi^2$-threshold cut, not a likelihood weight; panel (a)'s percentile bars
describe the empirical distribution of the selected Sobol nodes, not a
credible interval. Params are prior-normalized to $[0,1]$ over the empirical
Sobol box (256-row min/max), log10-uniform for the 19/30 params flagged
`LogFlag==1` (the same convention the Sobol design itself was drawn under —
`bind.params.PARAM_LOG_FLAG`). Panel (a) sorts all 30 params by KS-test
significance (smaller of the two BGS-cut p-values vs the full 253-node
grid); panel (b) is a pairwise scatter of the top-3 most-constrained params.
""")
code(r"""
import lightcone_m6_params as M6

m6_metrics = M6.main()
plt.close("all")
""")
code(r"""
show("M6_ksz_param_constraints.png", width=1100)
""")
code(r"""
d6 = verdict("M6")
print(f"M6 pass={d6['pass']}")
print("n_consistent [bgs110, bgs1125]:", d6["metrics"]["n_consistent"],
      " overlap:", d6["metrics"]["overlap"])
print("\ntop-5 most-constrained params (name, ks_p_min, width_ratio, median_shift -- "
      "using whichever BGS cut gave the smaller p):")
for t in d6["metrics"]["top5"]:
    cut = t["ks_p_min_sample"] if "ks_p_min_sample" in t else (
        "bgs110" if t["ks_p_bgs110"] <= t["ks_p_bgs1125"] else "bgs1125")
    wr = t[f"width_ratio_{cut}"]
    ms = t[f"median_shift_{cut}"]
    print(f"  {t['name']:32s} p={t['ks_p_min']:.4f} ({cut})  width_ratio={wr:.2f}  median_shift={ms:+.3f}")
print("\nmultiple testing:", d6["metrics"]["multiple_testing"])
print("\ncross-check vs the per-halo D5/D5-CAP wind/SN sector:")
print(" ", d6["metrics"]["wind_sn_sector_cross_check"]["note"])
""")
md(r"""
**What would look wrong (M6):** a parameter constrained at high significance
that has **no plausible gas-physics role** (e.g. a UVB or SNIa-rate param
landing in the top few with $p\ll0.05$) would suggest a selection artifact —
some spurious correlation between that parameter and the Sobol sequence's
finite-$N$ sampling noise, or a bug in the node-ID alignment between
`ksz_consistent_nodes.npz` and the design matrix — rather than a real kSZ
constraint. What is actually observed instead is reassuring: the leading
signals are `IMFslope`, `WindEnergyIn1e51erg`, `VariableWindSpecMomentum`,
`VariableWindVelFactor`, and `WindFreeTravelDensFac` — all direct levers on
how efficiently galactic winds eject/retain gas (or, for IMF slope, the
stellar-to-gas mass budget feeding those winds) — physically exactly where a
gas-content observable like $\tilde f_{\rm gas}$ should be most sensitive.
Three of the four per-halo D5/D5-CAP "wind/SN sector" params
(`VariableWindVelFactor`, `WindEnergyIn1e51erg`, `WindFreeTravelDensFac`)
independently land in this M6 top-5, a non-trivial cross-check between two
very different pipelines (per-halo GP-emulated posterior vs. map-level
selection test) landing on the same feedback knobs. Also worth checking: the
`filled_box_check` in the verdict should report "filled" (empirical Sobol box
recovered by large random subsets to a few percent) — if it does not, the
prior-normalization itself (and everything derived from it) is untrustworthy.
""")
md(r"""
### Optional: reproduce one shard from the raw ray-traced maps (`RUN_HEAVY`)

The money plots above consume only the small pre-reduced `KS/lightcone/*.npz`
products. To verify the reduction itself (not just the merge/plot step),
flip `RUN_HEAVY=True` at the top of this notebook and re-run the next cell —
it loads the fiducial's raw `tau_maps.npz` ($\approx$1GB for one map type,
50 realizations $\times$ 5 source planes $\times$ 1024$^2$ f4) and
recomputes the BGS $M_\star{>}11.0$ shard from scratch via
`lightcone_cap_stack.run_one`, then diffs it against the shard already on
disk. **Not run by default** (this is the one intentionally-gated heavy
path in this notebook, per the plan's `<300MB` / no-default-raw-map-load
budget).
""")
code(r"""
if RUN_HEAVY:
    import lightcone_cap_stack as CS
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = CS.run_one("fid", "tau", "bgs110", out_dir=Path(td), force=True, verbose=True)
        fresh = np.load(p)
        onfile = np.load(LIGHTCONE / "shards" / "bgs110_tau_runfid_beamnone_src4.npz")
        print("max|mean diff| vs the on-disk shard:",
              np.nanmax(np.abs(fresh["mean"] - onfile["mean"])))
else:
    print("RUN_HEAVY=False -- skipped (flip the flag at the top of the notebook to run).")
""")
md(r"""
**What would look wrong (this section):** if M1's recomputed consistent
counts differ from the P6b verdict's cached numbers, or if `cosmic_crossing_check`
showed the fiducial exceeding 1 WITHIN the 1-halo regime ($\theta\le1.4\,
\theta_{200}$), that would indicate either a stale merged product or a real
units bug in the $\tilde f_{\rm gas}$ definition — neither is the case here
(fiducial stays $\le$0.84/0.87 throughout the 1-halo regime; the cosmic
asymptote is only approached at $\theta\sim2.8$-$3.0\,\theta_{200}$, exactly
mirroring the per-halo money plot's own "cosmic" f$_{\rm gas}=1$ annotation).
If the RUN_HEAVY cell's diff is not $\approx$0, the shard reduction itself
(not just this notebook's re-plotting) has drifted from what's on disk.
""")

# ============================================================== §5 caveats
md(r"""
## $\S$5 · Caveats (read before quoting a number from this notebook)

- **Patch reuse (ELG only).** BIND paints only $M_{200}\ge10^{13}\,M_\odot/h$
  halos; ELG hosts (logM200 12.0-12.6) are below that floor and are physical
  only where they land inside an existing patch's 6.25 Mpc/h cutout — here,
  **29.4%** of the selected ELG sample (`in_patch`, P3). M4's panels carry
  the grey/green patch-reuse shading throughout (the *entire* ELG curve is
  in this regime, unlike the per-halo figure's partial mass range).
- **Per-node CAP_mat (BGS only, bug found and fixed this session).** BGS's
  M$_\star$-cut selects a DIFFERENT halo subset per Sobol node (feedback
  changes which halos cross the threshold — e.g. node 64 selects only 11
  halos at mean logM200=14.38, vs the fiducial's 1753 halos at 13.44). An
  earlier version of M1 divided every node's own gas column by the
  FIDUCIAL's matter column, which inflated $\tilde f_{\rm gas}$ to
  unphysical values (4-16) for such mass-mismatched nodes. Fixed by stacking
  the (fiducial-shared, single-trace) massplane FIELD at each node's OWN
  selected positions (`lightcone_capmat_merge.py`). ELG/LRG select by a
  fixed mass-proxy cut with **no** per-node dependence (independently
  verified byte-identical across nodes), so their single shared CAP_mat was
  always correct and needed no fix.
- **Shared-box realization covariance.** All 50 (47 for the massplane)
  realizations trace the SAME fiducial DMO box under different
  rotation/displacement — this **underestimates** true cosmic variance
  (no independent volumes) while *also*, in M1's per-node covariance,
  being **inflated** by the small per-realization galaxy count (tens of
  galaxies in one 25 deg$^2$ field, far below DESI's actual footprint) —
  the net effect that drives M1's map-level-vs-per-halo consistent-count gap
  (P6b `consistent_counts.note`). Both directions of this caveat matter;
  neither cancels the other in general.
- **Born + deflection smearing.** The analytic pixel chain uses Born
  (undeflected) ray positions; P1/P2 measured the true stacked-centroid
  deflection (0.3-1.7px, growing with z). For ELG (z=1.16) this notebook's
  $\tau$-CAP additionally applies a $\sigma$=0.222$'$ Gaussian smearing on
  top of Born positions (P2's recipe) — measured here (M4) to suppress the
  ELG $\tau$-CAP by a **mean 14%, up to 20%** at the smallest apertures,
  because that smearing scale is a large fraction of ELG's own
  $\theta_{200}\approx0.36'$ (negligible for BGS/LRG, whose $\theta_{200}$
  is several arcmin).
- **Beam.** ACT DR6 $y$-map $\approx$1.6$'$ FWHM Gaussian, applied before
  $y$-CAP in M2 (both BGS and LRG panels).
- **253/256 Sobol nodes.** Runs 0114/0115/0117 were never ray-traced (P5) —
  every merged product covers 253 nodes, not 256; fractions (not raw counts)
  are the number to compare against the per-halo 49/256, 65/256.
- **Empty-sample nodes.** Runs 0064/0087 have **zero** galaxies passing the
  BGS $M_\star{>}11.25$ cut — a genuine physical empty sample (matches the
  per-halo reference counts exactly, P3/P5), not missing data; those two
  rows are NaN in every `bgs1125` array.
- **Liu+2025 release CSV bug.** The 4 photo-z-bin columns (`pz1`-`pz4`) in
  `tsz_liu2025_official.npz` are **numerically identical** — an export bug
  in the downloaded release, not a modeling choice (P3lrg/P6a/P6b, re-checked
  in M2 above); `pz1` is used throughout with no loss of generality.

**What would look wrong (this section):** any of these caveats becoming
UNquantified (e.g., an `in_patch` fraction that can't be recomputed from the
catalog, or a beam/smearing parameter silently changed between the shard
generation and this notebook's re-plot) would mean a figure above is telling
a story its own caveat text can't support — every number quoted here should
trace back to a verdict JSON or a shard/merge script docstring.
""")

# ============================================================== §6 verdict
md(r"""
## $\S$6 · P6b verdict (verbatim)
""")
code(r"""
d = verdict("P6b")
print(json.dumps(d, indent=2))
""")

# ============================================================== §7 P4c
md(r"""
## $\S$7 · P4c addendum (2026-07-28): REAL ACT DR6 tSZ + DES Y3 WL vs the kSZ-selected subspace

Everything above confronted BIND with *published reductions*; P4c
(`docs/tsz_des_data_plan.md`) replaces the buggy Liu+2025 csv with **our own
stacked y-CAP measurement** on the ACT DR6+Planck y-map at 160,150 DESI DR1
SGC spec-LRG positions (engine `examples/act_ycap_measure.py`, validated on
ACT DR5 clusters at S/N 22–32 with passing nulls), and adds an independent
**DES Y3 weak-lensing** confrontation (`examples/des_bind_forward.py` /
`des_bind_consistency.py`).

**Fig M2R** (supersedes §4's M2 as the tSZ money plot). Three estimator
lessons bought with real data (first-draft M2R was wrong; see
`lightcone_m2r_ycap_real.py` docstring): (1) the discrete CAP filter is
pixel-scale-dependent — the data must be resampled to BIND's 0.29296875'/px
(at 0.5'/px the θ≲1.6' apertures are a *different estimator*); (2) compare
per-object-scaled xb=θ_d/θ200 apertures on BOTH sides, never fixed-arcmin vs
scaled; (3) the baseline ILC is dust/CIB-biased (negative!) at small θ —
the deproj-CIB map is the primary data vector, the 11-map spread a
covariance systematic. χ² on xb≤1.4 (the M1 kSZ 1-halo convention):
**30/253 nodes tSZ-consistent; P(tSZ|kSZ)=0.81 vs P(tSZ)=0.12 (~7×
enrichment); Spearman ρ(kSZ χ², tSZ χ²)=0.90**. The TNG300 fiducial
over-predicts small-aperture y by up to ~5.7× → the real sky prefers the
strong-feedback edge, the same direction as the kSZ f̃_gas selection.
""")
code(r"""
show("M2R_ycap_real.png", width=1150)
pverdict("T3")
""")
md(r"""
**Fig M7 (Stream D).** DES Y3 ξ± via the D1-mandated forward model
(pyccl theory Cl × the paired-DMO suppression ratio S_ab(ℓ) — direct Hankel
from the 5° FOV is fatally ℓ<87-truncated): raw TNG300 amplitude is rejected
for **all** 253 nodes (the fixed-cosmology trap), one amplitude nuisance
(B_fid=0.79 ≈ the S8 offset) accepts **all** 253 at χ²/dof≈1.0 — the ξ+
node envelope is ≤0.4% wide vs 11–19% errors, so DES 2pt carries **no
residual feedback discrimination**. Yet its χ² *ranking* tracks the kSZ
selection (ρ=0.78, median Δχ²=−5.8). The map leg (45 GLIMPSE/Wiener 5°
patches) is uniformly rejected (χ²/dof~20) for every node — the
reconstruction-filter systematic dominates; consistency-check-grade only,
as planned.
""")
code(r"""
show("M7_des_wl.png", width=1150)
pverdict("D4")
""")
md(r"""
**Capstone X.** The multi-probe verdict: **25 of the 31 kSZ-consistent
nodes survive the real-tSZ test** (12% base rate), and the survivors occupy
the flatter-IMF / lower-WindEnergy corner of the M6 parameter plane (median
IMFslope −2.08 vs −2.34, log₁₀WE 0.28 vs 0.62). DES WL co-ranks but cannot
cut. The kSZ-selected feedback subspace **survives multi-probe
confrontation**.
""")
code(r"""
show("X_multiprobe.png", width=1150)
pverdict("X")
""")

md(r"""
**Phase L — the 2-D gas-latent corner + the 30-dim backtrack.** After X, DES
WL was dropped as a constraint (user decision; D4: no residual feedback
discrimination). The kSZ (M1) and tSZ (T3) per-node χ² become importance
weights exp(−Δχ²/2) over the 253-node Sobol design — a *pseudo-posterior*
under the flat box prior (ESS 9/13/6 for kSZ/tSZ/joint; importance-sampling
grade, not MCMC) — shown in the parent paper's rotated gas latent (97% of
the (τ,y)-profile response in 2 components; ê1: r=+0.95 with
f̃_gas(<R500), ê2: r=+0.95 with the R500→R200 shell). **Both probes
independently select the low-inner-gas edge of the design; joint:
f̃_gas(<R500)=0.469+0.031/−0.005, f̃_gas(R500→R200)=0.881+0.066/−0.009 of
cosmic.** Backtrack to 30 params (`figs/L_param_backtrack.png`):
width-stable constraints on {IMFslope↑, RadioFeedbackReiorientationFactor↑,
ThermalWindFraction↓, UVBHepbeta} (top-8 rank ρ~0.7 under tempering/LOO —
report cautiously); median-pull *directions* are robust for 16/18 strongly
shifted params (wind/SN energy sector LOW: WindEnergyIn1e51erg 0.21,
WindEnergyReductionFactor 0.12, VariableWindVelFactor; IMFslope HIGH 0.77;
WindFreeTravelDensFac HIGH 0.82; BlackHoleEddingtonFactor LOW 0.20;
QuasarThreshold HIGH — `verdicts/L.json` robustness block).
""")
code(r"""
show("L_latent_corner.png", width=900)
show("L_param_backtrack.png", width=900)
pverdict("L")
""")

nb.cells = cells
out = "examples/paper_ksz_lightcone.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
