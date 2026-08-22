"""Generate examples/lightcone_emulator.ipynb — the field-level statistics emulator.

One runnable notebook for the whole `bind.emulator` workflow: assemble the SB35
Sobol suite into a training table, fit the emulator (GP / MLP / flow backends),
predict every ray-traced lightcone statistic in ~ms, show the feedback response,
and validate with response-aware k-fold CV + the twobound-corner OOD test.

A `QUICK` toggle keeps the in-notebook run light (CPU, minutes); the full run and
the GPU-only pieces (WST/DM backfill, the normalizing-flow backend) point to the
SLURM scripts.  Engine: src/bind/emulator/.  Run: python examples/_build_lightcone_emulator_nb.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md(r"""# `bind.emulator`: instant ray-traced lightcone statistics for the TNG model

Feed in the **30 SB35 astro/feedback parameters** $\theta$ (+ a source redshift
$z_s$) and get **every field-level statistic of the BIND$\to$TNG weak-lensing /
tSZ / kSZ lightcone in milliseconds**, with calibrated $1\sigma$ — the TNG analogue
of the BCM "instant-$C_\ell$" emulators (Stage-B of `docs/wl_tsz_plan.md` §5).

Statistics emulated: tomographic $C_\ell^{\kappa\kappa}$, the WL **suppression**
$S(\ell)=C/C^{\rm DMO}$, $C_\ell^{\kappa y},\,C_\ell^{yy},\,C_\ell^{\kappa\tau},\,
C_\ell^{\tau\tau},\,C_\ell^{y\tau}$, peak & minima counts, the convergence PDF +
moments + Minkowski functionals, the wavelet scattering transform, dispersion-measure
stats, the tSZ $R(\nu)$ at peaks, and the $Y$–$M$ / $f_{\rm gas}$–$M$ / $T$–$M$
relations.

**This notebook runs the whole pipeline.** Set `QUICK=True` for a fast CPU pass; the
full design and the GPU-only steps (WST/DM backfill, the flow backend) are run with
the SLURM scripts noted below.

Two design choices worth knowing up front:
* We **emulate the suppression $S(\ell)$ directly**, not by dividing a predicted raw
  $C_\ell$ — the raw spectrum is dominated by the fixed-cosmology $\Lambda$CDM shape
  and PCA buries the few-% baryon ratio. $C_\ell^{\kappa\kappa,\,\rm auto}=S\cdot
  C_\ell^{\rm DMO}$ is reconstructed at predict time.
* The default backend is a **Gaussian process**, not an MLP: with $\sim$100–256
  design points in 30-d the GP's smooth prior interpolates the feedback response,
  where an MLP ensemble over-fits and washes it toward the mean.""")

code(r"""%load_ext autoreload
%autoreload 2
import warnings; warnings.filterwarnings("ignore")
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import bind
from bind.emulator import EmulatorDataset, assemble
from bind.emulator.core import Emulator

# ── knobs ─────────────────────────────────────────────────────────────────────
QUICK       = True          # fast CPU settings; set False for the full run (slow — use SLURM)
# "auto" = GPU exact-GP (gpgpu) when a GPU is present, else sklearn "gp" on CPU.
# Other backends: "gpgpu" (force GPU GP), "mlp" (fast, approximate), "flow" (GPU, generative).
BACKEND     = "auto"
N_COMPONENTS = 12
RUNS_DIR    = Path("/mnt/home/mlee1/ceph/bind_sb35/runs")
DATASET     = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz")
BUNDLE_DIR  = Path("/mnt/home/mlee1/ceph/bind_sb35/emulator"); BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
print("QUICK =", QUICK, "| backend =", BACKEND)""")

md(r"""## 1. Backfill the emulator-only statistics (WST + DM) — *GPU, optional*

The original stats stage wrote $C_\ell$, peaks, PDF/Minkowski and $R(\nu)$ but **not**
the wavelet scattering transform (`wst.npz`) or the dispersion-measure stats
(`dm_stats.npz`). WST is GPU-bound (~95 s for two $1024^2$ maps on CPU), so we
backfill it on the cluster, skip-if-exists, and re-assemble:

```bash
sbatch run_emulator_stats.sh          # GPU array over the suite -> wst.npz / dm_stats.npz
bind-emulator-assemble --out $DATASET # re-pack once the backfill lands
```

The assembler below simply skips WST/DM on runs that don't have them yet (a per-stat
validity mask), so everything else works before the backfill completes.""")

md(r"""## 2. Assemble the training table

Walk every completed run in the suite into one `emulator_dataset.npz`: the 30 astro
params (normalized to $[0,1]$), every statistic block + its bin axes, the $Y$–$M$
family from the pre-reduced `integrated.parquet`, and the DMO trace for $S(\ell)$.""")

code(r"""if DATASET.exists():
    ds = EmulatorDataset.load(DATASET)
    print(f"loaded cached dataset ({ds.n_runs} runs)")
else:
    ds = assemble(RUNS_DIR, verbose=True)
    ds.save(DATASET)
print(ds.summary())""")

md(r"""## 3. Fit the emulator

Each statistic is compressed (transform $\to$ standardize $\to$ PCA, with a
variance-based component cap) and a backend regresses $\theta\to$ PCA-latent.
The GP gives analytic $1\sigma$; the whole fit is CPU-light.""")

code(r"""import torch
print("CUDA available:", torch.cuda.is_available(),
      "→ backend 'auto' uses", "gpgpu (GPU)" if torch.cuda.is_available() else "gp (CPU)")
bkw = None
if QUICK and BACKEND == "gp":
    bkw = {"n_restarts": 0}
elif QUICK and BACKEND in ("gpgpu", "auto"):
    bkw = {"epochs": 150}
elif QUICK and BACKEND == "mlp":
    bkw = {"epochs": 300, "n_models": 4}

t = time.time()
em = Emulator(backend=BACKEND, n_components=N_COMPONENTS, backend_kwargs=bkw).fit(ds, verbose=True)
print(f"\nfit {len(em.statistics)} statistics in {time.time()-t:.0f}s")
bundle = BUNDLE_DIR / f"lightcone_emulator_{em.backend}.pt"
em.save(bundle); print("saved", bundle)""")

md(r"""## 4. Predict — every statistic in milliseconds

`em.predict(params, z_s=...)` returns a dict of statistics (+ `*_err` $1\sigma$), the
derived `suppression`, the bin `axes`, and `source_redshifts`. `params` is the full
35-vector (cosmology slots ignored — fixed at TNG300) or a 30-vector of astro params;
`z_s` interpolates the source-plane statistics (omit for all five planes).""")

code(r"""fid = bind.fiducial_params()
t = time.perf_counter(); out = em.predict(fid, z_s=1.0); dt = 1e3*(time.perf_counter()-t)
print(f"predict(z_s=1.0) in {dt:.1f} ms\n")
for k in em.statistics:
    print(f"  {k:16s} {np.asarray(out[k]).shape}")
print(f"  {'suppression':16s} {out['suppression'].shape}  (derived S=C/C_DMO)")""")

md(r"""## 4b. User control — pick statistics, multipoles, $\nu$, masses

`predict` takes `stats=[...]` to return only the statistics you want, and
`grids={axis: values}` to resample each statistic onto your own bins — `ell` for the
power spectra / suppression (interpolated in log-space), `nu` for peaks / minima /
Minkowski / $R(\nu)$, `log_mass_bins` for the scaling relations. `em.describe()` lists
what is available and each statistic's native bin range.""")

code(r"""print(em.describe())

# only the suppression + tSZ auto, on MY multipoles:
my_ell = np.logspace(2.5, 4.0, 15)
o = em.predict(fid, z_s=1.0, stats=["suppression", "cl_yy"], grids={"ell": my_ell})
print("\nsuppression on my_ell:", o["suppression"].shape, "| cl_yy:", o["cl_yy"].shape)

# peak counts on MY S/N thresholds:
o = em.predict(fid, z_s=1.0, stats=["peak_counts"], grids={"nu": np.linspace(0, 5, 11)})
print("peak_counts on 11 nu:", o["peak_counts"].shape)""")

md(r"""## 5. The feedback response

The headline use case: sweep a feedback knob across its prior and watch the WL
suppression $S(\ell)$, the tSZ $C_\ell^{yy}$, and the $Y$–$M$ relation move — instantly.
`BlackHoleRadiativeEfficiency` and `IMFslope` are the WL-dangerous knobs; wind energy
mostly acts through parameter *combinations* (so a single-axis sweep moves $S$ little).""")

code(r"""PARAM = "BlackHoleRadiativeEfficiency"
fracs = np.linspace(0.0, 1.0, 7)
preds = [em.predict(bind.vary_param(PARAM, fraction=f), z_s=1.0) for f in fracs]
ell = out["axes"]["ell"]; mlb = out["axes"].get("log_mass_bins")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for p, f in zip(preds, fracs):
    c = plt.cm.viridis(f)
    ax[0].semilogx(ell, p["suppression"], color=c, label=f"{PARAM[:14]} {f:.2f}")
    ax[1].loglog(ell, p["cl_yy"], color=c)
    if mlb is not None:
        ax[2].semilogy(mlb, p["scaling_Y"], color=c, marker="o", ms=3)
ax[0].axhline(1, ls=":", c="k", lw=.8); ax[0].legend(fontsize=7)
ax[0].set(xlabel=r"$\ell$", ylabel=r"$S(\ell)$", title="WL suppression ($z_s=1$)")
ax[1].set(xlabel=r"$\ell$", ylabel=r"$C_\ell^{yy}$", title="tSZ auto-power")
ax[2].set(xlabel=r"$\log_{10}M$", ylabel=r"$Y_{500c}$", title="Y–M relation")
fig.tight_layout(); plt.show()""")

md(r"""## 6. Validation — k-fold CV + twobound OOD (response-aware)

The metric that matters is the **response** $R^2$: per bin, how well the emulator
predicts each run's *deviation from the ensemble mean* (a pooled per-element $R^2$ is
dominated by the bin-shape every run shares and would read $\approx 1$ even if the
emulator ignored $\theta$ entirely). `amp_ratio` is the recovered response amplitude
(1.0 = perfect). The OOD test trains on the interior Sobol cloud and predicts the 60
**twobound prior-corner** runs — where over-fitting shows up first.""")

code(r"""# the validation engine lives alongside this notebook
from emulator_validation import kfold_cv, ood_twobound, _print_table

kf = 3 if QUICK else 5
cv = kfold_cv(ds, backend=BACKEND, kfolds=kf, n_components=N_COMPONENTS, backend_kwargs=bkw)
_print_table(f"{kf}-fold CV (Sobol)", cv)

ood = ood_twobound(ds, backend=BACKEND, n_components=N_COMPONENTS, backend_kwargs=bkw)
if ood:
    _print_table("OOD (twobound corners)", ood)""")

code(r"""# response-R2 bar chart: Sobol CV vs twobound OOD
names = list(cv.keys()); x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(11, 4.3))
ax.bar(x-0.2, [cv[n].get("response_r2", np.nan) for n in names], 0.38, label="Sobol CV")
if ood:
    ax.bar(x+0.2, [ood.get(n, {}).get("response_r2", np.nan) for n in names], 0.38,
           label="twobound OOD", color="C3")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
ax.axhline(0.9, ls=":", c="k", lw=.8); ax.set_ylim(0, 1.02)
ax.set_ylabel(r"response $R^2$"); ax.set_title(f"feedback-response fidelity ({BACKEND})")
ax.legend(); fig.tight_layout(); plt.show()""")

code(r"""# overlay: emulator (solid) vs truth (dotted) suppression + peaks, a few runs at z_s=1
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
nu = ds.targets["peak_counts"].axes.get("nu")
for r in range(0, ds.n_runs, max(1, ds.n_runs//7)):
    o = em.predict(ds.X_native[r], z_s=1.0); c = plt.cm.viridis(r/ds.n_runs)
    axes[0].semilogx(ell, o["suppression"], color=c, lw=1.2)
    axes[0].semilogx(ell, ds.targets["suppression"].value[r, 1], ":", color=c, lw=1)
    axes[1].plot(nu, o["peak_counts"], color=c, lw=1.2)
    axes[1].plot(nu, ds.targets["peak_counts"].value[r, 1], ":", color=c, lw=1)
axes[0].axhline(1, c="k", lw=.5)
axes[0].set(xlabel=r"$\ell$", ylabel=r"$S(\ell)$ ($z_s=1$)", title="solid=emulator, dotted=truth")
axes[1].set(xlabel=r"$\nu$", ylabel=r"$N_{\rm peak}$", yscale="log")
fig.tight_layout(); plt.show()""")

md(r"""## 6b. Presentation figures + the 1P comparison

`examples/emulator_showcase.py` makes the two paper figures: (1) a **showcase** —
every statistic for one held-out feedback model, emulator vs ray-traced truth, with
the millisecond prediction time annotated; and (2) a **response** figure — the
emulator's feedback response vs truth, parameter by parameter, on a held-out
one-at-a-time suite.

It makes three figures: the **showcase** (all statistics, emulator vs truth, with the
prediction time), the **response** (suppression-dip response vs each parameter), and a
**ratio-to-fiducial** figure for one parameter — its two prior bounds divided by the
fiducial, emulator (lines) vs truth (points), per statistic. Dividing by the fiducial
(`bind/run_0000`) cancels the shared shape and isolates the feedback response; it is
cleanest for the positive power spectra / suppression.

It compares against the **twobound** prior-corner runs by default. The fuller **1P**
set (5 levels per parameter) is the ideal response test, but its lightcones still
need the pipeline — once generated, point `--compare_dir` at it:

```bash
# generate the 1P lightcone statistics (paste -> ray-trace -> reduce):
DESIGN=1P sbatch run_sobol_paste.sh
DESIGN=1P sbatch run_sobol_lux.sh
DESIGN=1P sbatch run_sobol_stats.sh
# then:
python examples/emulator_showcase.py --compare_dir /mnt/home/mlee1/ceph/bind_science/runs/1P
```""")

code(r"""# render the figures here (twobound by default; trains/saves the bundle if needed)
import subprocess, sys
RATIO_PARAM = "BlackHoleRadiativeEfficiency"
subprocess.run([sys.executable, "emulator_showcase.py", "--ratio_param", RATIO_PARAM], check=False)
from IPython.display import Image, display
for f in ("emulator_showcase.png", "emulator_response.png", f"emulator_ratio_{RATIO_PARAM}.png"):
    p = Path("figures_lightcone") / f
    if p.exists():
        display(Image(str(p)))""")

md(r"""## 7. The generative flow backend — *GPU, optional*

The conditional normalizing flow (zuko) models $p(\text{stats}\mid\theta, z_s)$, so it
*samples* mock data vectors — useful for covariances and SBI. It wants a GPU; train it
on the cluster and load the bundle here:

```bash
BACKENDS="flow" sbatch run_emulator_train.sh   # writes lightcone_emulator_flow.pt
```

```python
emf = Emulator.load(BUNDLE_DIR / "lightcone_emulator_flow.pt")
samples = emf.backends["suppression"].sample(
    bind.emulator.params_to_unit(fid[None]), n=512)   # (512, 1, k) mock latents
```

## 8. Full production run

For the final emulator on all 256 runs with all backends, skip the in-notebook fit and
use the SLURM driver (assembles + trains mlp/gp/flow):

```bash
sbatch run_emulator_train.sh        # -> /ceph/bind_sb35/emulator/lightcone_emulator_{mlp,gp,flow}.pt
```

then `Emulator.load(...)` here for analysis. The emulator improves as the suite grows
from 121 → 256 runs.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parent / "lightcone_emulator.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
