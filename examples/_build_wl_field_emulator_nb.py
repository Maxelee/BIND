"""Generate examples/wl_field_emulator.ipynb — the field-level WL κ emulator demo.

One runnable notebook for the whole `bind.wlemu` workflow: build (a slice of) the
pooled κ cache, train a θ→κ conditional flow-matching model, **generate fresh
convergence maps for any parameter point**, and validate them against held-out
ray-traced truth (power spectrum, PDF, peak counts) + show the baryon-feedback
response and classifier-free guidance.

A `QUICK` toggle trains a tiny model in-notebook on a handful of runs (CPU /
local GPU, minutes) so the notebook runs end-to-end without the SLURM jobs; the
full run loads a checkpoint from run_wlemu_train.sh.  Engine: src/bind/wlemu/.
Run: python examples/_build_wl_field_emulator_nb.py
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


md(r"""# `bind.wlemu`: a **field-level** weak-lensing emulator (θ → κ map)

Where `bind.emulator` predicts *summary statistics* of the lightcone, `bind.wlemu`
generates the **convergence map itself**: feed in the 30 SB35 astro/feedback
parameters $\theta$ (+ a source redshift $z_s$) and draw fresh $\kappa$ maps in
milliseconds — with cosmic variance, not just the mean.

**Method — Conditional Flow Matching (θ → κ), chosen over latent-CFM / score /
wavelet-flow.** BIND itself is a parameter-conditioned flow-matching U-Net, so the
conditioning machinery (`AdaGroupNorm` on a param embedding + sinusoidal time +
redshift embedding) is already validated here; sampling is fast (~20–50 NFE),
which matters because the whole point of a field-level emulator is to draw *many*
maps; and we avoid the latent-bottleneck compression loss that would wash out the
small-scale baryon response we care about. The model is identical in spirit to the
painter — only the plumbing differs (single channel, **no DMO input** to
concatenate, $\theta\to\kappa$ from pure noise).

**Data.** 256-point SB35 Sobol suite (`bind_sb35/runs/run_NNNN/kappa_maps.npz`):
50 realisations × 5 source planes ($z_s\in\{0.5,1,1.5,2,2.44\}$) × $1024^2$ each,
$5^\circ$ FoV, fixed TNG300 cosmology. We split **by parameter point** so validation
tests the real task — generalising to unseen feedback.

**Pipeline** (the two heavy steps run on SLURM; this notebook can do a tiny version
inline):
```
bind-wlemu-cache --resolution 1024 --out <cache>     # run_wlemu_cache.sh  (CPU)
bind-wlemu-train --cache <cache> --run_name fm_kappa  # run_wlemu_train.sh  (GPU/DDP)
```
""")

code(r"""%load_ext autoreload
%autoreload 2
import warnings; warnings.filterwarnings("ignore")
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

import bind
from bind.wlemu import build_cache, KappaCache, WLEmulator
from bind.wlemu.data import ASTRO_PARAM_NAMES, SOURCE_REDSHIFTS

# ── knobs ─────────────────────────────────────────────────────────────────────
QUICK       = True       # tiny in-notebook cache+train (minutes). False → load a trained ckpt.
RESOLUTION  = 128        # QUICK demo resolution (128/256). Full runs use 1024.
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Full-run paths (used when QUICK=False) — point these at the SLURM outputs:
FULL_CACHE  = Path(f"/mnt/home/mlee1/ceph/bind_sb35/wlemu_cache_1024")
FULL_CKPT   = Path("/mnt/home/mlee1/ceph/bind_sb35/wlemu_runs/fm_kappa_1024/best.pt")

# QUICK demo workspace (kept off the science tree):
DEMO_DIR    = Path("/tmp") / "wlemu_demo"
DEMO_CACHE  = DEMO_DIR / f"cache_{RESOLUTION}"
DEMO_RUN    = DEMO_DIR / "runs"
print("QUICK =", QUICK, "| device =", DEVICE, "| resolution =", RESOLUTION)""")

md(r"""## 1 · Build a (slice of the) κ cache

The 256 GB of compressed npz is pooled **once** into a single memmap
(`kappa.npy` + `meta.npz` + `norm.npz`), so training isn't I/O-bound. We store the
*raw* pooled κ and apply the normalisation on the fly, so it can be re-tuned
without rebuilding.

**Normalisation** (per source redshift — κ variance grows strongly with $z_s$ and
the tail is heavy, max $\approx 70\sigma$):
$$ y = \operatorname{arcsinh}(\kappa/\lambda_z), \qquad x = (y-\mu_z)/\sigma_z $$
which tames the halo tail while staying ~linear near 0. Inverse:
$\kappa=\lambda_z\sinh(\sigma_z x + \mu_z)$.

`QUICK` builds a tiny cache from a handful of runs (~1–2 min); the full cache is
made by `run_wlemu_cache.sh`.""")

code(r"""if QUICK:
    if not (DEMO_CACHE / "kappa.npy").exists():
        build_cache(DEMO_CACHE, resolution=RESOLUTION, max_runs=16, n_workers=4)
    cache = KappaCache.load(DEMO_CACHE)
else:
    cache = KappaCache.load(FULL_CACHE)

print("cache:", cache.kappa.shape, "(runs, real, z, R, R)")
print("resolution:", cache.resolution, "| FoV:", cache.fov_deg, "deg")
print("source redshifts:", cache.source_redshifts)
print("arcsinh λ_z:", np.array2string(cache.norm.lam, precision=4))
train_idx, val_idx = cache.run_split(val_frac=0.2, seed=0)
print(f"{len(train_idx)} train runs / {len(val_idx)} held-out val runs")""")

md(r"""## 2 · Train (or load) the θ→κ flow-matching model

`QUICK` trains a small model for a few hundred steps just to exercise the whole
path (it will **not** be science-converged). For real results, train with
`run_wlemu_train.sh` and set `QUICK=False`.""")

code(r"""if QUICK:
    from bind.wlemu.train import main as train_main
    ckpt = DEMO_RUN / "demo" / "best.pt"
    if not ckpt.exists():
        t0 = time.time()
        train_main([
            "--cache", str(DEMO_CACHE), "--run_name", "demo", "--output_dir", str(DEMO_RUN),
            "--base_ch", "64", "--n_blocks", "2", "--emb_dim", "128",
            "--batch_size", "16", "--epochs", "8", "--warmup_steps", "50",
            "--lr", "3e-4", "--val_every", "100", "--num_workers", "4", "--log_every", "50",
        ])
        print(f"trained in {time.time()-t0:.0f}s")
else:
    ckpt = FULL_CKPT

em = WLEmulator.load(ckpt, device=DEVICE)
print("loaded emulator:", em.resolution, "px |", len(em.param_names), "params |", em.device)""")

md(r"""## 3 · Generate maps for a parameter point

`em.generate(params, z_s, n)` draws `n` independent κ maps for one feedback vector
and source redshift (raw κ, denormalised). `params` is a native SB35 vector —
`bind.fiducial_params()` is the TNG300 fiducial.""")

code(r"""theta = bind.fiducial_params()
z_s = 1.0
t0 = time.time()
maps = em.generate(theta, z_s=z_s, n=8, n_steps=50, seed=0)
dt = time.time() - t0
print(f"generated {maps.shape} in {dt:.2f}s  ({1e3*dt/len(maps):.0f} ms/map)")

fig, ax = plt.subplots(2, 4, figsize=(14, 7))
vlim = np.percentile(np.abs(maps), 99)
for a, m in zip(ax.ravel(), maps):
    a.imshow(m, cmap="inferno", vmin=-vlim, vmax=vlim,
             extent=[0, cache.fov_deg, 0, cache.fov_deg])
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"Emulated κ — fiducial TNG300, $z_s$={z_s} ({len(maps)} realisations)")
plt.tight_layout(); plt.show()""")

md(r"""### Emulated vs. ray-traced truth — by eye

Pull the held-out truth maps (pooled, same resolution) for one validation run from
the cache, and put an emulated draw at the *same* parameters next to it.""")

code(r"""vrun = int(val_idx[0])   # a held-out parameter point
zk = 1                   # z_s = 1.0 plane
truth = np.asarray(cache.kappa[vrun, :, zk], dtype=np.float32)   # (n_real, R, R) raw κ
em_v = em.generate(cache.params_unit[vrun], z_s=float(cache.source_redshifts[zk]),
                   n=8, n_steps=50, params_are_unit=True, seed=1)

fig, ax = plt.subplots(2, 4, figsize=(14, 7))
vlim = np.percentile(np.abs(truth), 99)
for j in range(4):
    ax[0, j].imshow(truth[j], cmap="inferno", vmin=-vlim, vmax=vlim)
    ax[1, j].imshow(em_v[j],  cmap="inferno", vmin=-vlim, vmax=vlim)
    for r in (0, 1): ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
ax[0, 0].set_ylabel("truth", fontsize=13); ax[1, 0].set_ylabel("emulated", fontsize=13)
fig.suptitle(f"Held-out run {cache.run_ids[vrun]} — truth (top) vs emulated (bottom), $z_s$=1")
plt.tight_layout(); plt.show()""")

md(r"""## 4 · Statistical validation (the real test)

A field-level emulator must reproduce the *distribution* of maps, not a single
image. We compare emulated vs. held-out truth on:
* the **angular power spectrum** $C_\ell$ (with the cosmic-variance band from the
  50 realisations),
* the **one-point PDF** of κ,
* **peak counts** (local maxima vs. height) — the canonical non-Gaussian probe.

All three are computed at the cache resolution, so emulated and (pooled) truth are
apples-to-apples.""")

code(r"""def power_spectrum(maps, fov_deg, nbins=16):
    # Radially-binned C_ell of (..., R, R) maps. Returns (ell, Cl_mean, Cl_std).
    maps = np.atleast_3d(maps).reshape(-1, maps.shape[-2], maps.shape[-1])
    R = maps.shape[-1]
    omega = np.deg2rad(fov_deg) ** 2                       # survey solid angle (rad²)
    kf = np.fft.fftfreq(R) * R
    kx, ky = np.meshgrid(kf, kf)
    ell = (2 * np.pi / np.deg2rad(fov_deg)) * np.sqrt(kx**2 + ky**2)
    ell = np.fft.fftshift(ell)
    fk = np.fft.fftshift(np.fft.fft2(maps), axes=(-2, -1)) * (omega / R**2)
    cl2d = (np.abs(fk) ** 2) / omega                       # (n, R, R)
    lo, hi = ell[ell > 0].min(), ell.max() / np.sqrt(2)
    edges = np.geomspace(lo, hi, nbins + 1)
    idx = np.digitize(ell.ravel(), edges)
    cl = np.stack([m.ravel() for m in cl2d])
    out = np.array([[c[idx == b].mean() for b in range(1, nbins + 1)] for c in cl])
    cen = np.sqrt(edges[:-1] * edges[1:])
    return cen, out.mean(0), out.std(0)

def kappa_pdf(maps, bins):
    h = np.stack([np.histogram(m.ravel(), bins=bins, density=True)[0] for m in
                  np.atleast_3d(maps).reshape(-1, maps.shape[-2], maps.shape[-1])])
    return 0.5 * (bins[:-1] + bins[1:]), h.mean(0), h.std(0)

def peak_counts(maps, bins):
    from scipy.ndimage import maximum_filter
    cnt = []
    for m in np.atleast_3d(maps).reshape(-1, maps.shape[-2], maps.shape[-1]):
        pk = (m == maximum_filter(m, size=3)) & (m > m.mean())
        cnt.append(np.histogram(m[pk], bins=bins)[0])
    cnt = np.stack(cnt)
    return 0.5 * (bins[:-1] + bins[1:]), cnt.mean(0), cnt.std(0)

# emulate a matched ensemble at the held-out point
em_big = em.generate(cache.params_unit[vrun], z_s=float(cache.source_redshifts[zk]),
                     n=truth.shape[0], n_steps=50, params_are_unit=True, seed=2)
print("ensembles:", truth.shape, em_big.shape)""")

code(r"""fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

# C_ell
ell, ct, cs = power_spectrum(truth, cache.fov_deg)
_,   ce, es = power_spectrum(em_big, cache.fov_deg)
ax[0].fill_between(ell, ell*(ct-cs), ell*(ct+cs), alpha=.25, color="k", label="truth ±cv")
ax[0].plot(ell, ell*ct, "k-")
ax[0].fill_between(ell, ell*(ce-es), ell*(ce+es), alpha=.25, color="C3")
ax[0].plot(ell, ell*ce, "C3--", label="emulated")
ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\ell$"); ax[0].set_ylabel(r"$\ell\,C_\ell^{\kappa\kappa}$")
ax[0].set_title("power spectrum"); ax[0].legend()

# PDF
b = np.linspace(np.percentile(truth, .1), np.percentile(truth, 99.9), 60)
x, pt, ps = kappa_pdf(truth, b); _, pe, pse = kappa_pdf(em_big, b)
ax[1].fill_between(x, pt-ps, pt+ps, alpha=.25, color="k"); ax[1].plot(x, pt, "k-", label="truth")
ax[1].fill_between(x, pe-pse, pe+pse, alpha=.25, color="C3"); ax[1].plot(x, pe, "C3--", label="emulated")
ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\kappa$"); ax[1].set_title("one-point PDF"); ax[1].legend()

# peaks
pb = np.linspace(np.percentile(truth, 50), np.percentile(truth, 99.9), 18)
x, kt, ks = peak_counts(truth, pb); _, ke, kse = peak_counts(em_big, pb)
ax[2].fill_between(x, kt-ks, kt+ks, alpha=.25, color="k"); ax[2].plot(x, kt, "k-", label="truth")
ax[2].fill_between(x, ke-kse, ke+kse, alpha=.25, color="C3"); ax[2].plot(x, ke, "C3--", label="emulated")
ax[2].set_xlabel(r"$\kappa_{\rm peak}$"); ax[2].set_ylabel("counts / map"); ax[2].set_title("peak counts"); ax[2].legend()

plt.tight_layout(); plt.show()
print("NB: in QUICK mode the model is under-trained — expect only qualitative agreement.")""")

md(r"""## 5 · Baryon-feedback response

The science payoff: does the emulated map *respond* to feedback? We sweep one
parameter across its SB35 range (holding the rest at fiducial) and measure the WL
**suppression** $S(\ell)=C_\ell(\theta)/C_\ell(\theta_{\rm fid})$. We also show the
mean residual map $\langle\kappa(\theta_{\rm hi})\rangle-\langle\kappa(\theta_{\rm
lo})\rangle$.""")

code(r"""from bind.params import PARAM_NAMES, PARAM_MIN, PARAM_MAX

PARAM = "BlackHoleFeedbackFactor"   # try also "WindEnergyIn1e51erg", "VariableWindVelFactor"
pidx = PARAM_NAMES.index(PARAM)
lo, hi = PARAM_MIN[pidx], PARAM_MAX[pidx]

def at(value, n=24):
    th = bind.fiducial_params().copy(); th[pidx] = value
    return em.generate(th, z_s=1.0, n=n, n_steps=50, seed=7)

m_fid = at(bind.fiducial_params()[pidx]); m_lo = at(lo); m_hi = at(hi)
ell, cfid, _ = power_spectrum(m_fid, cache.fov_deg)
_,   clo,  _ = power_spectrum(m_lo,  cache.fov_deg)
_,   chi,  _ = power_spectrum(m_hi,  cache.fov_deg)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].axhline(1, color="k", lw=.8)
ax[0].plot(ell, clo/cfid, "C0-o", ms=3, label=f"{PARAM}={lo:.2g} (low)")
ax[0].plot(ell, chi/cfid, "C3-o", ms=3, label=f"{PARAM}={hi:.2g} (high)")
ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\ell$")
ax[0].set_ylabel(r"$S(\ell)=C_\ell/C_\ell^{\rm fid}$"); ax[0].set_title("WL suppression vs feedback"); ax[0].legend()

resid = m_hi.mean(0) - m_lo.mean(0)
v = np.percentile(np.abs(resid), 99)
im = ax[1].imshow(resid, cmap="RdBu_r", vmin=-v, vmax=v,
                  extent=[0, cache.fov_deg, 0, cache.fov_deg])
ax[1].set_title(r"$\langle\kappa_{\rm hi}\rangle-\langle\kappa_{\rm lo}\rangle$")
plt.colorbar(im, ax=ax[1], fraction=.046); plt.tight_layout(); plt.show()""")

md(r"""## 6 · Classifier-free guidance

The feedback response is intrinsically weak, so we trained with parameter dropout
(`cfg_dropout=0.1`). At sample time, `cfg_scale > 1` extrapolates away from the
param-dropped (unconditional) velocity, **amplifying** the parameter dependence —
useful for visualising response, at the cost of slightly inflated variance.""")

code(r"""def gen_at(value, cfg, n=16):
    th = bind.fiducial_params().copy(); th[pidx] = value
    return em.generate(th, z_s=1.0, n=n, n_steps=50, cfg_scale=cfg, seed=7)

for g in (1.0, 2.0, 4.0):
    _, chi, _ = power_spectrum(gen_at(hi, g), cache.fov_deg)
    _, clo, _ = power_spectrum(gen_at(lo, g), cache.fov_deg)
    plt.plot(ell, chi/clo, "-o", ms=3, label=f"cfg={g}")
plt.axhline(1, color="k", lw=.8); plt.xscale("log")
plt.xlabel(r"$\ell$"); plt.ylabel(r"$C_\ell(\rm hi)/C_\ell(\rm lo)$")
plt.title("guidance amplifies the feedback contrast"); plt.legend(); plt.show()""")

md(r"""## Recap & next steps

* **What this is** — a CFM generator $\theta\to\kappa$ that draws ray-traced-quality
  convergence maps for any of the 30 SB35 feedback parameters + $z_s$, in
  milliseconds, with cosmic variance.
* **Validation** — by parameter point (held-out runs), on $C_\ell$ / PDF / peak
  counts, with feedback-response and CFG diagnostics.
* **Scale up** — `run_wlemu_cache.sh` (1024²) then `run_wlemu_train.sh` (4× A100,
  DDP, grad-checkpointed) → `WLEmulator.load(best.pt)`; the cells above run
  unchanged with `QUICK=False`.
* **Caveats** — only 256 design points in 30-D, so parameter *interpolation* is the
  hard part (the 50 realisations only teach the noise model); the documented
  high-$\ell$ CIC-aliasing upturn lives in the maps, so trust feedback *ratios*
  over absolute high-$\ell$ power. Possible v2: latent-CFM for speed, or a wavelet
  multiscale head for explicit scale control.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parent / "wl_field_emulator.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
