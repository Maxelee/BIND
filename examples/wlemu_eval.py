"""Evaluate a trained ``bind.wlemu`` κ emulator against held-out test maps (GPU).

Loads a flow-matching κ checkpoint, draws convergence maps for the **held-out
(validation) SB35 parameter points the model never trained on**, and compares
the generated field distribution to the cached truth maps:

  * angular power spectrum ``C_ell`` (mean ± realization scatter) + emu/truth ratio
  * 1-point pixel PDF
  * map moments (std, skewness, kurtosis) and example map panels

The val split is reproduced with the SAME ``--val_frac`` / ``--seed`` as
``bind.wlemu.train`` (defaults match), so the evaluated parameters are genuinely
out-of-sample.  Outputs PNG figures + a ``metrics.npz`` + a ``summary.json`` to
``--output_dir``.

    python examples/wlemu_eval.py \
        --checkpoint .../wlemu_runs/fm_kappa_1024/best.pt \
        --cache /mnt/home/mlee1/ceph/bind_sb35/wlemu_cache_1024 \
        --output_dir .../wlemu_runs/fm_kappa_1024/eval

Cost scales as ``n_runs · n_z · n_maps · n_steps`` UNet forwards.  The defaults
are a quick look (3 runs × 3 planes × 8 maps × 40 steps ≈ 72 maps); scale up for
publication statistics.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from bind.wlemu import KappaCache, WLEmulator


# ── angular power spectrum — batched torch FFT on the generation device ────────
def _cl_setup(R, fov_deg, device, n_bins=26):
    """Precompute the ℓ grid + radial-bin assignment for ``R×R`` maps of angular
    width ``fov_deg``, as tensors on ``device``.  The maps are already produced on
    the GPU for generation, so the spectrum is just a batched ``torch.fft.rfft2``
    + a scatter-add radial average *there* — milliseconds, not a per-map CPU loop
    (numpy's FFT is single-threaded; on-GPU rfft2 is ~100× faster on a 1024²
    stack).  Flat-sky convention ``C_ℓ = |FFT(κ)|²·dθ⁴ / A_survey`` (radially
    averaged); truth and emulator use the same estimator → exact emu/truth ratio.
    """
    fov = np.deg2rad(fov_deg)
    dtheta = fov / R
    lx = 2 * np.pi * np.fft.fftfreq(R, d=dtheta)
    ly = 2 * np.pi * np.fft.rfftfreq(R, d=dtheta)          # rfft2 → half-plane
    ell2d = np.sqrt(lx[:, None] ** 2 + ly[None, :] ** 2).ravel()
    bins = np.logspace(np.log10(2 * np.pi / fov), np.log10(ell2d.max()), n_bins + 1)
    which = np.digitize(ell2d, bins) - 1                   # DC mode → -1, dropped
    valid = (which >= 0) & (which < n_bins)
    w = which[valid]
    counts = np.bincount(w, minlength=n_bins)
    ell = np.bincount(w, weights=ell2d[valid], minlength=n_bins) / np.maximum(counts, 1)
    keep = np.where(counts > 0)[0]
    dev = torch.device(device)
    return dict(
        device=dev,
        valid=torch.as_tensor(valid, device=dev),
        w=torch.as_tensor(w, dtype=torch.long, device=dev),
        keep=torch.as_tensor(keep, dtype=torch.long, device=dev),
        counts=torch.as_tensor(counts[keep], dtype=torch.float64, device=dev),
        ell=ell[keep], n_bins=n_bins, norm=float(dtheta ** 4 / fov ** 2))


@torch.no_grad()
def _cl_batch(maps, setup):
    """(n, R, R) ndarray → (ell, C_ell[n, n_bins]) via a batched FFT on the device."""
    x = torch.as_tensor(np.asarray(maps), device=setup["device"], dtype=torch.float32)
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    F = torch.fft.rfft2(x)
    P = (F.real ** 2 + F.imag ** 2).reshape(x.shape[0], -1)[:, setup["valid"]]
    P = P.double() * setup["norm"]
    acc = torch.zeros(x.shape[0], setup["n_bins"], dtype=torch.float64,
                      device=setup["device"])
    acc.index_add_(1, setup["w"], P)                       # radial bin (scatter-add)
    cl = (acc[:, setup["keep"]] / setup["counts"]).cpu().numpy()
    return setup["ell"], cl


def _moments(maps):
    """Per-map (std, skew, kurtosis), averaged over maps."""
    x = maps.reshape(maps.shape[0], -1)
    mu = x.mean(1, keepdims=True)
    s = x.std(1)
    d = x - mu
    skew = (d ** 3).mean(1) / np.maximum(s ** 3, 1e-30)
    kurt = (d ** 4).mean(1) / np.maximum(s ** 4, 1e-30) - 3.0
    return dict(std=float(s.mean()), skew=float(skew.mean()), kurt=float(kurt.mean()))


# ── figures ───────────────────────────────────────────────────────────────────
def _fig_maps(truth, emu, z_s, run_id, path, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, truth.shape[0], emu.shape[0])
    vlo, vhi = np.percentile(truth[:n], [1, 99])
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6.2))
    for j in range(n):
        ax[0, j].imshow(truth[j], vmin=vlo, vmax=vhi, cmap="inferno")
        ax[1, j].imshow(emu[j], vmin=vlo, vmax=vhi, cmap="inferno")
        for r in (0, 1):
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
    ax[0, 0].set_ylabel("truth", fontsize=12)
    ax[1, 0].set_ylabel("emulator", fontsize=12)
    fig.suptitle(f"κ maps — held-out run {run_id}, z_s={z_s:g} (shared color scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _fig_cl(ell, cl_truth, cl_emu, zs, path):
    """cl_*: (n_runs, n_z, n_ell). One column per source plane: C_ell + ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_z = cl_truth.shape[1]
    fig, ax = plt.subplots(2, n_z, figsize=(3.4 * n_z, 6.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1]))
    if n_z == 1:
        ax = ax[:, None]
    fac = ell * (ell + 1) / (2 * np.pi)
    for z in range(n_z):
        t, e = cl_truth[:, z], cl_emu[:, z]                 # (n_runs, n_ell)
        tm, em = t.mean(0), e.mean(0)
        tlo, thi = np.percentile(t, [16, 84], axis=0)
        ax[0, z].fill_between(ell, fac * tlo, fac * thi, color="k", alpha=0.18)
        ax[0, z].loglog(ell, fac * tm, "k-", lw=2, label="truth")
        ax[0, z].loglog(ell, fac * em, "C1--", lw=2, label="emulator")
        ax[0, z].set_title(f"z_s = {zs[z]:g}")
        ax[0, z].grid(alpha=0.3, which="both")
        ratio = em / np.where(tm > 0, tm, np.nan)
        ax[1, z].semilogx(ell, ratio, "C1-")
        ax[1, z].axhline(1, color="k", lw=0.8)
        ax[1, z].fill_between(ell, 0.9, 1.1, color="green", alpha=0.08)
        ax[1, z].set_ylim(0.7, 1.3)
        ax[1, z].set_xlabel(r"$\ell$")
        ax[1, z].grid(alpha=0.3, which="both")
    ax[0, 0].legend(fontsize=10)
    ax[0, 0].set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
    ax[1, 0].set_ylabel("emu / truth")
    fig.suptitle("WL convergence power spectrum — emulator vs held-out truth "
                 "(band = 16–84% over runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_pdf(truth, emu, path, fov_deg=5.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = np.percentile(truth, [0.2, 99.8])
    bins = np.linspace(lo, hi, 80)
    ht, _ = np.histogram(truth.ravel(), bins=bins, density=True)
    he, _ = np.histogram(emu.ravel(), bins=bins, density=True)
    c = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.step(c, ht, where="mid", color="k", lw=2, label="truth")
    ax.step(c, he, where="mid", color="C1", lw=2, ls="--", label="emulator")
    ax.set_yscale("log")
    ax.set_xlabel("κ (pixel value)")
    ax.set_ylabel("PDF")
    ax.set_title("1-point convergence PDF (all evaluated runs/planes)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def evaluate(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device visible — running on CPU (slow).", flush=True)

    em = WLEmulator.load(args.checkpoint, device=device, weights=args.weights)
    cache = KappaCache.load(args.cache)
    if em.resolution != cache.resolution:
        raise SystemExit(f"checkpoint resolution {em.resolution} != cache "
                         f"{cache.resolution}; need a matching cache.")

    # Reproduce the trainer's held-out split, then take the first n_runs of it.
    _train_idx, val_idx = cache.run_split(val_frac=args.val_frac, seed=args.seed)
    runs = val_idx[: args.n_runs]
    zs_all = cache.source_redshifts
    z_sel = ([int(z) for z in args.z_idx.split(",")] if args.z_idx
             else list(range(cache.n_z)))
    n_maps = min(args.n_maps, cache.n_real)

    # bf16 autocast only where it actually helps (Ampere+); V100 stays fp32.
    cap = torch.cuda.get_device_capability(0) if device != "cpu" else (0, 0)
    use_amp = (device != "cpu") and cap[0] >= 8

    def gen_ctx():
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp else contextlib.nullcontext())

    print(f"[eval] checkpoint={Path(args.checkpoint).name}  weights={args.weights}  "
          f"device={device} cap={cap} amp={use_amp}", flush=True)
    print(f"[eval] held-out runs={list(map(int, runs))}  z_idx={z_sel} "
          f"({[float(zs_all[z]) for z in z_sel]})  n_maps={n_maps}  "
          f"n_steps={args.n_steps}  cfg={args.cfg_scale}", flush=True)

    setup = _cl_setup(cache.resolution, cache.fov_deg, device)
    print(f"[eval] C_ell: batched torch FFT on {device}, {len(setup['ell'])} ℓ-bins "
          f"(ℓ={setup['ell'][0]:.0f}–{setup['ell'][-1]:.0f})", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ell_ref = None
    CL_T, CL_E = [], []          # (n_runs, n_z, n_ell)
    PDF_T, PDF_E = [], []        # pooled pixel samples (subsampled) for the PDF
    mom_rows = []
    t0 = time.time()
    for ri, run in enumerate(runs):
        run = int(run)
        cl_t_z, cl_e_z = [], []
        for z in z_sel:
            truth = np.asarray(cache.kappa[run, :n_maps, z], dtype=np.float32)
            with gen_ctx():
                emu = em.generate(
                    cache.params_unit[run], z_s=float(zs_all[z]), n=n_maps,
                    n_steps=args.n_steps, cfg_scale=args.cfg_scale,
                    seed=args.gen_seed, batch_size=args.gen_batch,
                    params_are_unit=True)
            emu = np.asarray(emu, dtype=np.float32)

            ell, clt = _cl_batch(truth, setup)
            _, cle = _cl_batch(emu, setup)
            ell_ref = ell
            cl_t_z.append(clt.mean(0))
            cl_e_z.append(cle.mean(0))
            if ri == 0 and z == z_sel[0]:
                _fig_maps(truth, emu, float(zs_all[z]), run, out / "maps_example.png")
            # subsample pixels for the pooled PDF (keep memory bounded)
            PDF_T.append(truth.ravel()[:: max(1, truth.size // 200_000)])
            PDF_E.append(emu.ravel()[:: max(1, emu.size // 200_000)])
            mom_rows.append((run, float(zs_all[z]), _moments(truth), _moments(emu)))
            frac = np.nanmedian(np.abs(cle.mean(0) / clt.mean(0) - 1))
            print(f"  run {run:4d}  z_s={float(zs_all[z]):4.2f}  "
                  f"median|ΔC_ell/C_ell|={frac:6.3f}  "
                  f"σ_truth={truth.std():.3e} σ_emu={emu.std():.3e}  "
                  f"[{time.time() - t0:5.0f}s]", flush=True)
        CL_T.append(np.asarray(cl_t_z))
        CL_E.append(np.asarray(cl_e_z))

    CL_T, CL_E = np.asarray(CL_T), np.asarray(CL_E)     # (n_runs, n_z, n_ell)
    PDF_T = np.concatenate(PDF_T)
    PDF_E = np.concatenate(PDF_E)

    # figures
    _fig_cl(ell_ref, CL_T, CL_E, [float(zs_all[z]) for z in z_sel], out / "cl_compare.png")
    _fig_pdf(PDF_T, PDF_E, out / "pdf_compare.png")

    # per-z aggregate Cl fractional error (median over ell, mean over runs)
    per_z = {}
    for k, z in enumerate(z_sel):
        r = CL_E[:, k] / np.where(CL_T[:, k] > 0, CL_T[:, k], np.nan)
        per_z[f"z_{float(zs_all[z]):g}"] = dict(
            median_frac_cl_err=float(np.nanmedian(np.abs(r - 1))),
            mean_frac_cl_err=float(np.nanmean(np.abs(r - 1))))
    overall = float(np.nanmedian(np.abs(CL_E / np.where(CL_T > 0, CL_T, np.nan) - 1)))

    summary = dict(
        checkpoint=str(args.checkpoint), weights=args.weights,
        resolution=int(em.resolution), fov_deg=float(cache.fov_deg),
        n_runs=int(len(runs)), held_out_runs=list(map(int, runs)),
        z_source=[float(zs_all[z]) for z in z_sel], n_maps=int(n_maps),
        n_steps=int(args.n_steps), cfg_scale=float(args.cfg_scale),
        cl_backend="flatsky_fft", overall_median_frac_cl_err=overall, per_z=per_z,
        moments=[dict(run=int(r), z_s=z, truth=t, emu=e) for r, z, t, e in mom_rows],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out / "metrics.npz", ell=ell_ref, cl_truth=CL_T, cl_emu=CL_E,
             z_source=np.array([float(zs_all[z]) for z in z_sel]),
             held_out_runs=np.array(list(map(int, runs))))

    print(f"\n[eval] DONE in {time.time() - t0:.0f}s → {out}", flush=True)
    print(f"[eval] overall median |ΔC_ell/C_ell| = {overall:.3f}", flush=True)
    for kz, v in per_z.items():
        print(f"       {kz:>10s}: median {v['median_frac_cl_err']:.3f}", flush=True)
    print(f"[eval] figures: maps_example.png  cl_compare.png  pdf_compare.png", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="trained .pt (best.pt/last.pt)")
    p.add_argument("--cache", required=True, help="KappaCache dir (truth + split)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--weights", choices=["ema", "model"], default="ema")
    p.add_argument("--val_frac", type=float, default=0.125, help="match training")
    p.add_argument("--seed", type=int, default=0, help="match training")
    p.add_argument("--n_runs", type=int, default=3, help="held-out param points to eval")
    p.add_argument("--n_maps", type=int, default=8, help="maps per (run, z_s)")
    p.add_argument("--z_idx", type=str, default="0,2,4",
                   help="source planes to eval; '' = all 5 (default spans z=0.5,1.5,2.44)")
    p.add_argument("--n_steps", type=int, default=40, help="Euler ODE steps")
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--gen_batch", type=int, default=8, help="sampling micro-batch")
    p.add_argument("--gen_seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
