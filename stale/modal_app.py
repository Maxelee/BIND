#!/usr/bin/env python3
"""
modal_app.py  —  Serverless GPU backend for the BIND Halo Explorer.

Deploys the fm_base flow-matching model to Modal.com so anyone can use the
interactive halo explorer without requiring an HPC node to stay running.
The HTML + sliders are served directly from this app; images are generated
on-demand (~2s per frame on a T4 GPU, scales to zero when idle).

─── One-time setup ──────────────────────────────────────────────────────────

1. Install Modal and authenticate (once per machine):

     pip install modal
     python -m modal setup          # opens browser to link your account

2. Create the data volume:

     modal volume create halo-explorer-data

3. Upload all required files (run from the HPC — see upload_to_modal.sh):

     modal volume put halo-explorer-data \\
         /mnt/home/mlee1/ceph/fm_runs/fm_base/checkpoints/last.ckpt \\
         last.ckpt

     ... (see upload_to_modal.sh for the full list)

─── Deploy ──────────────────────────────────────────────────────────────────

     cd /mnt/home/mlee1/vdm_bind2
     modal deploy modal_app.py

   Modal prints the public URL, e.g.
     https://mlee1--halo-explorer-web.modal.run

─── Tear down (stops billing) ───────────────────────────────────────────────

     modal app stop halo-explorer

─── Local dev / testing ─────────────────────────────────────────────────────

     modal serve modal_app.py      # hot-reload on file save

"""

from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal infrastructure
# ─────────────────────────────────────────────────────────────────────────────

# Persistent volume that stores the model checkpoint + halo data.
# Upload files once with:  modal volume put halo-explorer-data <local> <remote>
DATA_VOLUME = modal.Volume.from_name("halo-explorer-data", create_if_missing=True)
DATA_DIR    = Path("/vol")

# SB35 CSV is read at module-import time in data.py, so it must be baked
# into the image (not loaded from the Volume at runtime).
_SB35_CSV_LOCAL = "/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv"
# Remote path must match exactly what data.py hardcodes.
_SB35_CSV_REMOTE = _SB35_CSV_LOCAL

# Container image: PyTorch + our source modules
image = (
    modal.Image.debian_slim(python_version="3.11")
    # PyTorch with CUDA 12.1 wheels
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Everything else
    .pip_install(
        "lightning==2.5.1",
        "torch_ema",
        "numpy",
        "scipy",
        "matplotlib",
        "pillow",
        "pandas",
        "starlette",
        "uvicorn[standard]",
        "pydantic>=2",
        "tqdm",
    )
    # Embed the SB35 CSV at the exact path data.py reads at import time.
    # All other large files (checkpoint, cutouts, etc.) come from the Volume.
    .add_local_file(_SB35_CSV_LOCAL, _SB35_CSV_REMOTE)
    # Mount local source files into the container
    .add_local_python_source("data", "model", "metrics", "train", "generate_sobol_explorer")
)

app = modal.App("halo-explorer", image=image)


# ─────────────────────────────────────────────────────────────────────────────
# Halo server class
# ─────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu="T4",
    volumes={str(DATA_DIR): DATA_VOLUME},
    scaledown_window=600,   # keep container warm 10 min after last request
    timeout=300,            # max seconds for a single inference call
)
class HaloServer:
    """
    One instance = one warm container on a T4 GPU.
    Model loads once at startup (~30s); subsequent requests are ~2s each.
    """

    @modal.enter()
    def startup(self) -> None:
        """Called once when the container starts.  Loads model + data."""
        import numpy as np
        import torch

        # ── Override HPC-specific path constants BEFORE any gse helper uses them
        import generate_sobol_explorer as gse
        gse.CUTOUTS_PATH = DATA_DIR / "halo_cutouts.npz"
        gse.CATALOG_PATH = DATA_DIR / "halo_catalog.npz"
        gse.CV_PARAM_FILE = DATA_DIR / "CosmoAstroSeed_IllustrisTNG_L50n512_CV.txt"
        gse.SB35_CSV      = DATA_DIR / "SB35_param_minmax.csv"
        gse.RUN_DIR       = DATA_DIR
        gse.CKPT_PATH     = DATA_DIR / "last.ckpt"

        from generate_sobol_explorer import (
            N_ASTRO, CHANNEL_NAMES, CHANNEL_CMAPS,
            load_param_meta, load_cv12_params,
            fiducial_slider_positions, sobol_unit_to_physical,
            build_full_params, normalize_params_for_model,
            is_legacy_param_norm,
        )
        from data import NormStats, log_transform
        from train import FlowMatchingLit

        self.N_ASTRO        = N_ASTRO
        self.CHANNEL_NAMES  = CHANNEL_NAMES
        self.CHANNEL_CMAPS  = CHANNEL_CMAPS

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[startup] device = {device}")

        param_meta  = load_param_meta()
        cv12_params = load_cv12_params()
        legacy      = is_legacy_param_norm(DATA_DIR / "norm_stats.npz")
        use_log     = not legacy
        print(f"[startup] param norm: {'linear (legacy)' if legacy else 'log10 for flagged params'}")

        norm_stats = NormStats.load(DATA_DIR / "norm_stats.npz")
        if legacy:
            norm_stats.param_log_flag = np.zeros(35, dtype=np.int32)

        print("[startup] loading checkpoint …")
        fm_lit = FlowMatchingLit.load_from_checkpoint(
            str(DATA_DIR / "last.ckpt"), map_location=device
        )
        fm_lit.eval().to(device)
        if hasattr(fm_lit, 'ema'):
            fm_lit.ema.copy_to(fm_lit.unet.parameters())
        print("[startup] checkpoint loaded")

        # Fixed halo condition (most massive halo in CV_12)
        cutouts     = np.load(DATA_DIR / "halo_cutouts.npz")
        catalog     = np.load(DATA_DIR / "halo_catalog.npz")
        condition   = cutouts["condition"][0]
        large_scale = cutouts["large_scale"][0]
        halo_mass   = catalog["masses"][0]
        print(f"[startup] halo mass = {halo_mass:.3e} M_sun/h")

        c  = log_transform(condition)[None, None].astype(np.float32)
        c  = (c - norm_stats.cond_mean) / (norm_stats.cond_std + 1e-8)
        ls = log_transform(large_scale)[None].astype(np.float32)
        ls = (ls - norm_stats.ls_mean[None, :, None, None]) / (
             norm_stats.ls_std[None, :, None, None] + 1e-8)

        fiducial_pos = fiducial_slider_positions(param_meta, use_log=use_log)

        self.device       = device
        self.fm           = fm_lit.fm
        self.norm_stats   = norm_stats
        self.param_meta   = param_meta
        self.cv12_params  = cv12_params
        self.use_log      = use_log
        self.fiducial_pos = fiducial_pos
        self.cond_t       = torch.from_numpy(c.astype(np.float32)).to(device)
        self.ls_t         = torch.from_numpy(ls.astype(np.float32)).to(device)

        # ── Compute stable color range from a small reference batch
        print("[startup] calibrating color scale …")
        ref_u = np.tile(fiducial_pos, (8, 1)).astype(np.float32)
        rng   = np.random.default_rng(0)
        for i in range(1, 8):
            j = rng.integers(N_ASTRO)
            ref_u[i, j] = rng.uniform(0.1, 0.9)

        ref_gen = np.stack([self._infer(ref_u[i], n_steps=10) for i in range(len(ref_u))])
        log_ref = np.log10(1.0 + ref_gen)
        self.vmin = np.percentile(log_ref, 0.5,  axis=(0, 2, 3))
        self.vmax = np.percentile(log_ref, 99.5, axis=(0, 2, 3))
        print(f"[startup] vmin={self.vmin}, vmax={self.vmax}")

        # ── Run fiducial inference for difference maps
        print("[startup] running fiducial inference …")
        self.fiducial_gen = self._infer(fiducial_pos, n_steps=20)   # (3, H, W)
        # Smooth before differencing: removes per-pixel occupancy noise, especially in stars
        from scipy.ndimage import gaussian_filter
        DIFF_SIGMA = [1.0, 1.0, 2.0]   # ch0=DM, ch1=Gas, ch2=Stars
        self.diff_sigma = DIFF_SIGMA
        def _smooth_log(field3d):  # field3d: (3, H, W) physical
            out = np.empty_like(field3d)
            for c in range(3):
                out[c] = gaussian_filter(np.log10(1.0 + field3d[c]), sigma=DIFF_SIGMA[c])
            return out
        log_fid_sm  = _smooth_log(self.fiducial_gen)          # (3, H, W)
        log_all_sm  = np.stack([_smooth_log(ref_gen[i]) for i in range(len(ref_gen))])  # (8, 3, H, W)
        abs_diffs   = np.abs(log_all_sm - log_fid_sm[None])  # (8, 3, H, W)
        self.diff_vmax = np.percentile(abs_diffs, 98, axis=(0, 2, 3))  # (3,)
        self.diff_vmax = np.maximum(self.diff_vmax, 0.02)              # floor at 0.02 dex
        print(f"[startup] diff_vmax={self.diff_vmax}")
        print("[startup] ready")

    # ─────────────────────────────────────────────────────────────────────────
    # Inference helpers (called synchronously from ASGI route handlers)
    # ─────────────────────────────────────────────────────────────────────────

    def _infer(self, params_u, n_steps: int = 20):
        """Run a single inference.  Returns (3, H, W) float32 in physical units."""
        import numpy as np
        import torch
        from generate_sobol_explorer import (
            sobol_unit_to_physical, build_full_params,
            normalize_params_for_model, N_ASTRO,
        )

        params_u = np.asarray(params_u, dtype=np.float32).reshape(1, N_ASTRO)
        astro    = sobol_unit_to_physical(params_u, self.param_meta, use_log=self.use_log)
        full     = build_full_params(astro, self.cv12_params)
        pnorm    = normalize_params_for_model(full, self.norm_stats)
        params_t = torch.from_numpy(pnorm.astype(np.float32)).to(self.device)

        with torch.no_grad():
            torch.manual_seed(42)   # fixed seed → same noise every call → diff row = 0 at fiducial
            gen = self.fm.sample(self.cond_t, self.ls_t, params_t, n_steps=n_steps)

        gen_np = gen.float().cpu().numpy()
        ns = self.norm_stats
        for ch in range(3):
            gen_np[:, ch] = gen_np[:, ch] * ns.target_std[ch] + ns.target_mean[ch]
            gen_np[:, ch] = 10.0 ** gen_np[:, ch] - 1.0
        return np.clip(gen_np[0], 0.0, None)

    def _render(self, gen, size_px: int = 220) -> list[str]:
        """(3, H, W) float32 → list of 3 base64 JPEG strings."""
        import numpy as np
        import matplotlib.cm as mcm
        from PIL import Image

        result = []
        for ch in range(3):
            field_log  = np.log10(1.0 + gen[ch])
            dv         = max(float(self.vmax[ch]) - float(self.vmin[ch]), 1e-10)
            field_norm = np.clip((field_log - self.vmin[ch]) / dv, 0.0, 1.0)
            rgba       = (mcm.get_cmap(self.CHANNEL_CMAPS[ch])(field_norm) * 255).astype(np.uint8)
            img        = Image.fromarray(rgba, "RGBA").convert("RGB")
            img        = img.resize((size_px, size_px), Image.BILINEAR)
            buf        = BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            buf.seek(0)
            result.append(base64.b64encode(buf.read()).decode("ascii"))
        return result

    def _render_diff(self, gen, size_px: int = 220) -> list[str]:
        """(3, H, W) float32 → list of 3 base64 JPEG difference images (smoothed gen - fiducial)."""
        import numpy as np
        import matplotlib.cm as mcm
        from PIL import Image
        from scipy.ndimage import gaussian_filter

        result = []
        for ch in range(3):
            # Smooth in log-space to suppress per-pixel occupancy noise (critical for stars)
            log_gen = gaussian_filter(np.log10(1.0 + gen[ch]),             sigma=self.diff_sigma[ch])
            log_fid = gaussian_filter(np.log10(1.0 + self.fiducial_gen[ch]), sigma=self.diff_sigma[ch])
            diff    = log_gen - log_fid          # positive = more mass, negative = less
            vd      = float(self.diff_vmax[ch])
            diff_norm = np.clip(diff / vd * 0.5 + 0.5, 0.0, 1.0)   # centre = 0.5
            rgba    = (mcm.get_cmap("RdBu_r")(diff_norm) * 255).astype(np.uint8)
            img     = Image.fromarray(rgba, "RGBA").convert("RGB")
            img     = img.resize((size_px, size_px), Image.BILINEAR)
            buf     = BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            buf.seek(0)
            result.append(base64.b64encode(buf.read()).decode("ascii"))
        return result

    def _compute_stats(self, gen) -> dict:
        """Compute radial profile, histogram, and power spectrum for gen vs fiducial."""
        import numpy as np

        PIX_MPC = 0.048828125   # Mpc/h per pixel
        N_BINS  = 40
        fid = self.fiducial_gen  # (3, H, W)
        H, W = gen.shape[1], gen.shape[2]
        cy, cx = H / 2.0, W / 2.0

        # Radius map in Mpc/h
        ys, xs = np.ogrid[:H, :W]
        r_map_px  = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).ravel()
        r_map_mpc = r_map_px * PIX_MPC
        r_max_mpc = min(cx, cy) * PIX_MPC
        r_edges   = np.linspace(0, r_max_mpc, N_BINS + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

        # k map in h/Mpc (physical)
        ky = np.fft.fftshift(np.fft.fftfreq(H, d=PIX_MPC))
        kx = np.fft.fftshift(np.fft.fftfreq(W, d=PIX_MPC))
        KX, KY = np.meshgrid(kx, ky)
        K_map   = np.sqrt(KX ** 2 + KY ** 2).ravel()
        k_max   = K_map.max() * 0.7
        k_edges = np.linspace(0, k_max, N_BINS + 1)
        k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

        prof_gen, prof_fid = [], []
        hist_edges, hist_gen, hist_fid = [], [], []
        pk_gen, pk_fid = [], []

        for ch in range(3):
            lg = np.log10(1.0 + gen[ch]).ravel()
            lf = np.log10(1.0 + fid[ch]).ravel()

            # Radial profile (mean log10(1+rho) vs r)
            w_g, _ = np.histogram(r_map_mpc, bins=r_edges, weights=lg)
            w_f, _ = np.histogram(r_map_mpc, bins=r_edges, weights=lf)
            cnt, _ = np.histogram(r_map_mpc, bins=r_edges)
            cnt    = np.maximum(cnt, 1)
            prof_gen.append((w_g / cnt).tolist())
            prof_fid.append((w_f / cnt).tolist())

            # Pixel value histogram — for stars (ch=2), skip zero/near-zero pixels
            # to avoid a giant spike from unoccupied pixels swamping the plot
            if ch == 2:
                occ_thresh = 3.0  # only show occupied pixels: log10(1+M) > 3
                lg_hist = lg[lg > occ_thresh]
                lf_hist = lf[lf > occ_thresh]
            else:
                lg_hist, lf_hist = lg, lf
            all_vals = np.concatenate([lg_hist, lf_hist])
            if len(all_vals) == 0:
                all_vals = np.concatenate([lg, lf])
                lg_hist, lf_hist = lg, lf
            lo_h = np.percentile(all_vals, 1)
            hi_h = np.percentile(all_vals, 99)
            edges_h   = np.linspace(lo_h, hi_h, N_BINS + 1)
            cg, _     = np.histogram(lg_hist, bins=edges_h, density=True)
            cf, _     = np.histogram(lf_hist, bins=edges_h, density=True)
            centers_h = 0.5 * (edges_h[:-1] + edges_h[1:])
            hist_edges.append(centers_h.tolist())
            hist_gen.append(cg.tolist())
            hist_fid.append(cf.tolist())

            # Radially averaged power spectrum
            lg2d = np.log10(1.0 + gen[ch]) - np.log10(1.0 + gen[ch]).mean()
            lf2d = np.log10(1.0 + fid[ch]) - np.log10(1.0 + fid[ch]).mean()
            ps_g = np.abs(np.fft.fftshift(np.fft.fft2(lg2d))) ** 2
            ps_f = np.abs(np.fft.fftshift(np.fft.fft2(lf2d))) ** 2
            pg_w, _  = np.histogram(K_map, bins=k_edges, weights=ps_g.ravel())
            pf_w, _  = np.histogram(K_map, bins=k_edges, weights=ps_f.ravel())
            pk_cnt, _ = np.histogram(K_map, bins=k_edges)
            pk_cnt    = np.maximum(pk_cnt, 1)
            pk_gen.append((pg_w / pk_cnt).tolist())
            pk_fid.append((pf_w / pk_cnt).tolist())

        return {
            "profile": {"r": r_centers.tolist(), "gen": prof_gen, "fid": prof_fid},
            "hist":    {"x": hist_edges, "gen": hist_gen, "fid": hist_fid},
            "pk":      {"k": k_centers.tolist(), "gen": pk_gen, "fid": pk_fid},
        }

    def _build_html(self) -> str:
        """Build the full explorer HTML with embedded slider metadata."""
        from generate_sobol_explorer import ASTRO_INDICES, N_ASTRO

        slider_meta = []
        for j, idx in enumerate(ASTRO_INDICES):
            row      = self.param_meta.iloc[idx]
            log_flag = int(row["LogFlag"])
            lo, hi   = float(row["MinVal"]), float(row["MaxVal"])
            fid      = float(row["FiducialVal"])
            slider_meta.append({
                "name":        str(row["ParamName"]),
                "description": str(row["Description"]),
                "log_flag":    log_flag,
                "min_val":     lo,
                "max_val":     hi,
                "fiducial":    fid,
                "fiducial_u":  float(self.fiducial_pos[j]),
            })

        meta_json = json.dumps(slider_meta)
        fid_json  = json.dumps(self.fiducial_pos.tolist())

        # 1x1 black JPEG placeholder — prevents broken-image icon before first inference
        _blank = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AJQAB/9k="

        # Row 1: generated fields
        ch_html = "\n".join(
            f"""      <div class="channel-card">
        <div class="channel-label">{self.CHANNEL_NAMES[ch]}</div>
        <div class="img-wrap">
          <img id="img-{ch}" src="data:image/jpeg;base64,{_blank}" alt="{self.CHANNEL_NAMES[ch]}"/>
          <div class="loading-overlay" id="overlay-{ch}">
            <div class="spinner"></div><span id="overlay-msg-{ch}">Warming up…</span>
          </div>
        </div>
        <canvas class="colorbar" id="cbar-{ch}" width="220" height="22"></canvas>
      </div>"""
            for ch in range(3)
        )

        # Row 2: difference maps (generated - fiducial), same 3 channels
        diff_vmax_json = json.dumps([float(v) for v in self.diff_vmax])
        vmin_json      = json.dumps([float(v) for v in self.vmin])
        vmax_json      = json.dumps([float(v) for v in self.vmax])
        cmap_json      = json.dumps(self.CHANNEL_CMAPS)
        diff_html = "\n".join(
            f"""      <div class="channel-card diff-card">
        <div class="channel-label diff-label">{self.CHANNEL_NAMES[ch]} − fid</div>
        <div class="img-wrap">
          <img id="diff-{ch}" src="data:image/jpeg;base64,{_blank}" alt="diff {self.CHANNEL_NAMES[ch]}"/>
          <div class="loading-overlay" id="diff-overlay-{ch}">
            <div class="spinner"></div><span>Warming up…</span>
          </div>
        </div>
        <canvas class="colorbar" id="cbar-diff-{ch}" width="220" height="22"></canvas>
      </div>"""
            for ch in range(3)
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>BIND Halo Explorer</title>
<style>
  :root {{
    --bg: #0b0c14; --panel: #13141f; --panel2: #1a1b2e;
    --accent: #5e81f4; --accent2: #a78bfa; --text: #e2e4f0;
    --muted: #7b7e9e; --border: #2a2d45; --green: #34d399;
    --warn: #f59e0b; --radius: 10px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px;
    height: 100vh; overflow: hidden; display: flex; flex-direction: column;
  }}
  header {{
    background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 12px 24px; display: flex; align-items: baseline;
    gap: 16px; flex-shrink: 0; flex-wrap: wrap;
  }}
  header h1 {{ font-size: 17px; font-weight: 600; letter-spacing: .02em; }}
  header p  {{ font-size: 12px; color: var(--muted); flex: 1; min-width: 180px; }}
  .badge {{
    font-size: 11px; background: var(--panel2); border: 1px solid var(--border);
    border-radius: 20px; padding: 3px 10px; color: var(--muted); white-space: nowrap;
  }}
  .badge span {{ color: var(--accent); font-weight: 600; }}
  .timing-badge {{
    font-size: 12px; color: var(--muted); background: var(--panel);
    border: 1px solid var(--border); border-radius: 6px; padding: 3px 10px;
    white-space: nowrap;
  }}
  .timing-badge span {{ color: var(--green); font-family: 'Courier New', monospace; }}
  main {{ display: flex; flex: 1; overflow: hidden; }}

  /* ── Sidebar ── */
  #sidebar {{
    width: 320px; min-width: 320px; background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column; overflow: hidden;
  }}
  .sidebar-header {{ padding: 12px 16px 8px; border-bottom: 1px solid var(--border); flex-shrink: 0; }}
  .sidebar-header h2 {{
    font-size: 12px; text-transform: uppercase;
    letter-spacing: .1em; color: var(--muted); font-weight: 600;
  }}
  .sidebar-controls {{
    padding: 8px 16px; border-bottom: 1px solid var(--border);
    display: flex; gap: 8px; flex-shrink: 0;
  }}
  .btn {{
    background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
    border-radius: 6px; padding: 5px 10px; font-size: 12px; cursor: pointer;
    transition: background .15s, color .15s;
  }}
  .btn:hover {{ background: var(--border); color: var(--text); }}
  #slider-scroll {{
    flex: 1; overflow-y: auto; padding: 8px 0;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }}
  .group-label {{
    font-size: 10px; text-transform: uppercase; letter-spacing: .12em;
    color: var(--accent); font-weight: 700; padding: 10px 16px 4px;
  }}
  .slider-row {{ padding: 6px 16px; transition: background .1s; }}
  .slider-row.active {{ background: rgba(94,129,244,.07); }}
  .slider-top {{
    display: flex; justify-content: space-between;
    align-items: baseline; margin-bottom: 3px;
  }}
  .slider-name {{
    font-size: 12px; font-weight: 500; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 180px; cursor: help;
  }}
  .slider-val {{
    font-size: 12px; font-family: 'Courier New', monospace;
    color: var(--accent); min-width: 70px; text-align: right; white-space: nowrap;
  }}
  .fiducial-tick {{ position: relative; height: 4px; margin-bottom: 2px; }}
  .fiducial-marker {{
    position: absolute; top: 0; width: 2px; height: 4px;
    background: var(--accent2); opacity: .7; transform: translateX(-50%);
  }}
  input[type=range] {{
    -webkit-appearance: none; width: 100%; height: 4px;
    border-radius: 2px; background: var(--border); outline: none; cursor: pointer;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 13px; height: 13px;
    border-radius: 50%; background: var(--accent); cursor: pointer;
  }}

  /* ── Content panel ── */
  #content {{
    flex: 1; display: flex; flex-direction: column;
    padding: 16px; gap: 12px; overflow-y: auto;
  }}
  .images-row {{
    display: flex; gap: 12px; flex-shrink: 0; justify-content: center;
    flex-wrap: wrap;
  }}
  .channel-card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden; text-align: center;
    flex: 0 0 auto;
  }}
  .channel-label {{
    font-size: 11px; text-transform: uppercase; letter-spacing: .1em;
    color: var(--muted); padding: 6px 12px;
    border-bottom: 1px solid var(--border);
  }}
  .channel-card img {{
    display: block; width: 220px; height: 220px;
    background: var(--panel2); image-rendering: pixelated;
  }}
  canvas.colorbar {{
    display: block; width: 220px; height: 22px;
    border-top: 1px solid var(--border);
  }}
  .img-wrap {{ position: relative; }}
  .loading-overlay {{
    position: absolute; inset: 0;
    background: rgba(11,12,20,.85);
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; color: var(--muted); letter-spacing: .08em;
    text-transform: uppercase; opacity: 1;
    transition: opacity .15s; pointer-events: auto;
  }}
  .loading-overlay.hidden {{ opacity: 0; pointer-events: none; }}
  .spinner {{
    width: 22px; height: 22px; border: 2px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin .7s linear infinite; margin-right: 8px;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

  /* ── Param table ── */
  .param-table-wrap {{
    flex: 1; overflow-y: auto; background: var(--panel);
    border: 1px solid var(--border); border-radius: var(--radius);
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  thead th {{
    background: var(--panel2); color: var(--muted); text-align: left;
    padding: 7px 12px; font-size: 11px; text-transform: uppercase;
    letter-spacing: .08em; font-weight: 600;
    position: sticky; top: 0; border-bottom: 1px solid var(--border);
  }}
  tbody td {{
    padding: 5px 12px; border-bottom: 1px solid rgba(42,45,69,.5);
    font-family: 'Courier New', monospace; color: var(--muted); white-space: nowrap;
  }}
  tbody tr:hover td {{ background: rgba(94,129,244,.04); color: var(--text); }}
  td.name-col {{ font-family: inherit; color: var(--text); max-width: 160px;
    overflow: hidden; text-overflow: ellipsis; }}
  td.val-col  {{ color: var(--accent); }}
  td.fid-col  {{ color: var(--muted); }}
  .delta-up   {{ color: #f87171; }}
  .delta-dn   {{ color: #60a5fa; }}
  .delta-eq   {{ color: var(--muted); }}
  /* ── Difference row ── */
  .row-label {{
    font-size: 10px; text-transform: uppercase; letter-spacing: .1em;
    color: var(--muted); padding: 4px 0 2px; text-align: center;
    flex: 0 0 100%; opacity: .7;
  }}
  .diff-card .channel-label {{ color: var(--accent2); }}
  .diff-rows {{ display: flex; flex-direction: column; gap: 12px; flex-shrink: 0; }}
  /* ── Stats panel ── */
  .stats-panel {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: var(--radius); flex-shrink: 0;
  }}
  .stats-tabs {{
    display: flex; border-bottom: 1px solid var(--border);
  }}
  .stats-tab {{
    padding: 7px 16px; font-size: 11px; text-transform: uppercase;
    letter-spacing: .08em; color: var(--muted); cursor: pointer;
    border-bottom: 2px solid transparent; margin-bottom: -1px;
    transition: color .15s, border-color .15s;
  }}
  .stats-tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  .stats-tab:hover:not(.active) {{ color: var(--text); }}
  .stats-body {{
    display: flex; gap: 16px; padding: 12px 16px;
    overflow-x: auto; justify-content: center;
  }}
  .stats-chart-wrap {{
    display: flex; flex-direction: column; align-items: center; gap: 4px;
  }}
  .stats-ch-label {{
    font-size: 10px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--muted);
  }}
  .stats-chart-wrap canvas {{
    background: var(--panel2); border-radius: 4px;
    display: block;
  }}
</style>
</head>
<body>

<header>
  <h1>BIND Halo Explorer <span style="color:var(--accent2);font-size:12px;font-weight:400">(Live GPU)</span></h1>
  <p>Move sliders to explore astrophysical parameter space — inference runs in real time.</p>
  <div class="badge">CV_12 · most massive halo · <span>fm_base</span></div>
  <div class="timing-badge">Last inference: <span id="timing">—</span></div>
</header>

<main>
  <div id="sidebar">
    <div class="sidebar-header"><h2>Astrophysical Parameters</h2></div>
    <div class="sidebar-controls">
      <button class="btn" onclick="resetToFiducial()">⟳ Fiducial</button>
    </div>
    <div id="slider-scroll"></div>
  </div>

  <div id="content">
    <div class="diff-rows">
      <div class="images-row" id="images-row">
{ch_html}
      </div>
      <div class="images-row" id="diff-row">
{diff_html}
      </div>
    </div>
    <div class="param-table-wrap">
      <table>
        <thead>
          <tr>
            <th>Parameter</th><th>Current</th><th>Fiducial</th><th>Δ / fid</th>
          </tr>
        </thead>
        <tbody id="param-tbody"></tbody>
      </table>
    </div>
    <div class="stats-panel">
      <div class="stats-tabs">
        <div class="stats-tab active" data-tab="profile" onclick="switchTab('profile')">Radial Profile</div>
        <div class="stats-tab" data-tab="hist"    onclick="switchTab('hist')">Pixel PDF</div>
        <div class="stats-tab" data-tab="pk"      onclick="switchTab('pk')">Power Spectrum</div>
      </div>
      <div class="stats-body" id="stats-body">
        {''.join(f'<div class="stats-chart-wrap"><div class="stats-ch-label">{self.CHANNEL_NAMES[ch]}</div><canvas id="stat-canvas-{ch}" width="300" height="200"></canvas></div>' for ch in range(3))}
      </div>
    </div>
  </div>
</main>

<script>
const meta        = {meta_json};
const fiducialPos = {fid_json};
const diffVmax    = {diff_vmax_json};
const vminArr     = {vmin_json};
const vmaxArr     = {vmax_json};
const cmapNames   = {cmap_json};
const N_ASTRO     = {N_ASTRO};

const sliderU = new Float64Array(fiducialPos);

// ── Colorbar rendering ─────────────────────────────────────────────────────
// Matplotlib colormap data (sampled 256 stops) fetched once from a tiny
// canvas trick: we render the image off-screen and read pixels.
// Simpler approach: use pre-baked CSS gradient approximations per cmap.
const CMAP_STOPS = {{
  'inferno':  ['#000004','#1b0c41','#4a0c4e','#781c6d','#a52c60','#cf4446','#ed6925','#fb9b06','#f7d13d','#fcffa4'],
  'viridis':  ['#440154','#482878','#3e4989','#31688e','#26828e','#1f9e89','#35b779','#6ece58','#b5de2b','#fde725'],
  'YlOrRd':   ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026','#67000d'],
  'RdBu_r':   ['#053061','#2166ac','#4393c3','#92c5de','#d1e5f0','#f7f7f7','#fddbc7','#f4a582','#d6604d','#b2182b'],
}};

function drawColorbar(canvasId, cmapName, lo, hi, symmetric) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const barH = 10, tickY = barH + 1;

  // Gradient bar
  const stops = CMAP_STOPS[cmapName] || CMAP_STOPS['inferno'];
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  stops.forEach((c, i) => grad.addColorStop(i / (stops.length - 1), c));
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, barH);

  // Tick labels
  ctx.fillStyle = '#7b7e9e';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';

  const fmt = v => (Math.abs(v) < 0.01 || Math.abs(v) >= 1000)
    ? v.toExponential(1) : v.toFixed(2);

  if (symmetric) {{
    // show -vmax, 0, +vmax
    const ticks = [[-hi, 0], [0, W/2], [hi, W - 28]];
    ticks.forEach(([val, x]) => {{
      ctx.textAlign = x < 5 ? 'left' : (x > W - 30 ? 'right' : 'center');
      ctx.fillText((val >= 0 ? '+' : '') + fmt(val), x < 5 ? 2 : (x > W - 30 ? W - 2 : x), H - 2);
    }});
  }} else {{
    // show lo, mid, hi
    const mid = (lo + hi) / 2;
    [[fmt(lo), 2, 'left'], [fmt(mid), W/2, 'center'], [fmt(hi), W-2, 'right']].forEach(([label, x, align]) => {{
      ctx.textAlign = align;
      ctx.fillText(label, x, H - 2);
    }});
  }}
}}

function drawAllColorbars() {{
  for (let ch = 0; ch < 3; ch++) {{
    drawColorbar(`cbar-${{ch}}`, cmapNames[ch], vminArr[ch], vmaxArr[ch], false);
    drawColorbar(`cbar-diff-${{ch}}`, 'RdBu_r', -diffVmax[ch], diffVmax[ch], true);
  }}
}}

// ── Statistics charts ──────────────────────────────────────────────────────
let _currentTab = 'profile';
let _lastStats  = null;

const CH_COLORS    = ['#f472b6', '#34d399', '#fbbf24'];  // DM, Gas, Stars
const MASS_SYMBOLS = ['M_DM', 'M_gas', 'M★'];            // per-channel mass label

function switchTab(tab) {{
  _currentTab = tab;
  document.querySelectorAll('.stats-tab').forEach(el => {{
    el.classList.toggle('active', el.dataset.tab === tab);
  }});
  if (_lastStats) drawStats(_lastStats);
}}

function drawStats(stats) {{
  _lastStats = stats;
  const tab = _currentTab;
  const CH_NAMES = ['DM (Hydro)', 'Gas', 'Stars'];
  for (let ch = 0; ch < 3; ch++) {{
    const canvas = document.getElementById(`stat-canvas-${{ch}}`);
    if (!canvas) continue;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const PAD = {{ l:38, r:10, t:10, b:32 }};
    const pw = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;
    ctx.clearRect(0, 0, W, H);

    let xs, ygen, yfid, xLabel, yLabel, logX, logY;
    const msym = MASS_SYMBOLS[ch];
    if (tab === 'profile') {{
      xs = stats.profile.r;
      ygen = stats.profile.gen[ch];
      yfid = stats.profile.fid[ch];
      xLabel = 'r [Mpc/h]'; yLabel = 'log\u2081\u2080(1+' + msym + ') [M\u2609/h]'; logX = false; logY = false;
    }} else if (tab === 'hist') {{
      xs = stats.hist.x[ch];
      ygen = stats.hist.gen[ch];
      yfid = stats.hist.fid[ch];
      xLabel = 'log\u2081\u2080(1+' + msym + ') [M\u2609/h]'; yLabel = 'PDF'; logX = false; logY = false;
    }} else {{
      xs = stats.pk.k;
      ygen = stats.pk.gen[ch];
      yfid = stats.pk.fid[ch];
      // skip k=0 bin
      xs = xs.slice(1); ygen = ygen.slice(1); yfid = yfid.slice(1);
      xLabel = 'k [h/Mpc]'; yLabel = 'P(k)'; logX = true; logY = true;
    }}

    // Axis ranges
    const xvals = xs.filter(v => v > 0);
    const xmin = logX ? Math.log10(Math.min(...xvals)) : Math.min(...xs);
    const xmax = logX ? Math.log10(Math.max(...xs))    : Math.max(...xs);
    const allY  = [...ygen, ...yfid].filter(v => v > 0);
    const ymin  = logY ? Math.log10(Math.min(...allY)) : 0;
    const ymax  = logY ? Math.log10(Math.max(...allY)) : Math.max(...ygen, ...yfid);
    const xScale = v => PAD.l + (( (logX ? Math.log10(v) : v) - xmin ) / (xmax - xmin)) * pw;
    const yScale = v => PAD.t + ph - (( (logY ? Math.log10(Math.max(v,1e-30)) : v) - ymin ) / (ymax - ymin + 1e-30)) * ph;

    // Grid + axes
    ctx.strokeStyle = '#2a2d45'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {{
      const yy = PAD.t + ph * i / 4;
      ctx.moveTo(PAD.l, yy); ctx.lineTo(PAD.l + pw, yy);
    }}
    ctx.stroke();
    ctx.strokeStyle = '#3a3d55'; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, PAD.t + ph);
    ctx.lineTo(PAD.l + pw, PAD.t + ph); ctx.stroke();

    // Tick labels
    ctx.fillStyle = '#7b7e9e'; ctx.font = '8px monospace'; ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {{
      const frac = i / 4;
      const rawY = ymin + frac * (ymax - ymin);
      const val  = logY ? Math.pow(10, rawY) : rawY;
      const yy   = PAD.t + ph - frac * ph;
      ctx.fillText(val < 0.01 ? val.toExponential(0) : val.toFixed(val < 10 ? 2 : 0), PAD.l - 3, yy + 3);
    }}
    ctx.textAlign = 'center';
    for (let i = 0; i <= 3; i++) {{
      const frac = i / 3;
      const rawX = xmin + frac * (xmax - xmin);
      const val  = logX ? Math.pow(10, rawX) : rawX;
      const xx   = PAD.l + frac * pw;
      ctx.fillText(val < 0.1 ? val.toFixed(2) : val.toFixed(1), xx, PAD.t + ph + 10);
    }}
    // Axis labels
    ctx.fillStyle = '#9ba0be'; ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, PAD.l + pw / 2, H - 2);
    ctx.save(); ctx.translate(10, PAD.t + ph / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0); ctx.restore();

    // Draw lines
    const drawLine = (ys, color, dash) => {{
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.setLineDash(dash || []);
      let started = false;
      xs.forEach((x, i) => {{
        if (x <= 0 && logX) return;
        const py2 = ys[i];
        if (py2 <= 0 && logY) return;
        const px = xScale(x), py = yScale(py2);
        if (!started) {{ ctx.moveTo(px, py); started = true; }} else ctx.lineTo(px, py);
      }});
      ctx.stroke(); ctx.setLineDash([]);
    }};
    drawLine(ygen, CH_COLORS[ch]);
    drawLine(yfid, '#7b7e9e', [3, 3]);
  }}
}}

function uToPhys(u, m) {{
  if (m.log_flag) {{
    const lo = Math.log10(m.min_val), hi = Math.log10(m.max_val);
    return Math.pow(10, u * (hi - lo) + lo);
  }}
  return u * (m.max_val - m.min_val) + m.min_val;
}}
function fmtPhys(v, log_flag) {{
  if (log_flag) {{
    if (Math.abs(v) < 0.001 || Math.abs(v) >= 10000) return v.toExponential(2);
    return v.toPrecision(3);
  }}
  return v.toPrecision(4);
}}

function updateParamTable() {{
  const tbody = document.getElementById('param-tbody');
  const rows  = [];
  for (let j = 0; j < N_ASTRO; j++) {{
    const m      = meta[j];
    const curVal = uToPhys(sliderU[j], m);
    const fidVal = m.fiducial;
    const delta  = fidVal !== 0 ? (curVal - fidVal) / Math.abs(fidVal) : 0;
    const pct    = (delta * 100).toFixed(1);
    const cls    = Math.abs(delta) < 0.02 ? 'delta-eq' : (delta > 0 ? 'delta-up' : 'delta-dn');
    const sign   = delta >= 0 ? '+' : '';
    rows.push(`<tr>
      <td class="name-col" title="${{m.description}}">${{m.name}}</td>
      <td class="val-col">${{fmtPhys(curVal, m.log_flag)}}</td>
      <td class="fid-col">${{fmtPhys(fidVal, m.log_flag)}}</td>
      <td class="${{cls}}">${{sign}}${{pct}}%</td>
    </tr>`);
  }}
  tbody.innerHTML = rows.join('');
}}

let _debounceTimer = null;
let _inflight = false;
let _pendingParams = null;

function scheduleInference() {{
  clearTimeout(_debounceTimer);
  _debounceTimer = setTimeout(triggerInference, 400);
}}

async function triggerInference() {{
  const params = Array.from(sliderU);
  if (_inflight) {{
    _pendingParams = params;
    return;
  }}
  _inflight = true;
  setLoading(true);
  const t0 = performance.now();
  try {{
    const resp = await fetch('/generate', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{params_u: params}})
    }});
    if (!resp.ok) throw new Error(`HTTP ${{resp.status}}`);
    const data = await resp.json();
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    document.getElementById('timing').textContent = dt + ' s';
    for (let ch = 0; ch < 3; ch++) {{
      document.getElementById(`img-${{ch}}`).src =
        'data:image/jpeg;base64,' + data.images[ch];
      document.getElementById(`diff-${{ch}}`).src =
        'data:image/jpeg;base64,' + data.diff_images[ch];
    }}
    if (data.stats) drawStats(data.stats);
  }} catch(e) {{
    console.error('Inference error:', e);
    document.getElementById('timing').textContent = 'error (' + e.message + ')';
    for (let ch = 0; ch < 3; ch++) {{
      const m = document.getElementById(`overlay-msg-${{ch}}`);
      if (m) m.textContent = 'Error — check console';
    }}
    // Keep overlays visible so user sees the error, not a black image
    _inflight = false;
    if (_pendingParams) {{
      const p = _pendingParams; _pendingParams = null;
      for (let j = 0; j < N_ASTRO; j++) sliderU[j] = p[j];
      scheduleInference();
    }}
    return;
  }} finally {{
    setLoading(false);
    _inflight = false;
    if (_pendingParams) {{
      const p = _pendingParams; _pendingParams = null;
      for (let j = 0; j < N_ASTRO; j++) sliderU[j] = p[j];
      scheduleInference();
    }}
  }}
}}

function setLoading(on) {{
  for (let ch = 0; ch < 3; ch++) {{
    const el  = document.getElementById(`overlay-${{ch}}`);
    const el2 = document.getElementById(`diff-overlay-${{ch}}`);
    el.classList.toggle('hidden', !on);
    if (el2) el2.classList.toggle('hidden', !on);
    if (on) {{
      const m = el.querySelector('span');
      if (m) m.textContent = 'Computing…';
      const m2 = el2 && el2.querySelector('span');
      if (m2) m2.textContent = 'Computing…';
    }}
  }}
}}

function onSliderChange(j, u) {{
  sliderU[j] = u;
  document.querySelectorAll('.slider-row').forEach(r => r.classList.remove('active'));
  document.getElementById(`row-${{j}}`).classList.add('active');
  document.getElementById(`val-${{j}}`).textContent =
    fmtPhys(uToPhys(u, meta[j]), meta[j].log_flag);
  updateParamTable();
  scheduleInference();
}}

function resetToFiducial() {{
  for (let j = 0; j < N_ASTRO; j++) {{
    sliderU[j] = fiducialPos[j];
    const el = document.getElementById(`slider-${{j}}`);
    if (el) el.value = fiducialPos[j];
    const vEl = document.getElementById(`val-${{j}}`);
    if (vEl) vEl.textContent = fmtPhys(uToPhys(fiducialPos[j], meta[j]), meta[j].log_flag);
  }}
  document.querySelectorAll('.slider-row').forEach(r => r.classList.remove('active'));
  updateParamTable();
  scheduleInference();
}}

const GROUPS = [
  {{ label: "Supernova Feedback",  indices: [0,2,4,6,7,8,9,10,11,12,13,14] }},
  {{ label: "AGN Feedback",        indices: [1,3,15,16,17,18,19,20,21] }},
  {{ label: "UV Background",       indices: [22,23,24,25] }},
  {{ label: "Chemical Enrichment", indices: [26,27] }},
  {{ label: "Numerical / Other",   indices: [5,28,29] }},
];

function buildSliders() {{
  const container = document.getElementById('slider-scroll');
  container.innerHTML = '';
  const grouped   = new Set(GROUPS.flatMap(g => g.indices));
  const ungrouped = Array.from({{length: N_ASTRO}}, (_,i) => i).filter(i => !grouped.has(i));
  const allGroups = [...GROUPS, ...(ungrouped.length ? [{{label:'Other',indices:ungrouped}}] : [])];

  for (const grp of allGroups) {{
    const lbl = document.createElement('div');
    lbl.className = 'group-label'; lbl.textContent = grp.label;
    container.appendChild(lbl);

    for (const j of grp.indices) {{
      const m   = meta[j];
      const row = document.createElement('div');
      row.className = 'slider-row'; row.id = `row-${{j}}`;

      const top    = document.createElement('div'); top.className = 'slider-top';
      const nameEl = document.createElement('span');
      nameEl.className = 'slider-name'; nameEl.textContent = m.name;
      nameEl.title = m.description;
      const valEl  = document.createElement('span');
      valEl.className = 'slider-val'; valEl.id = `val-${{j}}`;
      valEl.textContent = fmtPhys(uToPhys(sliderU[j], m), m.log_flag);
      top.appendChild(nameEl); top.appendChild(valEl); row.appendChild(top);

      const tickWrap = document.createElement('div'); tickWrap.className = 'fiducial-tick';
      const tick     = document.createElement('div'); tick.className = 'fiducial-marker';
      tick.style.left = (m.fiducial_u * 100).toFixed(1) + '%';
      tickWrap.appendChild(tick); row.appendChild(tickWrap);

      const slider = document.createElement('input');
      slider.type = 'range'; slider.min = 0; slider.max = 1; slider.step = 0.001;
      slider.value = sliderU[j]; slider.id = `slider-${{j}}`;
      slider.addEventListener('input', () => onSliderChange(j, parseFloat(slider.value)));
      row.appendChild(slider);
      container.appendChild(row);
    }}
  }}
}}

buildSliders();
updateParamTable();
drawAllColorbars();
triggerInference();
</script>
</body>
</html>"""

    # ─────────────────────────────────────────────────────────────────────────
    # ASGI app (FastAPI served from this container)
    # ─────────────────────────────────────────────────────────────────────────

    @modal.asgi_app()
    def web(self):
        import asyncio
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        from starlette.applications import Starlette
        from starlette.middleware.cors import CORSMiddleware
        from starlette.responses import HTMLResponse, JSONResponse
        from starlette.routing import Route

        _executor = ThreadPoolExecutor(max_workers=1)
        server = self

        async def index(request):
            return HTMLResponse(server._build_html())

        async def generate(request):
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid JSON"}, status_code=400)
            params_u = np.array(body.get("params_u", []), dtype=np.float32)
            n_steps  = int(body.get("n_steps", 20))
            if len(params_u) != server.N_ASTRO:
                return JSONResponse(
                    {"error": f"Expected {server.N_ASTRO} params, got {len(params_u)}"},
                    status_code=400,
                )
            loop   = asyncio.get_event_loop()
            gen    = await loop.run_in_executor(
                _executor, server._infer, params_u, n_steps
            )
            images      = server._render(gen)
            diff_images = server._render_diff(gen)
            stats       = server._compute_stats(gen)
            return JSONResponse({"images": images, "diff_images": diff_images, "stats": stats})

        starlette_app = Starlette(routes=[
            Route("/",         index),
            Route("/generate", generate, methods=["POST"]),
        ])
        starlette_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
        return starlette_app
