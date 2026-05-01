#!/usr/bin/env python3
"""
Generate a self-contained HTML halo explorer via Sobol parameter sweep.

Samples N points from a 30-dimensional Sobol sequence covering the astrophysical
parameter space of IllustrisTNG-CAMELS, holding the 5 cosmological parameters
fixed at CV_12 fiducial values. Runs the fm_base flow-matching model for each
sample using the most massive halo from CV_12 as the fixed DMO condition, then
bakes every rendered image + metadata into a single zero-dependency HTML file
with live JavaScript nearest-neighbour sliders.

Usage
-----
  python generate_sobol_explorer.py               # 256 samples → halo_explorer.html
  python generate_sobol_explorer.py -n 512 -o my_explorer.html
  python generate_sobol_explorer.py --cache_npz generated.npz   # cache inference
  python generate_sobol_explorer.py --from_cache generated.npz  # skip inference
"""

import argparse
import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats.qmc import Sobol
from tqdm import tqdm

from data import (
    NormStats,
    PARAM_LOG_FLAG,
    PARAM_MIN_RAW,
    PARAM_MAX_RAW,
    log_transform,
)
from train import FlowMatchingLit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CUTOUTS_PATH = Path(
    "/mnt/home/mlee1/ceph/fm_testsuite/CV/sim_12/snap_090/"
    "mass_threshold_1p000e13/halo_cutouts.npz"
)
CATALOG_PATH = Path(
    "/mnt/home/mlee1/ceph/fm_testsuite/CV/sim_12/snap_090/"
    "mass_threshold_1p000e13/halo_catalog.npz"
)
CV_PARAM_FILE = Path(
    "/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/CV/"
    "CosmoAstroSeed_IllustrisTNG_L50n512_CV.txt"
)
SB35_CSV = Path(
    "/mnt/home/mlee1/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_param_minmax.csv"
)
RUN_DIR = Path("/mnt/home/mlee1/ceph/fm_runs/fm_base")
CKPT_PATH = RUN_DIR / "checkpoints" / "last.ckpt"

# ---------------------------------------------------------------------------
# Parameter index layout
# ---------------------------------------------------------------------------
# The 35-element param vector follows SB35 CSV order.
# Cosmological params (fixed to CV_12 values):
COSMO_INDICES = [0, 1, 6, 7, 8]  # Omega0, sigma8, OmegaBaryon, HubbleParam, n_s
# Astrophysical params (swept by Sobol):
ASTRO_INDICES = [i for i in range(35) if i not in COSMO_INDICES]  # 30 params
N_ASTRO = len(ASTRO_INDICES)  # 30

CHANNEL_NAMES = ["DM (hydro)", "Gas", "Stars"]
CHANNEL_CMAPS = ["magma", "viridis", "plasma"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_param_meta() -> pd.DataFrame:
    return pd.read_csv(SB35_CSV)


def load_cv12_params() -> np.ndarray:
    """Return the 35-element physical parameter vector for CV_12."""
    df = pd.read_csv(
        CV_PARAM_FILE, sep=r"\s+", comment="#", header=None, skiprows=1
    )
    row = df[df[0] == "CV_12"]
    return row.iloc[0, 1:36].values.astype(np.float32)


def sobol_unit_to_physical(
    u: np.ndarray, param_meta: pd.DataFrame, use_log: bool = True
) -> np.ndarray:
    """Map (N, 30) Sobol unit-cube samples to physical astrophysical param values.

    When use_log=True (modern models): log-flagged params span log10 space
    evenly, i.e. 10^(u*(log10_max - log10_min) + log10_min).
    When use_log=False (fm_base legacy): all params mapped linearly.
    """
    N = u.shape[0]
    result = np.zeros((N, N_ASTRO), dtype=np.float64)
    for j, idx in enumerate(ASTRO_INDICES):
        row = param_meta.iloc[idx]
        log_flag = int(row["LogFlag"]) if use_log else 0
        lo, hi = float(row["MinVal"]), float(row["MaxVal"])
        if log_flag:
            lo_log, hi_log = np.log10(lo), np.log10(hi)
            result[:, j] = 10.0 ** (u[:, j] * (hi_log - lo_log) + lo_log)
        else:
            result[:, j] = u[:, j] * (hi - lo) + lo
    return result.astype(np.float32)


def build_full_params(
    astro_phys: np.ndarray, cv12_params: np.ndarray
) -> np.ndarray:
    """Build (N, 35) full param vectors with cosmological params fixed to CV_12."""
    N = astro_phys.shape[0]
    params = np.tile(cv12_params, (N, 1)).astype(np.float32)
    for j, idx in enumerate(ASTRO_INDICES):
        params[:, idx] = astro_phys[:, j]
    return params


def normalize_params_for_model(params: np.ndarray, ns: NormStats) -> np.ndarray:
    """Apply training-consistent normalization to (N, 35) physical params → [0,1]^35.

    If ns.param_log_flag contains 1s (modern training), log10-transforms those
    params before min/max scaling (bounds are in log10 space for those entries).
    Legacy fm_base was trained with all-zero param_log_flag and raw linear bounds,
    so no log10 is applied in that case.
    """
    p = params.astype(np.float64)
    log_flags = ns.param_log_flag
    p = np.where(
        log_flags[None, :] == 1, np.log10(np.maximum(p, 1e-30)), p
    )
    rang = ns.param_max - ns.param_min
    return ((p - ns.param_min[None, :]) / (rang[None, :] + 1e-8)).astype(np.float32)


def is_legacy_param_norm(norm_stats_path: Path) -> bool:
    """Return True if this norm_stats.npz was saved without param_log_flag (fm_base legacy)."""
    d = np.load(norm_stats_path)
    return "param_log_flag" not in d.files


def fiducial_slider_positions(param_meta: pd.DataFrame, use_log: bool = True) -> np.ndarray:
    """Return the normalized [0,1] position of the fiducial value for each astro param."""
    positions = np.zeros(N_ASTRO, dtype=np.float64)
    for j, idx in enumerate(ASTRO_INDICES):
        row = param_meta.iloc[idx]
        log_flag = int(row["LogFlag"]) if use_log else 0
        fid = float(row["FiducialVal"])
        lo, hi = float(row["MinVal"]), float(row["MaxVal"])
        if log_flag:
            lo_log, hi_log = np.log10(lo), np.log10(hi)
            positions[j] = (np.log10(fid) - lo_log) / (hi_log - lo_log)
        else:
            positions[j] = (fid - lo) / (hi - lo)
    return positions.astype(np.float32)


def generate_1p_samples(
    fiducial_u: np.ndarray,  # (30,) normalized fiducial positions
    n_steps_per_param: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 1P (one-at-a-time) samples in normalized [0,1]^30 space.

    For each of the 30 astrophysical parameters, vary it linearly from 0→1
    while holding all others at fiducial.  Returns:
      u       : (N_1P, 30) normalized sample positions
      param_idx : (N_1P,) which parameter is being varied for each sample
    """
    steps = np.linspace(0.0, 1.0, n_steps_per_param, dtype=np.float32)
    rows, param_indices = [], []
    for j in range(N_ASTRO):
        for s in steps:
            row = fiducial_u.copy()
            row[j] = s
            rows.append(row)
            param_indices.append(j)
    u = np.stack(rows, axis=0)                           # (N_ASTRO*n_steps, 30)
    param_idx = np.array(param_indices, dtype=np.int32)  # (N_ASTRO*n_steps,)
    return u, param_idx


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    norm_stats: NormStats,
    condition: np.ndarray,    # (128, 128) raw
    large_scale: np.ndarray,  # (3, 128, 128) raw
    params_norm: np.ndarray,  # (N, 35) normalized
    fm,
    device: torch.device,
    n_steps: int = 50,
    batch_size: int = 16,
) -> np.ndarray:
    """Run batch inference and return (N, 3, 128, 128) physical-space outputs."""
    # Normalize condition/large_scale once (same halo for all samples)
    c = log_transform(condition)[None, None]  # (1, 1, H, W)
    c = (c - norm_stats.cond_mean) / (norm_stats.cond_std + 1e-8)

    ls = log_transform(large_scale)[None]  # (1, 3, H, W)
    ls = (ls - norm_stats.ls_mean[None, :, None, None]) / (
        norm_stats.ls_std[None, :, None, None] + 1e-8
    )

    cond_t = torch.from_numpy(c.astype(np.float32)).to(device)   # (1,1,H,W)
    ls_t   = torch.from_numpy(ls.astype(np.float32)).to(device)  # (1,3,H,W)

    N = params_norm.shape[0]
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc="Generating"):
            end = min(start + batch_size, N)
            B = end - start

            cond_b = cond_t.expand(B, -1, -1, -1)
            ls_b   = ls_t.expand(B, -1, -1, -1)
            params_b = torch.from_numpy(params_norm[start:end]).to(device)

            gen = fm.sample(cond_b, ls_b, params_b, n_steps=n_steps)
            gen_np = gen.float().cpu().numpy()

            # Denormalize: standardised log space → physical
            for ch in range(3):
                gen_np[:, ch] = (
                    gen_np[:, ch] * norm_stats.target_std[ch]
                    + norm_stats.target_mean[ch]
                )
                gen_np[:, ch] = 10.0 ** gen_np[:, ch] - 1.0
            gen_np = np.clip(gen_np, 0.0, None)
            outputs.append(gen_np)

    return np.concatenate(outputs, axis=0)  # (N, 3, 128, 128)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def compute_vrange(
    generated: np.ndarray,  # (N, 3, H, W) physical
    plo: float = 0.5,
    phi: float = 99.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel vmin/vmax in log10(1+x) space, stable across all samples."""
    log_gen = np.log10(1.0 + generated)  # (N, 3, H, W)
    vmin = np.percentile(log_gen, plo, axis=(0, 2, 3))   # (3,)
    vmax = np.percentile(log_gen, phi, axis=(0, 2, 3))   # (3,)
    return vmin, vmax


def field_to_png_b64(
    field_2d: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    size_px: int = 192,
) -> str:
    """Render a 2D physical-space field as a base64-encoded PNG string."""
    field_log = np.log10(1.0 + field_2d)
    dv = max(vmax - vmin, 1e-10)
    field_norm = np.clip((field_log - vmin) / dv, 0.0, 1.0)
    rgba = (mcm.get_cmap(cmap)(field_norm) * 255).astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA").convert("RGB")
    img = img.resize((size_px, size_px), Image.BILINEAR)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def render_all_images(
    generated: np.ndarray,  # (N, 3, H, W)
    vmin: np.ndarray,        # (3,)
    vmax: np.ndarray,        # (3,)
    size_px: int = 192,
) -> list[list[str]]:
    """Return list[N] of list[3] base64 PNG strings."""
    N = generated.shape[0]
    images: list[list[str]] = []
    for i in tqdm(range(N), desc="Rendering images"):
        row = []
        for ch in range(3):
            b64 = field_to_png_b64(
                generated[i, ch],
                CHANNEL_CMAPS[ch],
                float(vmin[ch]),
                float(vmax[ch]),
                size_px=size_px,
            )
            row.append(b64)
        images.append(row)
    return images


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(
    images: list[list[str]],         # [N_total][3] base64 JPEG strings
    sobol_norm: np.ndarray,          # (N_total, 30) normalized astro params [0,1]
    astro_phys: np.ndarray,          # (N_total, 30) physical astro param values
    param_meta: pd.DataFrame,
    fiducial_pos: np.ndarray,        # (30,) fiducial slider positions [0,1]
    sample_types: np.ndarray,        # (N_total,) 0=sobol, 1=1P
    onep_param_idx: np.ndarray,      # (N_total,) which param is varied (-1 for sobol)
    n_sobol: int,                    # number of Sobol samples
) -> str:
    N = len(images)

    # Build JSON payloads
    sobol_norm_list = sobol_norm.tolist()
    astro_phys_list = astro_phys.tolist()

    # Per-param metadata for the slider UI
    slider_meta = []
    for j, idx in enumerate(ASTRO_INDICES):
        row = param_meta.iloc[idx]
        log_flag = int(row["LogFlag"])
        lo, hi = float(row["MinVal"]), float(row["MaxVal"])
        fid = float(row["FiducialVal"])
        slider_meta.append({
            "name": str(row["ParamName"]),
            "description": str(row["Description"]),
            "log_flag": log_flag,
            "min_val": lo,
            "max_val": hi,
            "fiducial": fid,
            "fiducial_u": float(fiducial_pos[j]),
        })

    # Format physical value nicely
    def fmt_phys(val: float, log_flag: int) -> str:
        if log_flag:
            return f"{val:.3g}"
        return f"{val:.4g}"

    images_json   = json.dumps(images)
    sobol_json    = json.dumps(sobol_norm_list)
    phys_json     = json.dumps(astro_phys_list)
    meta_json     = json.dumps(slider_meta)
    fid_json      = json.dumps(fiducial_pos.tolist())
    stypes_json   = json.dumps(sample_types.tolist())
    onep_idx_json = json.dumps(onep_param_idx.tolist())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>BIND Halo Explorer — Astrophysical Parameter Space</title>
<style>
  :root {{
    --bg: #0b0c14;
    --panel: #13141f;
    --panel2: #1a1b2e;
    --accent: #5e81f4;
    --accent2: #a78bfa;
    --text: #e2e4f0;
    --muted: #7b7e9e;
    --border: #2a2d45;
    --green: #34d399;
    --warn: #f59e0b;
    --radius: 10px;
    --slider-thumb: #5e81f4;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 14px;
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}
  header {{
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: baseline;
    gap: 16px;
    flex-shrink: 0;
  }}
  header h1 {{
    font-size: 17px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: 0.02em;
  }}
  header p {{
    font-size: 12px;
    color: var(--muted);
    flex: 1;
  }}
  .badge {{
    font-size: 11px;
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px;
    color: var(--muted);
    white-space: nowrap;
  }}
  .badge span {{ color: var(--accent); font-weight: 600; }}
  main {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}

  /* ── Sidebar ── */
  #sidebar {{
    width: 320px;
    min-width: 320px;
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  .sidebar-header {{
    padding: 12px 16px 8px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }}
  .sidebar-header h2 {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-weight: 600;
  }}
  .sidebar-controls {{
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 8px;
    flex-shrink: 0;
  }}
  .btn {{
    background: var(--panel2);
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }}
  .btn:hover {{ background: var(--border); color: var(--text); }}
  #slider-scroll {{
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }}
  .group-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent);
    font-weight: 700;
    padding: 10px 16px 4px;
  }}
  .slider-row {{
    padding: 6px 16px;
    border-radius: 0;
    transition: background 0.1s;
  }}
  .slider-row.active {{ background: rgba(94,129,244,0.07); }}
  .slider-top {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 3px;
  }}
  .slider-name {{
    font-size: 12px;
    font-weight: 500;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 180px;
    cursor: help;
  }}
  .slider-val {{
    font-size: 12px;
    font-family: 'Courier New', monospace;
    color: var(--accent);
    min-width: 70px;
    text-align: right;
    white-space: nowrap;
  }}
  input[type=range] {{
    -webkit-appearance: none;
    width: 100%;
    height: 4px;
    border-radius: 2px;
    background: var(--border);
    outline: none;
    cursor: pointer;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: var(--slider-thumb);
    cursor: pointer;
    transition: transform 0.1s;
  }}
  input[type=range]::-webkit-slider-thumb:hover {{ transform: scale(1.3); }}
  .fiducial-tick {{
    height: 4px;
    position: relative;
  }}
  .fiducial-marker {{
    position: absolute;
    top: 0;
    width: 2px;
    height: 4px;
    background: var(--accent2);
    opacity: 0.6;
    border-radius: 1px;
  }}

  /* ── Content ── */
  #content {{
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: 20px;
    gap: 16px;
  }}
  .images-row {{
    display: flex;
    gap: 16px;
    justify-content: center;
    flex-shrink: 0;
  }}
  .channel-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  .channel-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
    width: 100%;
    text-align: center;
  }}
  .channel-card img {{
    display: block;
    image-rendering: auto;
    width: 256px;
    height: 256px;
  }}
  .status-row {{
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
  }}
  .dist-badge {{
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    color: var(--muted);
  }}
  .dist-badge span {{ color: var(--green); font-weight: 600; font-family: monospace; }}
  #dist-bar-wrap {{
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }}
  #dist-bar {{
    height: 100%;
    width: 0%;
    border-radius: 3px;
    transition: width 0.2s, background 0.2s;
  }}
  .param-table-wrap {{
    flex: 1;
    overflow-y: auto;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  thead th {{
    background: var(--panel2);
    color: var(--muted);
    text-align: left;
    padding: 7px 12px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    position: sticky;
    top: 0;
    border-bottom: 1px solid var(--border);
  }}
  tbody td {{
    padding: 5px 12px;
    border-bottom: 1px solid rgba(42,45,69,0.5);
    font-family: 'Courier New', monospace;
    color: var(--muted);
    white-space: nowrap;
  }}
  tbody tr:hover td {{ background: rgba(94,129,244,0.04); color: var(--text); }}
  td.name-col {{ font-family: inherit; color: var(--text); max-width: 160px; overflow: hidden; text-overflow: ellipsis; }}
  td.val-col {{ color: var(--accent); }}
  td.fid-col {{ color: var(--muted); }}
  td.delta-col {{ }}
  .delta-up {{ color: #f87171; }}
  .delta-dn {{ color: #60a5fa; }}
  .delta-eq {{ color: var(--muted); }}
</style>
</head>
<body>

<header>
  <h1>BIND Halo Explorer</h1>
  <p>Drag sliders to explore astrophysical parameter space — the nearest precomputed sample is shown instantly.</p>
  <div class="badge">CV_12 · most massive halo · <span>fm_base</span></div>
  <div class="badge"><span id="sample-idx">—</span> / {N - 1} · <span id="sample-type-badge">—</span></div>
</header>

<main>
  <div id="sidebar">
    <div class="sidebar-header">
      <h2>Astrophysical Parameters</h2>
    </div>
    <div class="sidebar-controls">
      <button class="btn" onclick="resetToFiducial()">⟳ Fiducial</button>
      <button class="btn" onclick="randomSample()">⚄ Random</button>
    </div>
    <div id="slider-scroll">
      <!-- sliders injected by JS -->
    </div>
  </div>

  <div id="content">
    <div class="images-row" id="images-row">
      <div class="channel-card">
        <div class="channel-label">DM (hydro)</div>
        <img id="img-0" src="" alt="DM"/>
      </div>
      <div class="channel-card">
        <div class="channel-label">Gas</div>
        <img id="img-1" src="" alt="Gas"/>
      </div>
      <div class="channel-card">
        <div class="channel-label">Stars</div>
        <img id="img-2" src="" alt="Stars"/>
      </div>
    </div>

    <div class="status-row">
      <div class="dist-badge">Nearest sample distance: <span id="dist-val">—</span></div>
      <div id="dist-bar-wrap"><div id="dist-bar"></div></div>
    </div>

    <div class="param-table-wrap">
      <table>
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Your value</th>
            <th>Nearest sample</th>
            <th>Fiducial</th>
            <th>Δ / fid</th>
          </tr>
        </thead>
        <tbody id="param-tbody"></tbody>
      </table>
    </div>
  </div>
</main>

<script>
// ──────────────────────────────────────────────────────────────
// Embedded data
// ──────────────────────────────────────────────────────────────
const N_SAMPLES = {N};
const N_SOBOL   = {n_sobol};
const N_ASTRO   = {N_ASTRO};

// sobolNormFlat[i][j]: normalised [0,1] position of sample i, param j
const sobolNormFlat = {sobol_json};
// physVals[i][j]: physical value of sample i, param j
const physVals = {phys_json};
// images[i][ch]: base64 JPEG string
const images   = {images_json};
// per-param metadata
const meta     = {meta_json};
// fiducial slider positions
const fiducialPos = {fid_json};
// sampleTypes[i]: 0 = Sobol, 1 = 1P variation
const sampleTypes = {stypes_json};
// onepParamIdx[i]: which param is varied for sample i (-1 = Sobol)
const onepParamIdx = {onep_idx_json};

// ──────────────────────────────────────────────────────────────
// Slider state: sliderU[j] in [0,1]
// ──────────────────────────────────────────────────────────────
const sliderU = new Float64Array(fiducialPos);  // copy



// ──────────────────────────────────────────────────────────────
// Parameter groups
// ──────────────────────────────────────────────────────────────
// ASTRO_INDICES order: [2,3,4,5,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
const GROUPS = [
  {{ label: "Supernova Feedback",
     indices: [0, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14] }},
  {{ label: "AGN Feedback",
     indices: [1, 3, 15, 16, 17, 18, 19, 20, 21] }},
  {{ label: "UV Background",
     indices: [22, 23, 24, 25] }},
  {{ label: "Chemical Enrichment",
     indices: [26, 27] }},
  {{ label: "Numerical / Other",
     indices: [5, 28, 29] }},
];

// ──────────────────────────────────────────────────────────────
// Build sliders
// ──────────────────────────────────────────────────────────────
function buildSliders() {{
  const container = document.getElementById('slider-scroll');
  container.innerHTML = '';

  // Collect ungrouped indices
  const grouped = new Set(GROUPS.flatMap(g => g.indices));
  const ungrouped = Array.from({{length: N_ASTRO}}, (_, i) => i).filter(i => !grouped.has(i));
  const allGroups = [
    ...GROUPS,
    ...(ungrouped.length ? [{{ label: "Other", indices: ungrouped }}] : [])
  ];

  for (const grp of allGroups) {{
    const lbl = document.createElement('div');
    lbl.className = 'group-label';
    lbl.textContent = grp.label;
    container.appendChild(lbl);

    for (const j of grp.indices) {{
      const m = meta[j];
      const row = document.createElement('div');
      row.className = 'slider-row';
      row.id = `row-${{j}}`;

      const top = document.createElement('div');
      top.className = 'slider-top';

      const nameEl = document.createElement('span');
      nameEl.className = 'slider-name';
      nameEl.textContent = m.name;
      nameEl.title = m.description;

      const valEl = document.createElement('span');
      valEl.className = 'slider-val';
      valEl.id = `val-${{j}}`;
      valEl.textContent = fmtPhys(uToPhys(sliderU[j], m), m.log_flag);

      top.appendChild(nameEl);
      top.appendChild(valEl);
      row.appendChild(top);

      // Fiducial tick bar
      const tickWrap = document.createElement('div');
      tickWrap.className = 'fiducial-tick';
      const tick = document.createElement('div');
      tick.className = 'fiducial-marker';
      tick.style.left = (m.fiducial_u * 100).toFixed(1) + '%';
      tickWrap.appendChild(tick);
      row.appendChild(tickWrap);

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = 0; slider.max = 1; slider.step = 0.001;
      slider.value = sliderU[j];
      slider.id = `slider-${{j}}`;
      slider.addEventListener('input', () => onSliderChange(j, parseFloat(slider.value)));
      row.appendChild(slider);

      container.appendChild(row);
    }}
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

// ──────────────────────────────────────────────────────────────
// Nearest-neighbour lookup (L2 in normalised space)
// ──────────────────────────────────────────────────────────────
function findNearest() {{
  let bestIdx = 0, bestDist = Infinity;
  for (let i = 0; i < N_SAMPLES; i++) {{
    let d = 0;
    for (let j = 0; j < N_ASTRO; j++) {{
      const diff = sobolNormFlat[i][j] - sliderU[j];
      d += diff * diff;
    }}
    if (d < bestDist) {{ bestDist = d; bestIdx = i; }}
  }}
  return {{ idx: bestIdx, dist: Math.sqrt(bestDist) }};
}}

// ──────────────────────────────────────────────────────────────
// UI update
// ──────────────────────────────────────────────────────────────
let lastIdx = -1;

function updateDisplay() {{
  const {{ idx, dist }} = findNearest();

  // Images
  for (let ch = 0; ch < 3; ch++) {{
    document.getElementById(`img-${{ch}}`).src =
      'data:image/jpeg;base64,' + images[idx][ch];
  }}

  // Sample index
  document.getElementById('sample-idx').textContent = idx;

  // Sample type badge
  const typeBadge = document.getElementById('sample-type-badge');
  if (sampleTypes[idx] === 1) {{
    const pName = meta[onepParamIdx[idx]].name;
    typeBadge.textContent = '1P:' + pName;
    typeBadge.style.color = 'var(--accent2)';
  }} else {{
    typeBadge.textContent = 'Sobol';
    typeBadge.style.color = 'var(--green)';
  }}

  // Distance indicator
  const maxDist = Math.sqrt(N_ASTRO); // theoretical max
  const normDist = Math.min(dist / (maxDist * 0.3), 1.0);
  document.getElementById('dist-val').textContent = dist.toFixed(3);
  const bar = document.getElementById('dist-bar');
  bar.style.width = (normDist * 100).toFixed(1) + '%';
  const r = Math.round(normDist * 240);
  const g = Math.round((1 - normDist) * 200 + 52);
  bar.style.background = `rgb(${{r}},${{g}},80)`;

  // Param table
  updateParamTable(idx);
  lastIdx = idx;
}}

function updateParamTable(idx) {{
  const tbody = document.getElementById('param-tbody');
  const rows = [];
  for (let j = 0; j < N_ASTRO; j++) {{
    const m = meta[j];
    const yourVal  = uToPhys(sliderU[j], m);
    const nearVal  = physVals[idx][j];
    const fidVal   = m.fiducial;
    const delta    = fidVal !== 0 ? (nearVal - fidVal) / Math.abs(fidVal) : 0;
    const pct      = (delta * 100).toFixed(1);
    const cls      = Math.abs(delta) < 0.02 ? 'delta-eq' : (delta > 0 ? 'delta-up' : 'delta-dn');
    const sign     = delta >= 0 ? '+' : '';
    rows.push(`<tr>
      <td class="name-col" title="${{m.description}}">${{m.name}}</td>
      <td class="val-col">${{fmtPhys(yourVal, m.log_flag)}}</td>
      <td class="val-col">${{fmtPhys(nearVal, m.log_flag)}}</td>
      <td class="fid-col">${{fmtPhys(fidVal, m.log_flag)}}</td>
      <td class="${{cls}}">${{sign}}${{pct}}%</td>
    </tr>`);
  }}
  tbody.innerHTML = rows.join('');
}}

function onSliderChange(j, u) {{
  sliderU[j] = u;
  // Highlight active row
  document.querySelectorAll('.slider-row').forEach(r => r.classList.remove('active'));
  document.getElementById(`row-${{j}}`).classList.add('active');
  // Update displayed value
  document.getElementById(`val-${{j}}`).textContent =
    fmtPhys(uToPhys(u, meta[j]), meta[j].log_flag);
  updateDisplay();
}}



// ──────────────────────────────────────────────────────────────
// Buttons
// ──────────────────────────────────────────────────────────────
function resetToFiducial() {{
  for (let j = 0; j < N_ASTRO; j++) {{
    sliderU[j] = fiducialPos[j];
    const el = document.getElementById(`slider-${{j}}`);
    if (el) el.value = fiducialPos[j];
    const vEl = document.getElementById(`val-${{j}}`);
    if (vEl) vEl.textContent = fmtPhys(uToPhys(fiducialPos[j], meta[j]), meta[j].log_flag);
  }}
  document.querySelectorAll('.slider-row').forEach(r => r.classList.remove('active'));
  updateDisplay();
}}

function randomSample() {{
  const i = Math.floor(Math.random() * N_SAMPLES);
  for (let j = 0; j < N_ASTRO; j++) {{
    sliderU[j] = sobolNormFlat[i][j];
    const el = document.getElementById(`slider-${{j}}`);
    if (el) el.value = sliderU[j];
    const vEl = document.getElementById(`val-${{j}}`);
    if (vEl) vEl.textContent = fmtPhys(uToPhys(sliderU[j], meta[j]), meta[j].log_flag);
  }}
  document.querySelectorAll('.slider-row').forEach(r => r.classList.remove('active'));
  updateDisplay();
}}

// ──────────────────────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────────────────────
buildSliders();
updateDisplay();
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--n_samples", type=int, default=256,
                   help="Number of Sobol samples to generate (default 256)")
    p.add_argument("-o", "--output", type=str, default="halo_explorer.html",
                   help="Output HTML path (default halo_explorer.html)")
    p.add_argument("--n_steps", type=int, default=50,
                   help="ODE steps for fm.sample (default 50)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Inference batch size (default 16)")
    p.add_argument("--img_px", type=int, default=192,
                   help="Rendered image size in pixels (default 192)")
    p.add_argument("--cache_npz", type=str, default=None,
                   help="If set, save generated arrays to this .npz path")
    p.add_argument("--from_cache", type=str, default=None,
                   help="Skip inference; load generated arrays from this .npz path")
    p.add_argument("--halo_idx", type=int, default=0,
                   help="Index into CV_12 halo catalog (0=most massive, default 0)")
    p.add_argument("--legacy_params", action="store_true", default=None,
                   help="Force linear param normalization (no log10). "
                        "Auto-detected from norm_stats if omitted.")
    p.add_argument("--n_1p_steps", type=int, default=32,
                   help="Number of evenly-spaced steps per parameter for 1P variations "
                        "(default 32; 0 to disable)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load param metadata ----
    param_meta  = load_param_meta()
    cv12_params = load_cv12_params()
    print(f"CV_12 params: Omega0={cv12_params[0]:.3f}, sigma8={cv12_params[1]:.3f}, "
          f"ASN1={cv12_params[2]:.3f}, AAGN1={cv12_params[3]:.3f}")

    # ---- Detect legacy param normalization (fm_base has no log10 on params) ----
    if args.legacy_params is None:
        legacy = is_legacy_param_norm(RUN_DIR / "norm_stats.npz")
    else:
        legacy = args.legacy_params
    use_log = not legacy
    print(f"Param normalization: {'linear (legacy, no log10)' if legacy else 'log10 for flagged params'}")

    # ---- Generate Sobol sequence in [0,1]^30 ----
    m = int(np.ceil(np.log2(args.n_samples)))
    sobol = Sobol(d=N_ASTRO, scramble=True, seed=42)
    sobol_u = sobol.random_base2(m=m)[: args.n_samples]   # (N_sobol, 30) in [0,1]
    print(f"Sobol grid: {sobol_u.shape[0]} samples × {N_ASTRO} astrophysical dims")

    # ---- Generate 1P samples ----
    fiducial_pos = fiducial_slider_positions(param_meta, use_log=use_log)  # (30,)
    if args.n_1p_steps > 0:
        onep_u, onep_param_idx = generate_1p_samples(fiducial_pos, n_steps_per_param=args.n_1p_steps)
        print(f"1P grid: {onep_u.shape[0]} samples ({N_ASTRO} params × {args.n_1p_steps} steps)")
    else:
        onep_u = np.zeros((0, N_ASTRO), dtype=np.float32)
        onep_param_idx = np.zeros(0, dtype=np.int32)

    # ---- Combine into one pool ----
    all_u = np.concatenate([sobol_u, onep_u], axis=0).astype(np.float32)  # (N_total, 30)
    sample_types   = np.concatenate([
        np.zeros(len(sobol_u), dtype=np.int32),
        np.ones(len(onep_u),   dtype=np.int32),
    ])  # 0=Sobol, 1=1P
    all_onep_param = np.concatenate([
        np.full(len(sobol_u), -1, dtype=np.int32),
        onep_param_idx,
    ])
    n_sobol = len(sobol_u)
    N_total = len(all_u)
    print(f"Total samples: {N_total}")

    # ---- Map to physical params ----
    astro_phys  = sobol_unit_to_physical(all_u, param_meta, use_log=use_log)  # (N_total, 30)
    full_params = build_full_params(astro_phys, cv12_params)                  # (N_total, 35)

    if args.from_cache:
        print(f"Loading cached inference from {args.from_cache}")
        cache = np.load(args.from_cache)
        generated = cache["generated"]
        if generated.shape[0] != N_total:
            raise ValueError(
                f"Cache has {generated.shape[0]} samples but expected {N_total}"
            )
    else:
        # ---- Load model ----
        print("Loading norm stats...")
        norm_stats = NormStats.load(RUN_DIR / "norm_stats.npz")
        print(f"  Stellar norm: mean={norm_stats.target_mean[2]:.4f}, "
              f"std={norm_stats.target_std[2]:.4f}")

        print("Loading model checkpoint...")
        model = FlowMatchingLit.load_from_checkpoint(
            str(CKPT_PATH), map_location=device
        )
        model.eval()
        model.to(device)
        if hasattr(model, 'ema'):
            model.ema.copy_to(model.unet.parameters())
        fm = model.fm
        print(f"  Checkpoint loaded")

        # ---- Load halo cutout ----
        cutouts = np.load(CUTOUTS_PATH)
        catalog = np.load(CATALOG_PATH)
        halo_idx = args.halo_idx
        condition   = cutouts["condition"][halo_idx]    # (128, 128)
        large_scale = cutouts["large_scale"][halo_idx]  # (3, 128, 128)
        halo_mass   = catalog["masses"][halo_idx]
        print(f"  Using halo index {halo_idx}, mass = {halo_mass:.3e} M_sun/h")

        # ---- For legacy models: override param_log_flag to all-zeros ----
        if legacy:
            norm_stats.param_log_flag = np.zeros(35, dtype=np.int32)

        # ---- Normalize params ----
        params_norm = normalize_params_for_model(full_params, norm_stats)  # (N_total, 35)

        # ---- Run inference ----
        generated = run_inference(
            norm_stats, condition, large_scale, params_norm,
            fm, device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
        )
        print(f"  Generated shape: {generated.shape}")

        if args.cache_npz:
            np.savez_compressed(
                args.cache_npz,
                generated=generated,
                all_u=all_u,
                astro_phys=astro_phys,
                sample_types=sample_types,
                all_onep_param=all_onep_param,
            )
            print(f"  Cached to {args.cache_npz}")

    # ---- Compute per-channel vrange ----
    vmin, vmax = compute_vrange(generated)
    print(f"  vmin: {vmin},  vmax: {vmax}")

    # ---- Render images ----
    images = render_all_images(generated, vmin, vmax, size_px=args.img_px)

    # ---- Build HTML ----
    print("Building HTML...")
    html = build_html(
        images=images,
        sobol_norm=all_u,
        astro_phys=astro_phys,
        param_meta=param_meta,
        fiducial_pos=fiducial_pos,
        sample_types=sample_types,
        onep_param_idx=all_onep_param,
        n_sobol=n_sobol,
    )

    out_path = Path(args.output)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"Written to {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
