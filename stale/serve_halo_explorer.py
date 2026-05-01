#!/usr/bin/env python3
"""
serve_halo_explorer.py  —  Live on-the-fly BIND halo explorer server.

Loads the fm_base flow-matching model once at startup, then serves a browser
UI where every slider change triggers real-time GPU inference (~1–2 s per
frame at n_steps=20).  The HTML served is tiny (<100 KB); no images are
pre-computed or embedded.

Quick start
-----------
  # On the compute node:
  python serve_halo_explorer.py --port 8765

  # On your laptop (SSH tunnel):
  ssh -L 8765:<nodename>:8765 mlee1@cluster

  # Open in Chrome:
  http://localhost:8765
"""

import argparse
import asyncio
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
import numpy as np
import torch
from PIL import Image

from data import NormStats, log_transform
from train import FlowMatchingLit
from generate_sobol_explorer import (
    ASTRO_INDICES, N_ASTRO, CHANNEL_NAMES, CHANNEL_CMAPS,
    CUTOUTS_PATH, CATALOG_PATH, RUN_DIR, CKPT_PATH,
    load_param_meta, load_cv12_params,
    fiducial_slider_positions, sobol_unit_to_physical,
    build_full_params, normalize_params_for_model,
    is_legacy_param_norm,
)

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise SystemExit(
        "Missing dependencies. Install with:\n"
        "  pip install fastapi uvicorn pydantic"
    )

# ---------------------------------------------------------------------------
# Global server state (populated at startup)
# ---------------------------------------------------------------------------
_S: dict = {}
_executor = ThreadPoolExecutor(max_workers=1)  # serialize GPU calls

app = FastAPI(title="BIND Halo Explorer")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_blocking(params_u: np.ndarray, n_steps: int) -> np.ndarray:
    """Run single-sample inference.  Runs in thread executor, not async-safe.

    Args:
        params_u : (N_ASTRO,) normalized [0,1] astro param vector.
        n_steps  : ODE integration steps.

    Returns:
        (3, H, W) float32 array in physical space (mass / (Msun/h) per pixel).
    """
    params_u = np.asarray(params_u, dtype=np.float32).reshape(1, N_ASTRO)
    astro    = sobol_unit_to_physical(params_u, _S["param_meta"], use_log=_S["use_log"])
    full     = build_full_params(astro, _S["cv12_params"])
    pnorm    = normalize_params_for_model(full, _S["norm_stats"])   # (1, 35)

    params_t = torch.from_numpy(pnorm).to(_S["device"])            # (1, 35)

    with torch.no_grad():
        gen = _S["fm"].sample(_S["cond_t"], _S["ls_t"], params_t, n_steps=n_steps)

    gen_np = gen.float().cpu().numpy()   # (1, 3, H, W)
    ns = _S["norm_stats"]
    for ch in range(3):
        gen_np[:, ch] = gen_np[:, ch] * ns.target_std[ch] + ns.target_mean[ch]
        gen_np[:, ch] = 10.0 ** gen_np[:, ch] - 1.0
    gen_np = np.clip(gen_np, 0.0, None)
    return gen_np[0]   # (3, H, W)


def _field_to_jpeg_b64(
    field_2d: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    size_px: int,
) -> str:
    field_log  = np.log10(1.0 + field_2d)
    dv         = max(vmax - vmin, 1e-10)
    field_norm = np.clip((field_log - vmin) / dv, 0.0, 1.0)
    rgba       = (mcm.get_cmap(cmap)(field_norm) * 255).astype(np.uint8)
    img        = Image.fromarray(rgba, "RGBA").convert("RGB")
    img        = img.resize((size_px, size_px), Image.BILINEAR)
    buf        = BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _render(gen: np.ndarray, size_px: int) -> list[str]:
    """Render (3, H, W) array → list of 3 base64 JPEG strings."""
    return [
        _field_to_jpeg_b64(
            gen[ch], CHANNEL_CMAPS[ch],
            float(_S["vmin"][ch]), float(_S["vmax"][ch]),
            size_px,
        )
        for ch in range(3)
    ]


# ---------------------------------------------------------------------------
# Startup: load model and compute reference vrange
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    args = _S["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[server] Device: {device}")

    param_meta  = load_param_meta()
    cv12_params = load_cv12_params()
    legacy      = is_legacy_param_norm(RUN_DIR / "norm_stats.npz")
    use_log     = not legacy
    print(f"[server] Param norm: {'linear (legacy)' if legacy else 'log10 for flagged'}")

    norm_stats = NormStats.load(RUN_DIR / "norm_stats.npz")
    if legacy:
        norm_stats.param_log_flag = np.zeros(35, dtype=np.int32)

    print("[server] Loading model checkpoint…")
    model = FlowMatchingLit.load_from_checkpoint(str(CKPT_PATH), map_location=device)
    model.eval()
    model.to(device)
    if hasattr(model, 'ema'):
        model.ema.copy_to(model.unet.parameters())
    print("[server] Model loaded.")

    # Load halo cutout (fixed condition)
    cutouts     = np.load(CUTOUTS_PATH)
    catalog     = np.load(CATALOG_PATH)
    halo_idx    = args.halo_idx
    condition   = cutouts["condition"][halo_idx]    # (H, W)
    large_scale = cutouts["large_scale"][halo_idx]  # (3, H, W)
    halo_mass   = catalog["masses"][halo_idx]
    print(f"[server] Halo {halo_idx}, mass = {halo_mass:.3e} M_sun/h")

    # Pre-normalize condition (same for every request)
    c  = log_transform(condition)[None, None].astype(np.float32)
    c  = (c - norm_stats.cond_mean) / (norm_stats.cond_std + 1e-8)
    ls = log_transform(large_scale)[None].astype(np.float32)
    ls = (ls - norm_stats.ls_mean[None, :, None, None]) / (
         norm_stats.ls_std[None, :, None, None] + 1e-8)

    fiducial_pos = fiducial_slider_positions(param_meta, use_log=use_log)

    _S.update(dict(
        device=device, fm=model.fm, norm_stats=norm_stats,
        param_meta=param_meta, cv12_params=cv12_params,
        use_log=use_log, fiducial_pos=fiducial_pos,
        cond_t=torch.from_numpy(c).to(device),
        ls_t=torch.from_numpy(ls).to(device),
    ))

    # Compute stable vrange from a small reference batch (fiducial ± a few variations)
    print("[server] Computing reference vrange…")
    ref_u = np.tile(fiducial_pos, (8, 1)).astype(np.float32)
    rng   = np.random.default_rng(0)
    for i in range(1, 8):
        j = rng.integers(N_ASTRO)
        ref_u[i, j] = rng.uniform(0.1, 0.9)

    ref_gen = np.stack([
        _infer_blocking(ref_u[i], n_steps=10) for i in range(len(ref_u))
    ])   # (8, 3, H, W)
    log_ref = np.log10(1.0 + ref_gen)
    _S["vmin"] = np.percentile(log_ref, 0.5,  axis=(0, 2, 3))
    _S["vmax"] = np.percentile(log_ref, 99.5, axis=(0, 2, 3))
    print(f"[server] vmin={_S['vmin']}, vmax={_S['vmax']}")
    print("[server] Ready — http://localhost:{args.port}")


# ---------------------------------------------------------------------------
# HTML (served once; slider metadata embedded as JSON)
# ---------------------------------------------------------------------------

def _build_app_html() -> str:
    param_meta   = _S["param_meta"]
    fiducial_pos = _S["fiducial_pos"]

    slider_meta = []
    for j, idx in enumerate(ASTRO_INDICES):
        row      = param_meta.iloc[idx]
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
            "fiducial_u":  float(fiducial_pos[j]),
        })

    meta_json = json.dumps(slider_meta)
    fid_json  = json.dumps(fiducial_pos.tolist())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>BIND Halo Explorer (Live)</title>
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
    gap: 16px; flex-shrink: 0;
  }}
  header h1 {{ font-size: 17px; font-weight: 600; letter-spacing: .02em; }}
  header p  {{ font-size: 12px; color: var(--muted); flex: 1; }}
  .badge {{
    font-size: 11px; background: var(--panel2); border: 1px solid var(--border);
    border-radius: 20px; padding: 3px 10px; color: var(--muted); white-space: nowrap;
  }}
  .badge span {{ color: var(--accent); font-weight: 600; }}
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
    padding: 16px; gap: 12px; overflow: hidden;
  }}
  .images-row {{
    display: flex; gap: 12px; flex-shrink: 0; justify-content: center;
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

  /* ── Loading overlay ── */
  .img-wrap {{ position: relative; }}
  .loading-overlay {{
    position: absolute; inset: 0;
    background: rgba(11,12,20,.75);
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; color: var(--muted); letter-spacing: .08em;
    text-transform: uppercase; opacity: 0;
    transition: opacity .15s; pointer-events: none;
  }}
  .loading-overlay.active {{ opacity: 1; pointer-events: auto; }}
  .spinner {{
    width: 22px; height: 22px; border: 2px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin .7s linear infinite; margin-right: 8px;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

  /* ── Status row ── */
  .status-row {{
    display: flex; align-items: center; gap: 12px; flex-shrink: 0;
  }}
  .timing-badge {{
    font-size: 12px; color: var(--muted); background: var(--panel);
    border: 1px solid var(--border); border-radius: 6px; padding: 3px 10px;
  }}
  .timing-badge span {{ color: var(--green); font-family: 'Courier New', monospace; }}

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
  td.name-col {{
    font-family: inherit; color: var(--text);
    max-width: 160px; overflow: hidden; text-overflow: ellipsis;
  }}
  td.val-col {{ color: var(--accent); }}
  td.fid-col {{ color: var(--muted); }}
  .delta-up {{ color: #f87171; }}
  .delta-dn {{ color: #60a5fa; }}
  .delta-eq {{ color: var(--muted); }}
</style>
</head>
<body>

<header>
  <h1>BIND Halo Explorer <span style="color:var(--accent2);font-size:12px;font-weight:400">(Live)</span></h1>
  <p>Drag sliders to explore astrophysical parameter space — inference runs on the GPU in real time.</p>
  <div class="badge">CV_12 · most massive halo · <span>fm_base</span></div>
  <div class="timing-badge">Last inference: <span id="timing">—</span></div>
</header>

<main>
  <div id="sidebar">
    <div class="sidebar-header"><h2>Astrophysical Parameters</h2></div>
    <div class="sidebar-controls">
      <button class="btn" onclick="resetToFiducial()">⟳ Fiducial</button>
    </div>
    <div id="slider-scroll"><!-- sliders injected by JS --></div>
  </div>

  <div id="content">
    <div class="images-row" id="images-row">
      {''.join(f'''
      <div class="channel-card">
        <div class="channel-label">{CHANNEL_NAMES[ch]}</div>
        <div class="img-wrap">
          <img id="img-{ch}" src="" alt="{CHANNEL_NAMES[ch]}"/>
          <div class="loading-overlay" id="overlay-{ch}">
            <div class="spinner"></div>Computing…
          </div>
        </div>
      </div>''' for ch in range(3))}
    </div>

    <div class="param-table-wrap">
      <table>
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Current</th>
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
// ── Embedded metadata ──────────────────────────────────────
const meta       = {meta_json};
const fiducialPos = {fid_json};
const N_ASTRO    = {N_ASTRO};

// ── Slider state ───────────────────────────────────────────
const sliderU = new Float64Array(fiducialPos);

// ── Helpers ────────────────────────────────────────────────
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

// ── Parameter table ────────────────────────────────────────
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

// ── Debounced inference ────────────────────────────────────
let _debounceTimer = null;
let _inflight = false;
let _pendingParams = null;

function scheduleInference() {{
  clearTimeout(_debounceTimer);
  _debounceTimer = setTimeout(triggerInference, 350);
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
    const data = await resp.json();
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    document.getElementById('timing').textContent = dt + ' s';
    for (let ch = 0; ch < 3; ch++) {{
      document.getElementById(`img-${{ch}}`).src =
        'data:image/jpeg;base64,' + data.images[ch];
    }}
  }} catch(e) {{
    console.error('Inference error:', e);
  }} finally {{
    setLoading(false);
    _inflight = false;
    if (_pendingParams) {{
      const p = _pendingParams;
      _pendingParams = null;
      for (let j = 0; j < N_ASTRO; j++) sliderU[j] = p[j];
      scheduleInference();
    }}
  }}
}}

function setLoading(on) {{
  for (let ch = 0; ch < 3; ch++) {{
    document.getElementById(`overlay-${{ch}}`).classList.toggle('active', on);
  }}
}}

// ── Slider change ──────────────────────────────────────────
function onSliderChange(j, u) {{
  sliderU[j] = u;
  document.querySelectorAll('.slider-row').forEach(r => r.classList.remove('active'));
  document.getElementById(`row-${{j}}`).classList.add('active');
  document.getElementById(`val-${{j}}`).textContent =
    fmtPhys(uToPhys(u, meta[j]), meta[j].log_flag);
  updateParamTable();
  scheduleInference();
}}

// ── Reset ──────────────────────────────────────────────────
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

// ── Build sliders ──────────────────────────────────────────
const GROUPS = [
  {{ label: "Supernova Feedback",   indices: [0,2,4,6,7,8,9,10,11,12,13,14] }},
  {{ label: "AGN Feedback",         indices: [1,3,15,16,17,18,19,20,21] }},
  {{ label: "UV Background",        indices: [22,23,24,25] }},
  {{ label: "Chemical Enrichment",  indices: [26,27] }},
  {{ label: "Numerical / Other",    indices: [5,28,29] }},
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

      const top  = document.createElement('div'); top.className = 'slider-top';
      const nameEl = document.createElement('span');
      nameEl.className = 'slider-name'; nameEl.textContent = m.name; nameEl.title = m.description;
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

// ── Init ───────────────────────────────────────────────────
buildSliders();
updateParamTable();
triggerInference();   // render fiducial on load
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_build_app_html())


class GenerateRequest(BaseModel):
    params_u: list[float]   # N_ASTRO normalized values in [0,1]
    n_steps:  int = 20


@app.post("/generate")
async def generate(req: GenerateRequest):
    params_u = np.array(req.params_u, dtype=np.float32)
    if len(params_u) != N_ASTRO:
        return JSONResponse({"error": f"Expected {N_ASTRO} params, got {len(params_u)}"}, status_code=400)
    n_steps = req.n_steps

    loop  = asyncio.get_event_loop()
    gen   = await loop.run_in_executor(_executor, _infer_blocking, params_u, n_steps)
    images = _render(gen, size_px=_S["args"].img_px)

    # Also return current physical values for the param table
    params_u_2d = params_u.reshape(1, N_ASTRO)
    astro = sobol_unit_to_physical(params_u_2d, _S["param_meta"], use_log=_S["use_log"])
    return JSONResponse({"images": images, "phys_vals": astro[0].tolist()})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",     type=int,   default=8765,  help="Server port (default 8765)")
    p.add_argument("--host",     type=str,   default="0.0.0.0", help="Bind address")
    p.add_argument("--img_px",   type=int,   default=220,   help="Rendered image size in pixels (default 220)")
    p.add_argument("--halo_idx", type=int,   default=0,     help="Halo index in CV_12 catalog (default 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _S["args"] = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
