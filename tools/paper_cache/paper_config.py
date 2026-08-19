"""Single source of truth for the BIND2 paper-figure cache pipeline.

Imported by both the parallel cache builders (`build_metric.py`,
`build_gpu_insets.py`) and the load-only figure notebooks, so there is exactly
one place to flip from the development suite (`fm_lowmass` / `fm_thermo`) to the
paper's spine suite (`fm_redshift` @ z=0) — via env vars, no code edits.

Env overrides
-------------
PAPER_SUITE_ROOT   suite-eval output root      (default /mnt/home/mlee1/ceph/fm_lowmass)
PAPER_MODEL_SUBDIR per-sim model output dir    (default fm_thermo_ema)
PAPER_MASS_DIR     mass-threshold subdir        (default mass_threshold_1p000e12)
PAPER_MODEL_TAG    cache namespace              (default derived from suite root)
PAPER_CACHE_DIR    cache output root            (default /mnt/home/mlee1/ceph/paper_cache/<tag>)

Spine (Phase III) flip:
    export PAPER_SUITE_ROOT=/mnt/home/mlee1/ceph/fm_redshift_suite
    export PAPER_MODEL_SUBDIR=fm_redshift_ema
    export PAPER_MODEL_TAG=fm_redshift
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo / package locations ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]          # .../vdm_bind2
SRC_ROOT = REPO_ROOT / "src"
ASSET_CSV = SRC_ROOT / "bind" / "assets" / "SB35_param_minmax.csv"

# ── Suite-eval location (env-overridable; dev defaults) ─────────────────────
SUITE_ROOT = Path(os.environ.get("PAPER_SUITE_ROOT", "/mnt/home/mlee1/ceph/fm_lowmass"))
MODEL_SUBDIR = os.environ.get("PAPER_MODEL_SUBDIR", "fm_thermo_ema")
MASS_DIR = os.environ.get("PAPER_MASS_DIR", "mass_threshold_1p000e12")
SNAP = os.environ.get("PAPER_SNAP", "snap_090")

MODEL_TAG = os.environ.get(
    "PAPER_MODEL_TAG",
    "fm_redshift" if "redshift" in str(SUITE_ROOT) else "fm_thermo",
)
CACHE_DIR = Path(
    os.environ.get("PAPER_CACHE_DIR", f"/mnt/home/mlee1/ceph/paper_cache/{MODEL_TAG}")
)
PARTIAL_DIR = CACHE_DIR / "partials"

# ── Suites ──────────────────────────────────────────────────────────────────
SUITES = ("CV", "1P", "Test")
SUITE_COLORS = {"CV": "tab:green", "1P": "tab:blue", "Test": "tab:red"}
SUITE_DISPLAY = {"CV": "CV", "1P": "1P", "Test": "SB35"}

# ── Box / patch geometry ────────────────────────────────────────────────────
BOX_SIZE = 50.0            # Mpc/h
N_PIX_FULL = 1024
PATCH_PIX = 128
PATCH_BOX = BOX_SIZE * PATCH_PIX / N_PIX_FULL        # 6.25 Mpc/h
MPC_PER_PIX_FULL = BOX_SIZE / N_PIX_FULL             # full-box pixel (Mpc/h)
MPC_PER_PIX_PATCH = PATCH_BOX / PATCH_PIX            # patch pixel (Mpc/h)
RHO_CRIT = 2.775e11        # M_sun/h per (Mpc/h)^3
N_PARAMS = 35

# ── Channels (generated_halos.npz `generated` is (N, 7, 128, 128)) ──────────
MASS_CHANNELS = ["DM_hydro", "Gas", "Stars"]         # generated ch 0,1,2
# NB: these are the paper-cache COLUMN NAMES, not bind.data.THERMO_KEYS
# (= compton_y, temperature, entropy, pressure). They are baked into every
# cached mass/thermo table already written under $PAPER_CACHE_DIR, so renaming
# them would orphan those caches -- the short forms are kept deliberately.
# "P_e" is a legacy misnomer: the channel is the TOTAL thermal pressure
# (gamma-1) rho u in Pa, not the electron pressure. See docs/thermo.md.
THERMO_CHANNELS = ["compton_y", "T", "entropy", "P_e"]  # generated ch 3,4,5,6
N_MASS_CH = len(MASS_CHANNELS)
N_THERMO_CH = len(THERMO_CHANNELS)
CH_DISPLAY = {"DM_hydro": "DM (hydro)", "Gas": "Gas", "Stars": "Stars"}
THERMO_DISPLAY = {"compton_y": r"$y$", "T": r"$T$", "entropy": r"$K$", "P_e": r"$P$"}
# Stars need a density floor before shape/lit-pixel weighting.
STAR_THRESH = 1e-3

# ── Mass bins for the by-mass-bin validation (log10 M200c) ──────────────────
# The aggregate parameter-response figures (mass Fig 3a, profile Fig 3b) average
# each sim over its halo population, then Spearman-correlate vs params across sims.
# Restrict that population to the *trained* regime (>=1e13) — matching the original
# paper — so the per-sim mean isn't dominated by the 1e12-1e13 extrapolation halos,
# which inflates the True-BIND residual. The by-mass-bin figures (2b, 4) are unaffected.
PARAM_RESPONSE_MASS_MIN = 1e13
# Parameter-response is computed per mass window: `trained` (>=1e13, the main figures)
# and `lowmass` (1e12-1e13, the extrapolation-regime appendix). Each param-response
# cache stores a rho grid per window; figures pick the one they need.
PARAM_WINDOWS = {"trained": (PARAM_RESPONSE_MASS_MIN, float("inf")),
                 "lowmass": (1e12, PARAM_RESPONSE_MASS_MIN)}

MASS_EDGES = np.array([12.0, 12.5, 13.0, 13.5, 14.0, 15.5])
MASS_BIN_LABELS = [
    f"[{MASS_EDGES[i]:.1f}, {MASS_EDGES[i+1]:.1f})" for i in range(len(MASS_EDGES) - 1)
]
N_MASS_BINS = len(MASS_EDGES) - 1


def mass_bin_index(log_m200c):
    """Bin index in [0, N_MASS_BINS); -1 if outside MASS_EDGES. Vectorized."""
    idx = np.searchsorted(MASS_EDGES, log_m200c, side="right") - 1
    idx = np.where((idx >= 0) & (idx < N_MASS_BINS), idx, -1)
    return idx


# ── Parameter metadata / labels (from the bundled SB35 asset) ───────────────
PARAM_LABELS = {
    1: r"$\Omega_m$", 2: r"$\sigma_8$", 3: r"$A_{\rm SN1}$",
    4: r"$A_{\rm AGN1}$", 5: r"$A_{\rm ASN2}$", 6: r"$A_{\rm AGN2}$",
}
PARAM_LOG: dict[int, bool] = {}
try:
    _meta = pd.read_csv(ASSET_CSV)
    _names = list(_meta["ParamName"])
    PARAM_LOG = {i + 1: bool(v) for i, v in enumerate(_meta["LogFlag"])}
    for i, name in enumerate(_names):
        PARAM_LABELS.setdefault(i + 1, name)
    for i in range(len(_names), N_PARAMS):
        PARAM_LABELS.setdefault(i + 1, f"p{i + 1}")
except Exception as exc:  # pragma: no cover - labels are cosmetic
    print(f"[paper_config] WARN could not read {ASSET_CSV}: {exc}")
    for i in range(N_PARAMS):
        PARAM_LABELS.setdefault(i + 1, f"p{i + 1}")

# Cosmology params (1-indexed): Omega_m, sigma8, Omega_b, h, n_s — excluded from
# the astrophysical parameter-response panels.
COSMO_PARAM_IDX = {1, 2, 7, 8, 9}


# ── Per-sim record + discovery ──────────────────────────────────────────────
def sim_record(sim_dir: Path, suite: str) -> dict:
    snap = sim_dir / SNAP
    mass = snap / MASS_DIR
    model = mass / MODEL_SUBDIR
    rec = {
        "suite": suite,
        "sim_id": sim_dir.name,
        "key": f"{suite}/{sim_dir.name}",
        "sim_dir": sim_dir,
        "full_maps": snap / "full_maps.npz",
        "catalog": mass / "halo_catalog.npz",
        "truth_thermo": mass / "truth_thermo_patches.npz",
        "cutouts": mass / "halo_cutouts.npz",
        "generated": model / "generated_halos.npz",
        "composite": model / "composite.npz",
        "summary": model / "summary.json",
    }
    rec["available"] = all(
        rec[k].exists() for k in ("full_maps", "catalog", "generated")
    )
    return rec


def discover_sims(suites=SUITES, available_only=True) -> list[dict]:
    """All sim records under SUITE_ROOT for the given suites (numeric-sorted)."""
    recs = []
    for suite in suites:
        root = SUITE_ROOT / suite
        if not root.exists():
            continue
        for sd in sorted(root.iterdir(), key=_sim_sort_key):
            if sd.is_dir():
                recs.append(sim_record(sd, suite))
    if available_only:
        recs = [r for r in recs if r["available"]]
    return recs


def _sim_sort_key(p: Path):
    name = p.name
    if "_" in name and name.rsplit("_", 1)[-1].isdigit():
        return (name.rsplit("_", 1)[0], int(name.rsplit("_", 1)[-1]))
    return (name, 0)


def resolve_record(key: str) -> dict:
    """'CV/sim_0' -> record dict."""
    suite, sim_id = key.split("/", 1)
    return sim_record(SUITE_ROOT / suite / sim_id, suite)


# ── Geometry helpers ────────────────────────────────────────────────────────
def extract_patch(field_2d, cx_pix, cy_pix, size=PATCH_PIX):
    """Periodic size×size cutout of a full-box 2D field centred on (cx,cy)."""
    n = field_2d.shape[0]
    half = size // 2
    ix = (cx_pix - half + np.arange(size)) % n
    iy = (cy_pix - half + np.arange(size)) % n
    return field_2d[np.ix_(ix, iy)]


def centers_to_pixels(centers_mpc):
    """(N,2) halo centres in Mpc/h -> integer pixel coords on the 1024 grid."""
    ppm = N_PIX_FULL / BOX_SIZE
    return (np.asarray(centers_mpc) * ppm).astype(np.int64) % N_PIX_FULL


def r200c_mpc(m200c_msunh):
    """R200c [Mpc/h] from M200c [M_sun/h] (200× critical density)."""
    m = np.asarray(m200c_msunh, dtype=np.float64)
    return (3.0 * m / (4.0 * np.pi * 200.0 * RHO_CRIT)) ** (1.0 / 3.0)


def r200_pix_patch(catalog):
    """R200c in *patch* pixels for every halo, preferring the stored `r200s`
    (Mpc/h, = FOF Group_R_Crit200) and falling back to M200c."""
    r_mpc = catalog["r200s"] if "r200s" in catalog else r200c_mpc(catalog["masses"])
    return np.asarray(r_mpc, dtype=np.float64) / MPC_PER_PIX_PATCH


# Centred pixel-radius grid on a patch (halo at pixel PATCH_PIX/2).
_gy, _gx = np.mgrid[0:PATCH_PIX, 0:PATCH_PIX]
RR_PIX_PATCH = np.hypot(_gx - PATCH_PIX / 2.0, _gy - PATCH_PIX / 2.0)  # (128,128)


# ── Loaders ─────────────────────────────────────────────────────────────────
def load_full_maps(rec) -> dict:
    d = np.load(rec["full_maps"])
    return {"dmo_fullbox": d["dmo_fullbox"], "truth_maps": d["truth_maps"]}


def load_catalog(rec) -> dict:
    d = np.load(rec["catalog"])
    return {k: d[k] for k in d.files}


def load_generated(rec) -> np.ndarray:
    """(N, 7, 128, 128): channels 0-2 mass, 3-6 thermo."""
    return np.load(rec["generated"])["generated"]


def load_composite(rec) -> dict:
    d = np.load(rec["composite"])
    return {k: d[k] for k in d.files}


def load_truth_thermo(rec) -> np.ndarray:
    """(N, 4, 128, 128) truth thermo patches, or None if absent."""
    if not rec["truth_thermo"].exists():
        return None
    return np.load(rec["truth_thermo"])["truth_thermo"]


def extract_truth_mass_patches(full_maps, centers_pix) -> np.ndarray:
    """(N, 3, 128, 128) truth mass patches at the halo centres."""
    tm = full_maps["truth_maps"]
    n = len(centers_pix)
    out = np.zeros((n, N_MASS_CH, PATCH_PIX, PATCH_PIX), dtype=np.float32)
    for i, (cx, cy) in enumerate(centers_pix):
        for c in range(N_MASS_CH):
            out[i, c] = extract_patch(tm[c], cx, cy)
    return out


def sim_params(catalog) -> np.ndarray:
    """(35,) parameter vector for a sim (identical across its halos)."""
    p = np.asarray(catalog["params"], dtype=np.float64)
    return p[0] if p.ndim == 2 else p


def describe():
    print(f"SUITE_ROOT   {SUITE_ROOT}")
    print(f"MODEL_SUBDIR {MODEL_SUBDIR}   MASS_DIR {MASS_DIR}   SNAP {SNAP}")
    print(f"MODEL_TAG    {MODEL_TAG}")
    print(f"CACHE_DIR    {CACHE_DIR}")
    recs = discover_sims()
    from collections import Counter
    c = Counter(r["suite"] for r in recs)
    print("available sims:", dict(c), "total", len(recs))


if __name__ == "__main__":
    describe()
