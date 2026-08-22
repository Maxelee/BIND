"""Shared scaffolding for the fidswap figure re-renders.

Everything here is lifted VERBATIM from papers/01_pipeline/_build_figures_nb.py's
setup cell (rebin / load_maps_prefix / dmo_paired_cl / running_stat /
ell_trust_marker / ELL_TRUST / ELL_MAX_PLOT / ZS / ZI) so the re-rendered
figures use byte-identical conventions to the shipped ones.  The ONLY thing
that changes is where the fiducial comes from:

    FID_RUN   = bind_science/runs/twobound/run_0049       (canonical replica)
    FID_FC    = referee_work/fidswap/field_cache/field_stats_fidtb49.npz
    FID_MF    = referee_work/fidswap/mf_cache/mf_nu8_snap096_fidtb49.npz
    FID_ATLAS = referee_work/fidswap/halo_atlas/fidtb49_snap096.npz
    FID_PROF  = referee_work/fidswap/profiles/perhalo_fidtb49_snap096.npz
    FID_NU05  = referee_work/fidswap/nu05_shards/sci_bind_tb49.npz

Set BIND_FID_REPLICA=tb18|tb49|tb53 to re-render against another replica.

Figures are written to /mnt/home/mlee1/BIND/imgs/<stem>_fidswap.{png,pdf}.
save_imgs() REFUSES to overwrite any file whose name lacks the _fidswap tag.
"""
from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as npf

sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/_tools")
sys.path.insert(0, "/mnt/home/mlee1/BIND/papers/01_pipeline")
from paper_style import (  # noqa: E402
    BAND_ALPHA, COLORS, ONE_COL, ONE_COL_SQ, TWO_COL, TWO_COL_TALL,
    panel_label, setup,
)

setup()
import matplotlib.pyplot as plt  # noqa: E402

CEPH = Path("/mnt/home/mlee1/ceph")
SB35, SCI, LC = CEPH / "bind_sb35", CEPH / "bind_science", CEPH / "bind_lightcone_tng"
FS = CEPH / "referee_work/fidswap"
IMGS = Path("/mnt/home/mlee1/BIND/imgs")

REPLICA = os.environ.get("BIND_FID_REPLICA", "tb49")
_RUN = {"tb18": "run_0018", "tb49": "run_0049", "tb53": "run_0053"}[REPLICA]

# ── realization campaign: 50-real tree (default) or the N=1000 campaign ──────
# BIND_CAMPAIGN=n1000 re-points the MAP-LEVEL roots at ceph/bind_n1000, whose
# layout drops the 'runs/' level.  Derived per-halo/field caches under FS are
# 50-real ONLY and have no n1000 twin: under n1000 they are set to a path that
# does not exist, so a figure that needs one dies with a clear error instead of
# silently mixing realization counts.  See papers/_tools/campaign_roots.py.
CAMPAIGN = os.environ.get("BIND_CAMPAIGN", "sci50")
if CAMPAIGN not in ("sci50", "n1000"):
    raise SystemExit(f"BIND_CAMPAIGN must be sci50|n1000, got {CAMPAIGN!r}")
_N1K = CEPH / "bind_n1000"


def _run(cat: str, name: str) -> Path:
    return (_N1K / cat / name) if CAMPAIGN == "n1000" else (SCI / "runs" / cat / name)


# ── the swap, in one place ───────────────────────────────────────────────────
FID_RUN = _run("twobound", _RUN)                 # replaces SCI/'runs/bind/run_0000'
OLD_FID = _run("bind", "run_0000")               # the retired fiducial (for before/after)
RT = _run("truth", "run_0000")                   # unaffected by the replica swap
DMO_RUN = _run("dmo", "run_0000")
TB_ROOT = (_N1K / "twobound") if CAMPAIGN == "n1000" else (SCI / "runs/twobound")
TB_PARAMS = SCI / "runs/twobound/twobound_params.npy"   # design, campaign-independent
FID_FC = FS / f"field_cache/field_stats_fid{REPLICA}.npz"
OLD_FC = SCI / "field_cache/field_stats_fid.npz"
FID_MF = FS / f"mf_cache/mf_nu8_snap096_fid{REPLICA}.npz"
FID_ATLAS = FS / f"halo_atlas/fid{REPLICA}_snap096.npz"
OLD_ATLAS = SCI / "halo_atlas/fid_snap096.npz"
TRUTH_ATLAS = SCI / "halo_atlas/truth_snap096.npz"
FID_PROF = FS / f"profiles/perhalo_fid{REPLICA}_snap096.npz"
OLD_PROF = SCI / "profiles/perhalo_fid_snap096.npz"
TRUTH_PROF = SCI / "profiles/perhalo_truth_snap096.npz"
FID_NU05 = FS / f"nu05_shards/sci_bind_{REPLICA}.npz"
OLD_NU05 = SB35 / "nu05_shards/sci_bind.npz"
TRUTH_NU05 = SB35 / "nu05_shards/sci_truth.npz"
FID_NG10 = FS / f"nu05_shards/peak_counts_ngal10_{REPLICA}.npz"

if CAMPAIGN == "n1000":
    # the n1000 field cache IS built (build_field_cache_n1000.py); the rest are
    # 50-real derived caches with no n1000 build yet -> point at a path that
    # cannot exist so a figure needing one fails loudly.
    _NO = FS / "_no_n1000_equivalent"
    _NS = CEPH / "bind_n1000/nu05_shards"
    FID_FC = CEPH / "bind_n1000/field_cache/field_stats_fid_n1000.npz"
    FID_NU05 = _NS / f"sci_bind_{REPLICA}_n1000.npz"
    TRUTH_NU05 = _NS / "sci_truth_n1000.npz"
    FID_NG10 = FID_RUN / "peak_counts_ngal10.npz"
    FID_MF = _NO / "mf_cache.npz"
    FID_ATLAS = _NO / "halo_atlas.npz"
    FID_PROF = _NO / "profiles.npz"

# ── dataset-level constants (setup cell) ─────────────────────────────────────
# the Sobol design dataset: under n1000 this is assembled from however many
# campaign runs have stats so far (bind-emulator-assemble), so the response
# bands are drawn from FEWER nodes than the 256-node 50-real set -- rebuild and
# re-render as more runs land.
DS_PATH = (CEPH / "bind_n1000/emulator_dataset_n1000.npz" if CAMPAIGN == "n1000"
           else SB35 / "emulator_dataset_nu05.npz")
_d = np.load(DS_PATH, allow_pickle=True)
ZS = _d["source_redshifts"]
ZI = 1
ELL = _d["a__suppression__ell"]
ELL_TRUST = 3.0e4
ELL_MAX_PLOT = 36864.0
ELL_LO = 100.0


def dataset():
    return _d


def ell_trust_marker(ax, label=True, fontsize=4.0):
    ax.axvline(ELL_TRUST, color="0.55", lw=0.5, ls=":", zorder=1)
    if label:
        ax.text(ELL_TRUST, 0.06, "CIC/pixel upturn\n(measured, 0.8 $\\ell_{\\rm Nyq}$)",
                transform=ax.get_xaxis_transform(), fontsize=fontsize, color="0.4",
                ha="right", va="bottom", rotation=0)


def rebin(a, k):
    n = (a.shape[-1] // k) * k
    return a[..., :n].reshape(*a.shape[:-1], n // k, k).mean(-1)


def load_maps_prefix(path, key, n, plane=None):
    """Stream realizations 0..n-1 of a (n_real, n_plane, ny, nx) cube npz."""
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as fh:
        version = npf.read_magic(fh)
        shape, _, dtype = (npf.read_array_header_1_0(fh) if version == (1, 0)
                           else npf.read_array_header_2_0(fh))
        n_avail, n_plane, ny, nx = shape
        assert n <= n_avail, f"{Path(path).name}: {n_avail} reals < requested {n}"
        p_want = None if plane is None else plane % n_plane
        plane_bytes = ny * nx * dtype.itemsize
        out = np.empty((n, n_plane, ny, nx) if p_want is None else (n, ny, nx),
                       dtype=dtype)
        for r in range(n):
            for p in range(n_plane):
                buf = bytearray()
                while len(buf) < plane_bytes:
                    chunk = fh.read(plane_bytes - len(buf))
                    assert chunk, f"truncated {key}.npy at real {r} plane {p}"
                    buf += chunk
                if p_want is None:
                    out[r, p] = np.frombuffer(bytes(buf), dtype=dtype).reshape(ny, nx)
                elif p == p_want:
                    out[r] = np.frombuffer(bytes(buf), dtype=dtype).reshape(ny, nx)
    return out


def dmo_paired_cl(n, fid_clk=None):
    """Paired-prefix DMO mean C_l^kk, (5, n_ell) -- the S(ell) denominator.

    Read-only variant of the notebook helper: the disk cache
    runs/dmo/run_0000/Cl_kappa_paired.npz already holds the per-realization
    spectra (n_real=50), so nothing is rebuilt and nothing is written into the
    read-only campaign tree.  The notebook's seed-pairing guard (per-realization
    low-ell corr > 0.99) is re-run here against the NEW fiducial's per-real clk
    when one is supplied -- that is the check that certifies the paired-50
    denominator is still valid after the swap.
    """
    if CAMPAIGN == "n1000":
        # No paired-PREFIX needed here: the campaign traced the DMO run to the
        # same 1000 realizations as the numerator, on the same seed ladder
        # (verified byte-identical config.dat across categories at realizations
        # 1/51/52/300), so the plain mean IS the seed-paired denominator.  The
        # rho>0.99 guard below exists to catch a re-seeded 550-real DMO trace
        # sitting under 50-real numerators; that mismatch cannot arise here, and
        # the per-realization DMO spectra needed to run it are not stored.
        dcl = np.load(DMO_RUN / "Cl_kappa.npz")
        n_dmo = int(dcl["cl"].shape[-1] and np.load(DMO_RUN / "kappa_maps.npz")["n_real"])
        if n != n_dmo:
            raise SystemExit(
                f"dmo_paired_cl: numerator uses {n} realizations but the n1000 DMO "
                f"mean is over {n_dmo}; means over different realization counts is "
                f"exactly the bias this helper exists to prevent. Rebuild the "
                f"numerator at {n_dmo} reals, or store per-realization DMO spectra.")
        c_auto = np.asarray(dcl["cl"])
        print(f"dmo_paired_cl: n1000 -- full {n_dmo}-realization DMO denominator, "
              f"seed-paired by construction (no prefix truncation)")
        return np.array([c_auto[i, i] for i in range(c_auto.shape[0])])

    cache = SCI / "runs/dmo/run_0000/Cl_kappa_paired.npz"
    c = np.load(cache)
    assert int(c["n_real"]) >= n, f"{cache} holds {int(c['n_real'])} < {n}"
    cl_real = c["cl_real"][:n]
    if fid_clk is not None:
        lo = (ELL >= 100) & (ELL <= 1000)
        nv = min(n, fid_clk.shape[0])
        rho = np.corrcoef(np.log(cl_real[:nv, ZI][:, lo]).mean(1),
                          np.log(fid_clk[:nv][:, lo]).mean(1))[0, 1]
        assert rho > 0.99, (
            f"dmo_paired_cl: per-realization low-ell corr(DMO, new fiducial) = "
            f"{rho:.4f} -- the swapped fiducial is NOT seed-paired with the DMO trace")
        print(f"dmo_paired_cl: seed-pairing guard PASSED, corr(DMO, {REPLICA}) = {rho:+.5f}")
    return cl_real.mean(0)


def running_stat(x, y, edges, min_n=8, boot=0):
    ib = np.digitize(x, edges)
    out = np.full((4, len(edges) - 1), np.nan)
    rng = np.random.default_rng(2)
    for i in range(1, len(edges)):
        s = y[ib == i]
        if len(s) >= min_n:
            out[:3, i - 1] = np.percentile(s, [50, 16, 84])
            if boot:
                bs = np.median(s[rng.integers(0, len(s), (boot, len(s)))], axis=1)
                out[3, i - 1] = bs.std()
    cen = 0.5 * (edges[1:] + edges[:-1])
    ok = np.isfinite(out[0])
    return cen[ok], out[0, ok], out[1, ok], out[2, ok], out[3, ok]


def save_imgs(fig, stem):
    """Write imgs/<stem>_fidswap[_n1000].{png,pdf}; never a non-_fidswap file."""
    IMGS.mkdir(exist_ok=True)
    name = f"{stem}_fidswap" + ("_n1000" if CAMPAIGN == "n1000" else "")
    assert "_fidswap" in name
    for ext, dpi in (("png", 200), ("pdf", None)):
        p = IMGS / f"{name}.{ext}"
        assert "_fidswap" in p.name, p
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02,
                    **({"dpi": dpi} if dpi else {}))
        print(f"wrote {p}")


class Tee:
    """Mirror stdout into a numbers log next to the figures."""

    def __init__(self, path):
        self.f = open(path, "w")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def tee(name):
    p = FS / f"numbers_{name}_{REPLICA}{'_n1000' if CAMPAIGN == 'n1000' else ''}.txt"
    t = Tee(p)
    sys.stdout = t
    print(f"# fidswap re-render: {name}   replica={REPLICA}  campaign={CAMPAIGN}")
    print(f"# fiducial: {FID_RUN}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return p
