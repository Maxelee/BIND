#!/usr/bin/env python3
"""THE GATE: verify the corrected fiducial repaint before anything downstream runs.

Four hard checks, each PASS/FAIL:

  (a) PARAMS      every snapshot of the new tree carries the corrected TNG300
                  vector, bit-for-bit equal to the verified-correct reference,
                  and consistent with the substrate cosmology in the manifest.
  (b) PHYSICS     the gas/tau amplitude excess is gone: mean gas surface density
                  ratio new/old ~= 1/1.038141 and gas plane POWER ratio
                  ~= 1/1.038141^2 = 0.9279 at snap_096; while the TOTAL mass
                  channel (what lensing sees, pinned by patch_mass_match) is
                  unchanged to <1%.
  (c) GEOMETRY    halo counts, centres, masses, R200 and the map geometry are
                  identical to the released tree, snapshot by snapshot (the two
                  trees share the same stage-1 cutouts, so they MUST match).
  (d) UNTOUCHED   no released tree was modified: a recorded size+mtime inventory
                  still matches, and the released params.npy still holds the OLD
                  (buggy) vector.

Usage::

    # once, BEFORE submitting anything (records the released-tree baseline):
    python repaint/verify_repaint.py --record_baseline

    # after stage 2 (generate) — the gate before planes/lux:
    python repaint/verify_repaint.py

    # structural checks only (no FFTs, seconds):
    python repaint/verify_repaint.py --skip_power

Exit status is 0 only if every non-skipped check passes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repaint_paths as rp  # noqa: E402

# ── tolerances (documented, not tuned to make a run pass) ─────────────────────
TOL_MEAN_GAS = 0.015     # |ratio/expected - 1| for the mean gas surface density
TOL_POWER_GAS = 0.050    # ... for the gas plane band power  (expected 0.927887)
TOL_MASS_POWER = 0.010   # total-mass LOW-k band power must be within 1% of unity
TOL_MASS_POWER_MID = 0.050   # ... and mid-k within 5%: this is the TAPER check
TOL_MASS_SUM = 0.002     # provenance only — see note below
# The audit measured, against three correct-cosmology replicas: gas +4.97% per
# halo, C_l^tautau +7.95%, C_l^kk -0.03% (ell<4e3) to -1.9% (ell>1e4).  The
# mass-power tolerance is deliberately applied to a LOW/MID-k band only.
#
# TOL_POWER_GAS was widened 3% -> 5% after review.  POWER_RATIO_EXPECTED = 0.92787
# assumes the correction is a flat amplitude rescale, but sigma8/n_s/h/Om move too
# and the gas SHAPE changes, so the true band ratio is k-dependent and sits a little
# BELOW the flat prediction: recompositing the three trio replicas onto the shared
# snap_096/slab00 stage-1 gave 0.9124 / 0.9231 / 0.9134 in this gate's own band
# (reviewer measurement; ~1.2% replica scatter), and 0.915 / 0.911 / 0.856 / 0.778
# in k = [20,100) / [100,300) / [300,800) / [800,1500).  At 5% the band is
# [0.881, 0.974], which comfortably contains all three while an UNCORRECTED repaint
# still lands at exactly 1.000 — 2.6% above the upper edge — so discrimination is
# unharmed.  Treat the printed value as a measurement to read, not just a threshold:
# GATE 2 should land near 0.91-0.93, and anything at 1.00 means the fix did not take.
#
# TOL_MASS_SUM is NOT evidence of anything physical.  build_bind_composite sets
# scale_global = dmo.sum()/composite.sum() and multiplies it in (pipeline.py:788-789),
# so every composite's total equals its own DMO total BY CONSTRUCTION, and the two
# trees share the same symlinked DMO cutouts.  The ratio is therefore identically 1
# whatever cosmology was painted.  It is kept as a cheap PROVENANCE assertion — it
# fires if the new tree stopped sharing the released DMO, or if patch_mass_match was
# turned off — and is labelled as such in the output.  The checks that actually move
# with the bug are the gas mean and the gas band power.


class Table:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, group: str, name: str, status: str, detail: str = "") -> None:
        self.rows.append((group, name, status, detail))

    @property
    def failed(self) -> int:
        return sum(r[2] == "FAIL" for r in self.rows)

    def render(self) -> str:
        w0 = max(len(r[0]) for r in self.rows) if self.rows else 8
        w1 = max(len(r[1]) for r in self.rows) if self.rows else 8
        out = ["", "=" * 110,
               f"{'check':<{w0}}  {'item':<{w1}}  {'status':<6}  detail",
               "-" * 110]
        for g, n, s, d in self.rows:
            out.append(f"{g:<{w0}}  {n:<{w1}}  {s:<6}  {d}")
        out.append("-" * 110)
        n_pass = sum(r[2] == "PASS" for r in self.rows)
        n_skip = sum(r[2] == "SKIP" for r in self.rows)
        out.append(f"{n_pass} PASS   {self.failed} FAIL   {n_skip} SKIP")
        out.append("=" * 110)
        return "\n".join(out)


# ── (a) params ────────────────────────────────────────────────────────────────
def check_params(t: Table, root: Path, ref: np.ndarray) -> None:
    for s in rp.SNAPSHOTS:
        d = rp.snap_dir(root, s) / "stage1"
        name = f"snap_{s:03d}"
        pp = d / "params.npy"
        if not pp.exists():
            t.add("a-params", name, "FAIL", f"missing {pp}")
            continue
        try:
            got = rp.load_params(pp)
        except Exception as e:  # noqa: BLE001
            t.add("a-params", name, "FAIL", f"unreadable: {e}")
            continue
        if not np.array_equal(got, ref):
            bad = np.nonzero(got != ref)[0].tolist()
            t.add("a-params", name, "FAIL", f"differs from reference at idx {bad}")
            continue
        try:
            man = json.loads((d / "stage1_manifest.json").read_text())
        except Exception as e:  # noqa: BLE001
            t.add("a-params", name, "FAIL", f"manifest unreadable: {e}")
            continue
        om = float(man.get("Omega_m", float("nan")))
        if not abs(got[0] - om) / om < 1e-3:
            t.add("a-params", name, "FAIL",
                  f"Omega_m {om:.4f} (header) vs params {got[0]:.4f}")
            continue
        t.add("a-params", name, "PASS",
              f"Om={got[0]:.4f} Ob={got[6]:.4f} Ob/Om={got[6] / got[0]:.6f} "
              f"z={man.get('redshift', float('nan')):.4f}")


# ── (b) physics ───────────────────────────────────────────────────────────────
def _band_power(m: np.ndarray, klo: int, khi: int) -> float:
    """Mean |FFT|^2 of the map in an annulus of integer wavenumber [klo, khi)."""
    m = np.asarray(m, dtype=np.float32)
    n = m.shape[0]
    f = np.fft.rfft2(m)
    p = (f.real.astype(np.float64) ** 2 + f.imag.astype(np.float64) ** 2)
    del f
    ky = np.fft.fftfreq(n, d=1.0 / n)[:, None]
    kx = np.fft.rfftfreq(n, d=1.0 / n)[None, :]
    kk = np.sqrt(ky ** 2 + kx ** 2)
    sel = (kk >= klo) & (kk < khi)
    return float(p[sel].mean())


def _composite(path: Path) -> np.ndarray:
    with np.load(path) as f:
        if "composite" not in f.files:
            raise KeyError(f"{path} has no 'composite' (patches-only file?)")
        return np.asarray(f["composite"], dtype=np.float32)


def check_physics(t: Table, root: Path, old: Path, snap: int, slabs: list[int]) -> None:
    exp_amp = 1.0 / rp.FB_RATIO
    exp_pow = rp.POWER_RATIO_EXPECTED
    for si in slabs:
        name = f"snap_{snap:03d} slab{si:02d}"
        pn = rp.snap_dir(root, snap) / f"composite_slab{si:02d}.npz"
        po = rp.snap_dir(old, snap) / f"composite_slab{si:02d}.npz"
        if not pn.exists():
            t.add("b-physics", name, "SKIP", "new composite not painted yet")
            continue
        if not po.exists():
            t.add("b-physics", name, "SKIP", f"released composite missing ({po})")
            continue
        try:
            cn = _composite(pn)
            co = _composite(po)
        except Exception as e:  # noqa: BLE001
            t.add("b-physics", name, "FAIL", str(e))
            continue

        npix = cn.shape[-1]
        klo, khi = 20, npix // 8          # mid-k band, away from the box and Nyquist
        klo_l, khi_l = 4, 40              # low/mid-k band for the mass channel

        # amplitude (first moment)
        r_gas_sum = float(cn[1].sum() / co[1].sum())
        r_tot_sum = float(cn.sum() / co.sum())
        r_star_sum = float(cn[2].sum() / co[2].sum())
        r_dm_sum = float(cn[0].sum() / co[0].sum())

        # band power
        p_gas = _band_power(cn[1], klo, khi) / _band_power(co[1], klo, khi)
        tot_n = cn.sum(axis=0)
        tot_o = co.sum(axis=0)
        p_tot_lo = _band_power(tot_n, klo_l, khi_l) / _band_power(tot_o, klo_l, khi_l)
        p_tot_mid = _band_power(tot_n, klo, khi) / _band_power(tot_o, klo, khi)
        del cn, co, tot_n, tot_o

        msgs = []
        ok = True
        if abs(r_gas_sum / exp_amp - 1.0) > TOL_MEAN_GAS:
            ok = False
            msgs.append(f"gas mean ratio {r_gas_sum:.4f} vs expected {exp_amp:.4f}")
        if abs(p_gas / exp_pow - 1.0) > TOL_POWER_GAS:
            ok = False
            msgs.append(f"gas band power {p_gas:.4f} vs expected {exp_pow:.4f}")
        if abs(r_tot_sum - 1.0) > TOL_MASS_SUM:
            ok = False
            msgs.append(f"[provenance] total mass sum ratio {r_tot_sum:.5f} != 1 "
                        "(shared DMO / patch_mass_match broken)")
        if abs(p_tot_lo - 1.0) > TOL_MASS_POWER:
            ok = False
            msgs.append(f"total-mass low-k power {p_tot_lo:.4f} != 1")
        # Mid-k total-mass power is the TAPER check: a square-tapered composite
        # (r200_factor=0, i.e. stage 2b skipped) moves it by tens of percent,
        # while the cosmology correction itself moves it by <2%.
        if abs(p_tot_mid - 1.0) > TOL_MASS_POWER_MID:
            ok = False
            msgs.append(f"total-mass mid-k power {p_tot_mid:.4f} != 1 — paste geometry "
                        "differs from the released tree (was stage 2b run?)")

        detail = (f"gas mean {r_gas_sum:.4f} (exp {exp_amp:.4f}), gas P {p_gas:.4f} "
                  f"(exp {exp_pow:.4f}, trio 0.912-0.923); mass sum {r_tot_sum:.5f} "
                  f"[provenance], mass P lo/mid "
                  f"{p_tot_lo:.4f}/{p_tot_mid:.4f}; DM {r_dm_sum:.4f} star {r_star_sum:.4f}")
        t.add("b-physics", name, "PASS" if ok else "FAIL",
              detail if ok else detail + "  <-- " + "; ".join(msgs))


# ── (c) geometry / halo population ────────────────────────────────────────────
_GEOM_KEYS = ("halo_centers", "halo_masses", "halo_r200")


def check_geometry(t: Table, root: Path, old: Path) -> None:
    for s in rp.SNAPSHOTS:
        name = f"snap_{s:03d}"
        try:
            man = json.loads(
                (rp.snap_dir(root, s) / "stage1" / "stage1_manifest.json").read_text())
        except Exception as e:  # noqa: BLE001
            t.add("c-geom", name, "FAIL", f"manifest unreadable: {e}")
            continue
        n_slabs = int(man["n_slabs"])
        tot_new = tot_old = 0
        msgs = []
        painted = 0
        for si in range(n_slabs):
            pn = rp.snap_dir(root, s) / f"composite_slab{si:02d}.npz"
            po = rp.snap_dir(old, s) / f"composite_slab{si:02d}.npz"
            if not pn.exists() or not po.exists():
                continue
            painted += 1
            with np.load(pn) as fn, np.load(po) as fo:
                nn, no = int(fn["n_halos"]), int(fo["n_halos"])
                tot_new += nn
                tot_old += no
                if nn != no:
                    msgs.append(f"slab{si}: n_halos {nn} != {no}")
                    continue
                if float(fn["box_size"]) != float(fo["box_size"]):
                    msgs.append(f"slab{si}: box_size differs")
                for k in _GEOM_KEYS:
                    if k in fn.files and k in fo.files:
                        if not np.array_equal(fn[k], fo[k]):
                            msgs.append(f"slab{si}: {k} differs")
                    elif k in fo.files:
                        msgs.append(f"slab{si}: missing {k}")
        if painted == 0:
            t.add("c-geom", name, "SKIP", "no painted composites to compare yet")
            continue
        if tot_new != int(man["n_halos"]) and painted == n_slabs:
            msgs.append(f"sum n_halos {tot_new} != manifest {man['n_halos']}")
        ok = not msgs
        t.add("c-geom", name, "PASS" if ok else "FAIL",
              f"{painted}/{n_slabs} slabs, {tot_new} halos (released {tot_old})"
              + ("" if ok else "  <-- " + "; ".join(msgs)))


# ── (d) released trees untouched ──────────────────────────────────────────────
#: Number of lux planes in the released tree (20 snapshots x 4 slabs).
N_PLANES = 80
#: The lenspot planes are the highest-value, highest-risk released files: 53.7 GB
#: that a single write-through-a-symlink would destroy in place.  They get a cheap
#: content anchor on top of size+mtime, because an in-place rewrite of the same
#: length by the same tool could in principle land on the same size.
_ANCHOR_BYTES = 4096


def _inventory_paths(old: Path) -> list[Path]:
    """A fixed, explicitly named file list — no recursive searching of ceph."""
    out: list[Path] = []
    for s in rp.SNAPSHOTS:
        d = old / f"snap_{s:03d}"
        out += [d / "stage1" / "params.npy",
                d / "stage1" / "fiducial_params.npy",
                d / "stage1" / "stage1_manifest.json",
                d / "summary.json"]
        out += [d / "stage1" / f"stage1_slab{si:02d}.npz" for si in range(4)]
        out += [d / f"composite_slab{si:02d}.npz" for si in range(4)]
    # THE REAL RISK SURFACE.  Everything above is read-only by construction; the
    # lensplanes are the one released asset a campaign stage can plausibly write
    # into (a symlink planted in the new tree, written through by
    # lensplane.write_* / lux).  241 explicitly named paths — still no recursion.
    lp = old / "lensplanes"
    for pre in ("lenspot", "yplane", "tauplane"):
        out += [lp / f"{pre}{p:02d}.dat" for p in range(1, N_PLANES + 1)]
    out.append(lp / "config.dat")
    out += [old / "lightcone_transforms.json",
            old / "kappa_maps.npz", old / "y_maps.npz", old / "tau_maps.npz",
            old / "Cl_kappa.npz", old / "Cl_kappa_y.npz", old / "Cl_tau.npz",
            old / "dm_stats.npz", old / "nongaussian_stats.npz",
            old / "peak_counts.npz", old / "peak_cross.npz"]
    out.append(rp.PARAMS_REFERENCE)
    return out


def _anchor_paths(old: Path) -> list[Path]:
    lp = old / "lensplanes"
    return [lp / f"lenspot{p:02d}.dat" for p in range(1, N_PLANES + 1)]


def _anchor(p: Path) -> str | None:
    """Hash of the first + last 4 KB — cheap witness that a big file's content
    did not change (size+mtime alone is a weak witness for an in-place rewrite)."""
    import hashlib
    try:
        size = os.stat(p).st_size
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            h.update(fh.read(_ANCHOR_BYTES))
            if size > _ANCHOR_BYTES:
                fh.seek(max(0, size - _ANCHOR_BYTES))
                h.update(fh.read(_ANCHOR_BYTES))
        return h.hexdigest()
    except OSError:
        return None


def _stat(p: Path) -> dict | None:
    try:
        st = os.stat(p)
    except OSError:
        return None
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def record_baseline(root: Path, old: Path) -> Path:
    rp.assert_writable(root, "baseline")
    inv = {str(p): _stat(p) for p in _inventory_paths(old)}
    anchors = {str(p): _anchor(p) for p in _anchor_paths(old)}
    root.mkdir(parents=True, exist_ok=True)
    dst = root / ".released_inventory.json"
    dst.write_text(json.dumps({"stat": inv, "anchor": anchors},
                              indent=1, sort_keys=True))
    n = sum(v is not None for v in inv.values())
    na = sum(v is not None for v in anchors.values())
    print(f"[baseline] recorded {n}/{len(inv)} released files "
          f"(+{na} lenspot content anchors) -> {dst}")
    return dst


def check_untouched(t: Table, root: Path, old: Path) -> None:
    inv_path = root / ".released_inventory.json"
    if not inv_path.exists():
        t.add("d-untouched", "inventory", "SKIP",
              f"no baseline at {inv_path} (run --record_baseline first)")
    else:
        raw = json.loads(inv_path.read_text())
        # baselines written before the lensplane inventory landed are flat dicts
        inv = raw.get("stat", raw) if isinstance(raw, dict) and "stat" in raw else raw
        anchors = raw.get("anchor", {}) if isinstance(raw, dict) else {}
        changed, gone = [], []
        for k, v in inv.items():
            now = _stat(Path(k))
            if v is None:
                continue
            if now is None:
                gone.append(k)
            elif now != v:
                changed.append(k)
        if changed or gone:
            t.add("d-untouched", "inventory", "FAIL",
                  f"{len(changed)} modified, {len(gone)} missing; first: "
                  f"{(changed + gone)[0]}")
        else:
            n_lp = sum(1 for k in inv if "/lensplanes/" in k and inv[k] is not None)
            t.add("d-untouched", "inventory", "PASS",
                  f"{sum(v is not None for v in inv.values())} released files "
                  f"unchanged (size+mtime), incl. {n_lp} lensplane files")

        if anchors:
            bad = [k for k, v in anchors.items() if v is not None and _anchor(Path(k)) != v]
            t.add("d-untouched", "lenspot content",
                  "FAIL" if bad else "PASS",
                  f"{len(bad)} lenspot planes rewritten in place; first: {bad[0]}" if bad
                  else f"{sum(v is not None for v in anchors.values())} lenspot planes "
                       "byte-anchored (first+last 4 KB) unchanged")
        else:
            t.add("d-untouched", "lenspot content", "SKIP",
                  "baseline predates the content anchor — re-run --record_baseline")

    # the released params must STILL be the buggy CAMELS vector
    p = old / "snap_096" / "stage1" / "params.npy"
    if not p.exists():
        t.add("d-untouched", "released params", "FAIL", f"missing {p}")
    else:
        v = rp.load_params(p)
        wrong = np.allclose(v[[0, 1, 6, 7, 8]], [0.3, 0.8, 0.049, 0.6711, 0.9624])
        t.add("d-untouched", "released params", "PASS" if wrong else "FAIL",
              "unchanged (still the CAMELS cosmology, as it must be)" if wrong
              else f"released vector was MODIFIED: {v[[0, 1, 6, 7, 8]]}")

    # the new tree must not live inside a released tree, and its stage-1 slabs
    # must be symlinks (a copy would silently duplicate 14.5 GiB)
    inside = rp.is_inside_released(root)
    t.add("d-untouched", "new tree path", "FAIL" if inside else "PASS",
          f"{root} is inside {inside}" if inside else f"{root} (outside every released tree)")

    n_link = n_copy = 0
    for s in rp.SNAPSHOTS:
        for si in range(4):
            q = rp.snap_dir(root, s) / "stage1" / f"stage1_slab{si:02d}.npz"
            if q.is_symlink():
                n_link += 1
            elif q.exists():
                n_copy += 1
    if n_link + n_copy == 0:
        t.add("d-untouched", "stage1 links", "SKIP", "stage-1 not linked yet")
    else:
        t.add("d-untouched", "stage1 links", "PASS" if n_copy == 0 else "FAIL",
              f"{n_link} symlinks, {n_copy} real copies (copies waste 14.5 GiB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repaint_root", type=Path, default=rp.REPAINT_ROOT)
    ap.add_argument("--old", type=Path, default=rp.LC_OLD)
    ap.add_argument("--reference", type=Path, default=rp.PARAMS_REFERENCE)
    ap.add_argument("--snapshot", type=int, default=96,
                    help="snapshot used for the physics check (default 96)")
    ap.add_argument("--slabs", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="slabs used for the physics check (default: all four). "
                         "Slab 0 of snap_096 alone is only 666 halos; because the "
                         "sampler is unseeded the new and old paints are independent "
                         "replicas (~30.6%% per-patch RMS), so a 666-halo mean carries "
                         "~1%% replica scatter against a 1.5%% tolerance. All four "
                         "slabs = 2933 halos, ~2.1x tighter, and costs nothing. Use "
                         "'--slabs 0' only as a memory-capped fallback.")
    ap.add_argument("--skip_power", action="store_true",
                    help="skip the FFT-based physics check (fast structural pass)")
    ap.add_argument("--record_baseline", action="store_true",
                    help="record the released-tree size+mtime inventory and exit")
    args = ap.parse_args()

    if args.record_baseline:
        record_baseline(args.repaint_root, args.old)
        return 0

    print("=" * 110)
    print(f"REPAINT VERIFICATION   new={args.repaint_root}   released={args.old} (read-only)")
    print("=" * 110)

    ref = rp.load_params(args.reference)
    t = Table()
    check_params(t, args.repaint_root, ref)
    if args.skip_power:
        t.add("b-physics", f"snap_{args.snapshot:03d}", "SKIP", "--skip_power")
    else:
        check_physics(t, args.repaint_root, args.old, args.snapshot, args.slabs)
    check_geometry(t, args.repaint_root, args.old)
    check_untouched(t, args.repaint_root, args.old)

    print(t.render())
    if t.failed:
        print("\nGATE: FAIL — do not proceed downstream until every check passes.")
        return 1
    print("\nGATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
