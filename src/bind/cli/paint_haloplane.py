"""``bind-paint-haloplane`` (kSZ-paper Task D companion): paint a halo-position/
mass indicator field through the identical lux geometry as the mass plane.

**Why this exists.** To CAP-stack individual FoF halos on the final ray-traced
sky maps, we need to know which pixel each halo lands on -- but
``RT_randomization``/``RT_random_seed`` govern lux's *internal*, per-realization
window/pointing choice, which is opaque outside the compiled binary (no BIND
utility inverts it; every existing per-halo reduction in this repo, e.g.
``examples/_reduce_tauy_fiducial.py``, works on the pre-trace per-halo patches,
never on the traced sky maps). Rather than reverse-engineer lux, this paints a
SECOND synthetic field -- ``log10(M200)`` inside each qualifying (>=1e12) FoF
halo's R200 disk, 0 elsewhere -- and traces it through **the same lux
invocation** (as the ``tsz_input_dir``/``yplane`` slot, alongside the mass field
in the ``tau_input_dir``/``tauplane`` slot). Since both fields share the
identical per-snapshot shift/realization geometry by construction (one lux
process, one ``RT_random_seed``), their traced outputs are pixel-aligned
automatically -- we recover each halo's sky pixel by **peak-finding directly in
the traced halo-indicator output**, with no need to invert anything.

Positions come from the FULL ``>=1e12`` FoF catalog (not just this run's own
painted >=1e13 centrals), transformed with the SAME
``bind.inference.lightcone_transforms.LightconeTransforms`` + slab assignment
used by ``examples/_reduce_fgas_lowmass.py``, so the low-mass "patch reuse"
population (down to the 1e12 floor) is included exactly as in that companion
figure.

    bind-paint-haloplane --stage1_dir /ceph/.../bind_lightcone_tng/snap_096/stage1 \\
        --output_dir /ceph/.../mass_lensplanes --lc_snap_idx 0 --lc_n_snaps 20 \\
        --transforms /ceph/.../bind_lightcone_tng/lightcone_transforms.json \\
        --fof_root /mnt/sdceph/.../L205n2500TNG_DM/output --snapshot 96
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bind.inference.lensplane import write_yplane
from bind.inference.lightcone_transforms import LightconeTransforms
from bind.inference.paint import _assign_halos_to_slabs
from bind.inference import io_gadget


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1_dir", type=Path, required=True,
                   help="Stage-1 dir for the manifest (box_size, n_slabs, npix)")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="DEDICATED mass-plane directory (yplane*.dat, alongside the massplane's tauplane*.dat)")
    p.add_argument("--lc_snap_idx", type=int, required=True)
    p.add_argument("--lc_n_snaps", type=int, required=True)
    p.add_argument("--planes_per_snapshot", type=int, default=None)
    p.add_argument("--lp_grid", type=int, default=4096)
    p.add_argument("--transforms", type=Path, required=True,
                   help="lightcone_transforms.json (the SHARED, fixed-seed geometry)")
    p.add_argument("--fof_root", required=True, help="parent dir holding groups_<snap>")
    p.add_argument("--snapshot", type=int, required=True)
    p.add_argument("--halo_mass_min", type=float, default=1e12)
    p.add_argument("--mass_field", default="Group_M_Crit200")
    p.add_argument("--disk_pix_min", type=float, default=2.0,
                   help="minimum painted-disk radius in pixels (tiny/unresolved halos)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.stage1_dir / "stage1_manifest.json").read_text())
    n_slabs = int(manifest["n_slabs"])
    box_size = float(manifest["box_size"])
    npix = int(manifest["npix"])
    pps = args.planes_per_snapshot or n_slabs
    if pps != n_slabs:
        raise SystemExit(f"--planes_per_snapshot={pps} must equal n_slabs={n_slabs}")

    tf = LightconeTransforms.load(args.transforms)
    gc = f"{args.fof_root}/groups_{args.snapshot:03d}"
    cat = io_gadget.read_fof_catalog(gc, snapshot=args.snapshot,
                                      halo_mass_min=args.halo_mass_min, mass_field=args.mass_field)
    xyz = tf.apply(cat["positions"], args.lc_snap_idx, box_size)
    slab = _assign_halos_to_slabs(xyz[:, 2], box_size, n_slabs)
    pix = box_size / npix
    logM = np.log10(cat["mass"])

    plane_offset = args.lc_snap_idx * pps + 1
    for si in range(n_slabs):
        plane_idx = plane_offset + si
        idx = np.where(slab == si)[0]
        field = np.zeros((npix, npix), dtype=np.float64)
        yy, xx = None, None
        for k in idx:
            r_pix = max(cat["r200"][k] / pix, args.disk_pix_min)
            cx, cy = xyz[k, 0] / pix, xyz[k, 1] / pix
            w = int(np.ceil(r_pix)) + 1
            x0, x1 = int(np.floor(cx - w)), int(np.ceil(cx + w)) + 1
            y0, y1 = int(np.floor(cy - w)), int(np.ceil(cy + w)) + 1
            xs = (np.arange(x0, x1) % npix)
            ys = (np.arange(y0, y1) % npix)
            gy, gx = np.meshgrid(ys, xs, indexing="ij")
            # periodic distance (halo disks are tiny vs box, so a single wrap suffices)
            dx = (np.arange(x0, x1) - cx); dx = dx - box_size / pix * np.round(dx / (box_size / pix))
            dy = (np.arange(y0, y1) - cy); dy = dy - box_size / pix * np.round(dy / (box_size / pix))
            DY, DX = np.meshgrid(dy, dx, indexing="ij")
            mask = DX ** 2 + DY ** 2 <= r_pix ** 2
            field[gy[mask], gx[mask]] = np.maximum(field[gy[mask], gx[mask]], logM[k])

        n = field.shape[0]
        if n != args.lp_grid:
            if args.lp_grid > n:
                raise SystemExit(f"--lp_grid {args.lp_grid} > map size {n}")
            off = (n - args.lp_grid) // 2
            field = field[off:off + args.lp_grid, off:off + args.lp_grid]

        out = args.output_dir / f"yplane{plane_idx:02d}.dat"
        write_yplane(field, out)
        print(f"[haloplane] plane {plane_idx:02d} (snap={args.lc_snap_idx} slab={si}) "
              f"n_halos={len(idx)} -> {out.name}")

    print(f"[haloplane] snapshot {args.lc_snap_idx} done.")


if __name__ == "__main__":
    main()
