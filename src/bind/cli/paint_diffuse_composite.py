"""``python -m bind.cli.paint_diffuse_composite``: add a diffuse-gas approximation
outside the pasted halo regions of an existing composite.

The pasted composites carry gas/y/tau ONLY inside the halo paste apertures
(alpha > 0); ~80-85% of each slab has exactly zero gas along the line of sight.
This tool writes a variant composite where the unpainted area gets a minimal
diffuse-gas model:

    gas_diffuse = f_b * dmo_background * (1 - alpha)          [Msun/h per pixel]
    tau_diffuse = _tau_per_gas_pixel(box, npix, a_l) * gas_diffuse
    y_diffuse   = (k_B * T_diffuse / m_e c^2) * tau_diffuse   [* a_l^2 if legacy]

i.e. the cosmic baryon fraction of the DMO density, fully ionized (the
pipeline's fixed x_e = 0.88 tau convention), at a single photoionization
temperature T_diffuse (default 1e4 K).  Mass is MOVED, not added: the diffuse
gas is subtracted from channel 0 and added to channel 1, so the summed mass map
(and therefore the lensing planes / kappa) is bit-identical to the input
composite — only tau and y change.  Output npz carries only the keys stage 3
reads (composite, composite_thermo) plus bookkeeping.

    python -m bind.cli.paint_diffuse_composite \
        --composite_dir .../runs/truth/run_0000/snap_096 \
        --scale_factor 0.96738 \
        --output_dir .../tng_full_validation/composites_diffuse/snap_096
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bind.data import N_THERMO
from bind.inference.lightcone_maps import _tau_per_gas_pixel, _thermo_y_from_patches
from bind.inference.pipeline import K_B_J_PER_K, M_E_C2_J

# TNG300 cosmology (src/bind/inference/design.py TNG300_COSMO; verified against
# the live snapshot header: OmegaBaryon=0.0486, Omega0=0.3089)
F_B_DEFAULT = 0.0486 / 0.3089


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--composite_dir", type=Path, required=True,
                   help="snap dir holding pasted composite_slab{NN}.npz (read-only)")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--scale_factor", type=float, required=True,
                   help="a_l of this snapshot (from the stage1 manifest) — sets the "
                        "proper pixel area in the tau conversion")
    p.add_argument("--f_b", type=float, default=F_B_DEFAULT,
                   help=f"cosmic baryon fraction (default {F_B_DEFAULT:.5f} = "
                        "Omega_b/Omega_m for TNG300)")
    p.add_argument("--t_diffuse_K", type=float, default=1e4,
                   help="single temperature of the diffuse component [K] (default 1e4, "
                        "the photoionized-IGM value)")
    p.add_argument("--y_convention", choices=["legacy", "physical"], default="physical",
                   help="convention of the y channel being augmented; the truth "
                        "lightcone thermo is 'physical' (proper-area, truth_lightcone.py). "
                        "'legacy' (comoving area) multiplies y_diffuse by a_l^2. "
                        "The diffuse y term is ~0.3% of pasted y either way.")
    p.add_argument("--r200_factor", type=float, default=4.0,
                   help="paste-aperture geometry for the thermo-from-patches fallback "
                        "(must match the composite's own build)")
    p.add_argument("--taper_frac", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a_l = args.scale_factor
    y_per_tau = K_B_J_PER_K * args.t_diffuse_K / M_E_C2_J
    if args.y_convention == "legacy":
        y_per_tau *= a_l ** 2

    slab_files = sorted(args.composite_dir.glob("composite_slab*.npz"))
    if not slab_files:
        raise FileNotFoundError(f"no composite_slab*.npz under {args.composite_dir}")

    stats = []
    for sf in slab_files:
        d = np.load(sf)
        comp = np.asarray(d["composite"], dtype=np.float32)
        box = float(d["box_size"])
        npix = comp.shape[-1]
        # empty slabs (n_halos==0, _save_empty_slab) carry no alpha/thermo:
        # alpha=0 everywhere is the correct physics (fully unpainted slab)
        alpha = (np.asarray(d["alpha"], dtype=np.float32) if "alpha" in d.files
                 else np.zeros((npix, npix), np.float32))
        # recomposited npz always store the raw `dmo`; the dmo_aa branch is
        # future-proofing for composites written with an anti-aliased background
        dmo_bg = np.asarray(d["dmo_aa"] if "dmo_aa" in d.files else d["dmo"],
                            dtype=np.float32)

        gas_diffuse = (args.f_b * dmo_bg * (1.0 - np.clip(alpha, 0.0, 1.0))
                       ).astype(np.float32)

        out = comp.copy()
        out[0] -= gas_diffuse          # move, don't add: kappa stays identical
        out[1] += gas_diffuse

        tau_per_pix = _tau_per_gas_pixel(box, npix, a_l)
        y_diffuse = (y_per_tau * tau_per_pix * gas_diffuse).astype(np.float32)

        if "composite_thermo" in d.files:
            y_pasted = np.asarray(d["composite_thermo"][0], dtype=np.float32)
        elif "thermo_patches" in d.files:
            y_pasted = _thermo_y_from_patches(
                d, r200_factor=args.r200_factor, taper_frac=args.taper_frac
            ).astype(np.float32)
        else:
            y_pasted = np.zeros((npix, npix), np.float32)   # empty slab
        thermo = np.zeros((N_THERMO, npix, npix), dtype=np.float32)
        thermo[0] = y_pasted + y_diffuse

        np.savez_compressed(
            args.output_dir / sf.name,
            composite=out, composite_thermo=thermo,
            box_size=np.float64(box),
            slab_idx=d["slab_idx"] if "slab_idx" in d.files else np.int64(len(stats)),
            n_slabs=np.int64(len(slab_files)),
            scale_factor=np.float64(a_l), y_convention=str(args.y_convention),
        )
        cov = float((alpha > 0.01).mean())
        stats.append({
            "slab": sf.name, "coverage_pct": round(100 * cov, 2),
            "gas_pasted_msunh": float(comp[1].sum()),
            "gas_diffuse_msunh": float(gas_diffuse.sum()),
            "y_pasted_mean": float(y_pasted.mean()),
            "y_diffuse_mean": float(y_diffuse.mean()),
        })
        print(f"[diffuse] {sf.name}: coverage {100*cov:.1f}%  "
              f"gas pasted {comp[1].sum():.3e} + diffuse {gas_diffuse.sum():.3e} Msun/h  "
              f"<y> pasted {y_pasted.mean():.3e} + diffuse {y_diffuse.mean():.3e}")

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump({
            "kind": "diffuse_gas_composite",
            "composite_dir": str(args.composite_dir),
            "f_b": args.f_b, "t_diffuse_K": args.t_diffuse_K,
            "scale_factor": a_l, "y_convention": args.y_convention,
            "slabs": stats,
        }, f, indent=2)


if __name__ == "__main__":
    main()
