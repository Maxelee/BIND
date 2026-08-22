# Diffuse-gas validation: full-hydro TNG300 trace vs the pasted-patch construction

**Status: COMPLETE, 2026-08-11.** Verdict (paired 50 reals, details in
`ceph/tng_full_validation/analysis/tng_full_validation_summary.md`):

- **τ**: pasted carries only 16–37% of the true mean column (z_s 2.44→0.5).
  The f_b·DMO·(1−α) diffuse add-on recovers the mean to **1.0–1.2%** at every
  z_s. → **folded into the N1000 campaign** (2026-08-11).
  Scale-resolved vs the full-hydro ground truth, z_s=2.44, 30 paired reals
  (measured 2026-08-14, supersedes the coarser "×2.8 above ℓ>5000" figure):

  | ℓ | pasted/full | diffuse/full |
  |---|---|---|
  | 10²–10³ | 0.850 | **1.042** |
  | 10³–5×10³ | 1.055 | 1.110 |
  | 5×10³–10⁴ | 0.465 | **1.313** |
  | 10⁴–2×10⁴ | 0.326 | **1.682** |
  | 2×10⁴–3.6×10⁴ | 0.174 | **2.070** |

  Diffuse is closer to truth at every scale except 10³–5×10³ (5% vs 11%, both
  fine). **Usable τ power: ℓ ≲ 4–5×10³** (4–11% accurate); above that the DMO
  proxy overshoots because real diffuse gas is pressure-smoothed, and beyond
  ℓ~10⁴ neither construction (nor the CIC-limited reference) is quantitative.
  Beware band-medians over "ℓ>5×10³": 81% of those bins are above 10⁴, near the
  0.29′/pixel Nyquist, so such a median reports the pixel-scale tail, not the
  science regime. Improvement path if ever needed: smooth the diffuse term with
  a filtering-scale kernel calibrated on the full-hydro trace (needs a re-trace).
- **CAP on halos (2026-08-14, `examples/cap_tau_validation.py`)**: the campaign
  maps are BETTER than the pasted ones for compensated aperture photometry —
  CAP(τ) stacked at 2662 halo positions (κ peaks ν>3, identical positions in all
  three maps, z_s=2.44, 12 reals) recovers the full-hydro truth to **0.6–3.1% at
  every aperture 0.5′–8′**, vs 5–10% errors for pasted. Pasted was too HIGH at
  large apertures (1.09 at 6–8′: its compensating annulus held no diffuse gas, so
  it under-subtracted) and too LOW at small ones (0.93–0.95 at 1–2′: the disk
  missed diffuse gas) — the diffuse term fixes both ends. This is consistent
  with, not contrary to, the Cl_ττ small-scale overshoot: that excess is diffuse
  fluctuation power spread over the 80–85% of sky *between* halos, which a
  halo-centred compensated aperture filters out. Production CAP machinery
  (`lightcone_cap_stack.py`, `cap_on_map`) lives on `analysis/ksz-desi-act-v2`;
  a real analysis wants per-halo θ200-scaled apertures and must match the
  0.29296875′/px + `xb` conventions on both sides.
- **y**: pasted misses 11–28% of the *mean* (WHIM, growing with z_s) but
  **Cl_yy agrees with full-hydro to ≤2% at ALL ℓ** — the missing WHIM y is
  smooth; fluctuation statistics are essentially unbiased. The T=10⁴ K diffuse
  term adds ~0.3% (by design). → keep pasted y; record the per-z_s mean-y
  correction factors (0.889/0.852/0.805/0.757/0.719) as calibration metadata.
- **κ**: pasted vs full-hydro Cl_κκ = 1.003/1.018/1.152 (ℓ<1e3 / 1e3–5e3 / >5e3)
  — the paste construction is validated at the percent level where WL lives.
- Convention lesson: truth-lightcone thermo is PHYSICAL (proper-area) y
  (`truth_lightcone.py`); the legacy comoving convention lives only in the
  CAMELS z≈0 code. The first hydro_full trace used legacy (preserved at
  `runs/hydro_full_legacyy`); stage-A composites were rescaled by 1/a² and
  re-traced. Mean-y ratios predicted from the planes matched the traced maps
  to 3 decimal places.

Original plan below (executed as written; runbook unchanged).

## The systematic

The truth/BIND lightcone composites paste halo patches onto DMO background slabs.
Mass (κ) gets an alpha-blend with the DMO background everywhere, but the **gas and
thermo channels have no background term**: outside the paste apertures
(`alpha = 0`, i.e. **~80–85% of every slab** — measured coverage 15.7–19.7% at
snap_096) the gas surface density, Compton-y, and τ are identically zero. τ counts
*all* electrons along the line of sight, so the traced τ maps miss most of the
diffuse/IGM electron column; y misses the WHIM contribution (halo gas dominates y,
so the y bias is smaller but nonzero). This affects every `tau_maps.npz` /
`y_maps.npz` product and everything downstream of them (kSZ/FRB-DM work is the
most exposed).

## The two tests

1. **Full-hydro ray trace (ground truth).** Project the full TNG300 hydro box
   (`/mnt/sdceph/users/sgenel/IllustrisTNG/L205n2500TNG`, all 20 lightcone
   snapshots, gas+DM+stars+BH ≈ 3×10¹⁰ particles/snapshot) through the *same*
   `lightcone_transforms.json` geometry into the native 4198²×4-slab format, as
   drop-in `composite_slab*.npz` (full mass channels + full-box Compton-y), then
   paint planes and lux-trace with the canonical base seed 1992 → per-realization
   paired κ/y/τ vs the existing truth trace.
2. **Diffuse approximation.** Rebuild the truth composites (recomposite from
   patches), then add `gas_diffuse = f_b · ρ_DMO · (1−alpha)` outside the paste
   mask, fully ionized (the pipeline's fixed x_e = 0.88 τ convention) at
   T = 10⁴ K (photoionized IGM): `τ_d` via the standard `_tau_per_gas_pixel`
   conversion, `y_d = (k_B T/m_e c²)·τ_d`. Mass is **moved** from ch0 to ch1
   (not added), so κ is bit-identical to truth — only y/τ change.

Conventions held fixed on purpose: y uses the pipeline's legacy comoving-area
convention in *all three* variants (physical y = legacy/a² per plane, scale
factors stored in the npz), so the comparison isolates the diffuse-gas effect
from the known z>0 a-factor gap. τ uses the standard proper-area conversion
everywhere. The full-hydro y uses the pipeline's per-particle physics
(T from InternalEnergy+ElectronAbundance, SFR>0 excluded); the gas *mass*
channel includes all gas (the τ convention). Verified exact on synthetic data
(per-channel mass conservation and y normalization ratio 1.000000).

## Tooling

| File | Role |
|---|---|
| `src/bind/cli/paint_project_hydro.py` | MPI-streaming full-hydro snapshot → slab projector (mirrors `paint_stages`' proven pattern; ~2 GB/rank regardless of snapshot size) |
| `src/bind/cli/paint_diffuse_composite.py` | pasted composite → +diffuse variant (`--f_b`, `--t_diffuse_K`, `--y_convention`) |
| `run_hydro_full_project.sh` | array 0–19 (one snapshot each), 2 icelake nodes / 64 ranks, ~3 h cap |
| `run_tng_validation_trace.sh` | array 0–1: task 0 = hydro_full trace, task 1 = diffuse trace (recomposites truth inline); 50 reals, seed 1992, collect+stats |
| `examples/tng_full_validation.py` | paired comparison: Cl_ττ/Cl_yy/Cl_κκ ratios, mean τ/y per z_s, τ PDF → figure + `tng_full_validation_summary.md` |

Everything writes to `/mnt/home/mlee1/ceph/tng_full_validation/`; all inputs
(sgenel's snapshots, truth patches, fiducial stage1) are read-only.

## Runbook

```bash
sbatch run_hydro_full_project.sh                 # stage A: 20 tasks, ~1-2 h each
sbatch --array=1 run_tng_validation_trace.sh     # diffuse trace (independent of A)
# after stage A completes:
sbatch --array=0 run_tng_validation_trace.sh     # full-hydro trace
# after both traces:
python examples/tng_full_validation.py           # workstation, ~10 min
```

Cost: stage A ≈ 60–120 node-hr; each trace ≈ 25 node-hr. Trivial next to the
campaign (~50k node-hr) — that is the point of running it first.

## Decision criteria for the N1000 campaign

- **τ:** if `+diffuse/full` mean-τ and low-ℓ Cl_ττ land near 1 (say within a few
  %), the diffuse add-on is a cheap, honest fix → fold
  `paint_diffuse_composite` into the campaign's paint stage for every run
  (per-run cost: seconds; the DMO background and alpha are already in each
  composite). If pasted/full is far from 1 but +diffuse/full still misses
  badly, the campaign's τ legs should wait for a better model (e.g. full-hydro
  planes exist only for truth; BIND runs would need a DMO-side surrogate — which
  is exactly what the f_b approximation is).
- **y:** the T=10⁴ K term is negligible by construction; the y gap vs full-hydro
  measures WHIM y that no pasted construction captures. If it is large at the
  Cl_yy level, that is a known limitation to state (and a per-snapshot boost
  factor could be calibrated from this very comparison), not something the
  approximation can fix.
- **κ:** pasted vs full-hydro κ quantifies the mass-channel construction error
  (independently cross-checkable against `~/hydro_replace2`'s mass-only planes).

Note for BIND (non-truth) runs in the campaign: the same diffuse add-on applies
unchanged — the composite's `alpha`/`dmo` are present in every run's recomposite,
and the diffuse term is model-independent (DMO × f_b), so it is exactly as valid
for painted BIND runs as for truth.
