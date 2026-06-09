# Circular paste aperture (the standard, `r200_factor = 4.0`)

**As of June 2026 the BIND composite uses a circular paste aperture by default**
(`r200_factor = 4.0`, i.e. each generated halo patch is blended into the full-box
map within a Hann-tapered circle of radius `4 × R200c`). The previous default — a
square Hann taper covering the whole 128² patch (`r200_factor = 0.0`) — is still
available but is now the *legacy* option.

This page records why, with numbers, so the choice is auditable.

## TL;DR

The circular aperture **removes the small-scale total-matter power deficit** that
the square taper leaves, at the cost of a mild over-production at intermediate
scales. The fix is geometric: confining each (slightly over-smooth) generated
patch to a tight aperture raises its concentration and restores the high-`k` power.

## Evidence (26 CV sims, `fm_two_head`)

Total-matter suppression `P(k)/P_DMO`, BIND vs the truth hydro maps. The control
"Hydro-replaced" pastes the *true* hydro patches through the same compositing, so
it isolates the aperture/paste from the emulator.

![Square vs circular, CV](figures/circular_vs_square_cv.png)

**BIND / Truth total `P(k)` (1.0 = perfect) by `k`-band [h/Mpc]:**

| aperture | 1–5 | 5–10 | 10–20 | 20–40 | **40–70** |
|---|---|---|---|---|---|
| square (legacy) | 1.03 | 1.07 | 1.07 | 1.02 | **0.894** |
| **circular 4×R200 (default)** | 1.03 | 1.08 | 1.11 | 1.08 | **0.992** |

- Small scales (`k` 40–70): square is **−10.6%**, circular is **−0.8%**.
- The hydro-replaced control shows the square-aperture high-`k` deficit is a *model*
  issue (truth patches reach ~1.0 there; generated patches don't — the cores are
  too smooth). Circular confinement compensates geometrically.
- Trade-off: circular runs +7–11% at `k` 10–40 vs square's +2–7% — all still inside
  the ±20% band. Tune `r200_factor` (3 → 3.5 → 4) to trade intermediate vs small scales.

Reproduce: `experiments/composite_study/exp4_cv_suppression.py` (see that folder's
`FINDINGS.md` for the full study, including the scale_global and taper analyses).

## Caveats / notes

- **Per-channel gas alone:** the square taper actually looks *better* for the gas
  channel in isolation (circular makes gas blobby with empty inter-halo gaps). The
  circular win is specifically for **total matter** `P(k)`. Pick the aperture for
  the target observable.
- **Graceful fallback:** `circular_taper_weight` reverts to the square taper for any
  halo lacking a valid `R200c`, so circular is always safe to leave on.
- **R200c source:** read from the FOF catalog (`Group_R_Crit200`, kpc/h → Mpc/h).
  Cached `halo_catalog.npz` files now expose it directly (`load_halo_catalog` reads
  the legacy kpc/h `radii` key as well as the current Mpc/h `r200s`), so a repaste
  no longer needs to re-read the FOF files.
- **Revert:** pass `--r200_factor 0` to any CLI (or set `RunConfig.r200_factor = 0`)
  for the legacy square taper. A circular composite can always be rebuilt from the
  cached `generated_halos.npz` with `--repaste`, so switching is cheap and reversible.
