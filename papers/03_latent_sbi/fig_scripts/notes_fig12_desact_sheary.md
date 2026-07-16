# notes: fig12_desact_sheary.pdf

## Content mapping (for the caption pass)

- 2 panels, side by side (TWO_COL), panel tags `(a)`/`(b)` added (style
  requires panel labels; the placeholder had none, only in-axes "DES bin 3 /
  4" text, which is kept).
  - (a) = DES source bin 3 ($\bar z \approx 0.74$) — was the *left* panel in
    the placeholder, unchanged.
  - (b) = DES source bin 4 ($\bar z \approx 0.94$) — *right* panel, unchanged.
- Colors changed from the notebook's ad hoc `tab:red`/`k` to the suite's
  fixed semantic palette:
  - **BIND** (Sobol band + both model curves) is now `COLORS["bind"]`
    (blue, `#0C5DA5`) — was `tab:red` in the notebook/placeholder.
  - **DES Y3$\times$ACT real data** points/errorbars are `COLORS["truth"]`
    (black, `#111111`) — same as placeholder (was already black).
  - Grey `axvspan` shading (aperture-limited / robust-window exclusion) is
    unchanged neutral grey (`"0.85"`).
- Line styles carried over unchanged: solid = BIND with the 2.4' ACT beam
  applied, dotted = BIND with no beam (exposes the intrinsic shape), filled
  band = Sobol 16-84% envelope.
- **Dropped**: the placeholder's `fig.suptitle` ("BIND shear$\times y$ vs
  REAL DES Y3$\times$ACT — ~1.5-2x high (too gas-bound) + too steep (grey =
  aperture-limited)"). FIGURE_STYLE.md forbids titles/suptitles on
  regenerated figures — that summary sentence must move into the caption
  text verbatim (it is the paper's actual conclusion for this figure, not
  decorative). Suggested caption line: *"BIND is systematically ~1.5-2x
  high in the robust aperture window (too gas-bound) and falls off too
  steeply with $\theta$ relative to the real DES Y3$\times$ACT measurement;
  grey bands mark scales excluded by beam/resolution (< 8') or field size
  (> 40') from the robust ratio quoted in each panel."*
- In-panel annotations kept verbatim: "robust $8$-$40'$: BIND/data
  $\approx$X$\times$" (recomputed here: 2.47x for bin 3, 2.29x for bin 4 —
  matches the placeholder's quoted 2.5x/2.3x to rounding).
- Legend entries (both panels, upper right, fontsize 5.6): "DES
  Y3$\times$ACT", "BIND Sobol 16-84%", "BIND (beam)", "BIND (no beam)" —
  same 4 line types as the placeholder, just recolored/relabeled slightly
  more tersely (dropped "$2.4'$" from the beam-curve legend label since the
  beam FWHM is stated once in the caption/data description instead).

## Data provenance recap

- `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz`: tomographic
  $C_\ell^{\kappa y}$ cube (253 Sobol nodes x 5 source planes x 724 $\ell$),
  scaled by the hardcoded `F_XPK = 1.2016e7` Pylians normalization constant.
- `/mnt/home/mlee1/ceph/bind_science/ksz_confront/desact_data.npz`: real DES
  Y3 shear x ACT DR6 y cross-correlation data vector, covariance, and source
  n(z) per tomographic bin (bins 3 & 4 used here, 0-indexed bi=2,3).
- Hankel transform (`scipy.special.jv`, order 2) and Gaussian-beam
  attenuation ported verbatim from notebook cell 12; no re-fitting, no
  re-running any pipeline — closed-form transform of already-cached spectra.

## Verification

- Printed robust-window BIND/data ratios: bin3 = 2.47x, bin4 = 2.29x
  (placeholder text says $\approx$2.5x / $\approx$2.3x — matches to
  rounding, confirming the regenerated figure reproduces the same
  underlying numbers).
