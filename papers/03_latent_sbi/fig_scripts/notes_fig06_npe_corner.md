# notes: fig06_npe_corner

**Script**: `fig_scripts/fig06_npe_corner.py` -> `figs/fig06_npe_corner.pdf`

## Content mapping (for the caption pass)
- `corner.corner` plot, 6x6 grid, lower-triangle. Same 6 "best-constrained"
  (lowest posterior-std/prior-std "shrink") of the 30 astro parameters as
  the placeholder, selected by the identical `argsort(shrink)[:6]` rule:
  IMFslope, log WindEIn1e51erg, log VariableWindVe, log BHRadiativeEff,
  log WindFreeTravel, log WindEReduction (same order, same labels).
- Posterior color = `COLORS["bind"]` (blue; was default matplotlib "C0").
- Truth marker/lines (TNG fiducial = prior centre, u=0.5 in every
  dimension) = `COLORS["highlight"]` (red; was "C3", already red-family).
- Contour levels unchanged (68%/95%), `plot_datapoints=False` unchanged.
- Dropped `fig.suptitle("P(theta | fiducial C_l) -- 6 best-constrained
  params (red = TNG fiducial)")` -- no titles/suptitles allowed by house
  style; this sentence is exactly the caption's job and IS already in
  `main.tex`'s caption (\label{fig:corner}).

## Numbers reproduced (printed by the script)
```
IMFslope                         0.63   (main.tex text: 0.66)
WindEnergyIn1e51erg              0.70   (main.tex text: 0.72, "wind energy")
VariableWindVelFactor            0.72   (main.tex text: 0.75)
BlackHoleRadiativeEfficiency     0.82   (main.tex text: 0.83)
WindFreeTravelDensFac            0.86   (no main.tex number sourced -- flagged
                                          as unconfirmed in main.tex's own text)
WindEnergyReductionFactor        0.94   (same caveat)
mean over 30 = 0.96
```
The four main-text-sourced values are close (within 0.01-0.03) but not
bit-identical to the numbers already in `main.tex`'s prose (0.66/0.72/0.75/
0.83 vs this script's 0.63/0.70/0.72/0.82) -- both are read from the SAME
cached `npe_cl.npz` samples array (no re-run occurred; the script's `shrink`
computation is a direct, deterministic `samples.std(0)/pr`, ported verbatim
from the notebook cell), so this is most likely a transcription/rounding
difference from whichever intermediate mined-text source `main.tex`'s
numbers were pulled from (WORKLOG prose vs. this exact cache), not a
data or methodology discrepancy. The qualitative story (same 6 params, same
rank order, same 1-2sigma recovery of the fiducial) is unchanged and
visually matches the placeholder corner exactly. Flagging the small
numeric drift for the caption-integration pass in case main.tex's prose
numbers should be refreshed to this script's printed values.

## Data provenance
- `/mnt/home/mlee1/BIND/examples/wl_latent_sbi_figs/npe_cl.npz` ->
  `samples` (24000, 30), the already-materialized NPE posterior cache
  (no `sbi` training invoked).
- `/mnt/home/mlee1/ceph/bind_sb35/emulator_dataset.npz` -> `param_names`
  only (for axis labels).
- Installed package constants (not data files, not re-run):
  `bind.inference.design.ASTRO_PARAM_INDICES`/`_unit_to_native`,
  `bind.params.PARAM_LOG_FLAG`/`PARAM_MIN`/`PARAM_MAX`.
