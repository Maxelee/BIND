# fig09_progressive_stacking -- caption/content notes

## What changed vs. the placeholder
- **Panel (b) y-tick labels are now fully resolved.** main.tex's current
  caption flags that two of the ten y-tick labels in the placeholder both
  truncate to the identical string "WindEnergyReductio," and could not
  determine (from the extracted image alone) which 2 of the 3 similarly-named
  SB35 parameters (`WindEnergyReductionFactor`, `WindEnergyReductionMetallicity`,
  `WindEnergyReductionExponent`) were plotted. Loading the full
  `param_names` array from `emulator_dataset.npz` and reproducing the exact
  same `argsort(shr["C_l"])[:10]` selection resolves this unambiguously: the
  two shown are **WindEnergyReductionFactor** (rank 6) and
  **WindEnergyReductionMetallicity** (rank 8); `WindEnergyReductionExponent`
  is not in the top 10. The full (untruncated) 10-parameter order is:
  IMFslope, WindEnergyIn1e51erg, VariableWindVelFactor,
  BlackHoleRadiativeEfficiency, WindFreeTravelDensFac,
  WindEnergyReductionFactor, SNII_MinMass_Msun,
  WindEnergyReductionMetallicity, VariableWindSpecMomentum,
  QuasarThresholdPower.
  - **Caption action needed**: remove the "labeling ambiguity" sentence and
    its `\todo` in the fig09 caption; the disambiguated parameter list above
    can be cited directly if useful.

## Panel/color mapping (for caption reconciliation)
- 3 data vectors -> 3 colors, held fixed across both panels: `$C_\ell$` =
  blue (`COLORS["bind"]`), `$C_\ell$+peaks` = green (`COLORS["secondary"]`),
  `$C_\ell$+peaks+$\kappa\times y$` = red (`COLORS["highlight"]`). This
  differs from the placeholder's C0/C2/C3 (matplotlib blue/green/red) only
  in exact hex values -- same 3-way color identity.
- Black dotted vline/hline at 1.0 = prior width (no constraint).
- Panel (a) x-axis uses math-mode labels ($C_\ell$, etc.) instead of the
  placeholder's plain-text "C_l" for consistency with house style (units/
  symbols always in math mode).

## Panels kept
Both original panels kept. The in-panel titles ("stacking emulated HOS
DEGRADES the C_l constraint", "per-parameter constraint") were removed per
house style (no titles) -- the degrading-fit narrative belongs in the
caption text, which main.tex's surrounding paragraph already states
("stacking ... degrades the fit ... mean posterior shrink goes
0.97->1.01->1.03"). Measured values here: 0.961 -> 1.016 -> 1.032 (matches
the caption's rounded numbers).

## Numbers (for caption cross-check)
mean shrink: C_l = 0.961, C_l+peaks = 1.016, C_l+peaks+kxy = 1.032.
