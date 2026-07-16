# notes: fig03_fgas_erosita

**Panels kept**: single panel, unchanged content from placeholder.

**Content mapping** (for caption pass):
- Blue band = BIND Sobol 16-84% spread (`COLORS["bind"]`, was `tab:blue`).
- Blue solid line = BIND median (`COLORS["bind"]`).
- Navy dotted line = BIND strongest-feedback edge (2.5th percentile) —
  colour unchanged (`navy`, distinct from the semantic "bind" blue so it
  reads as a distinct sub-curve, matches placeholder).
- Green solid line = Eckert+19 X-ray fit (`COLORS["secondary"]`, was
  `tab:green`).
- Red solid line = Popesso+24 eROSITA strong-feedback fit
  (`COLORS["highlight"]`, was `tab:red`).
- Black dashed horizontal = Omega_b/Omega_m cosmic baryon fraction.

**Headline result preserved**: every BIND curve (median, band, even the
strongest-feedback 2.5% edge) sits strictly above the eROSITA red curve
across the full mass range — the "0% of Sobol reaches the eROSITA
strong-feedback band" statement is visibly intact.

**Style changes from placeholder / bugs fixed**:
1. The original attempted `\%` in a `no-latex` mathtext label, which
   renders as a literal backslash-percent glyph, not `%`. Fixed to plain
   `%` (this is a rendering-mode bug fix, not a content change — the
   original notebook likely also has this artifact since it uses the
   same `plt.style.use(["science","no-latex"])`).
2. Legend moved outside the axes (`bbox_to_anchor=(1.01,1.03)`) because
   the 6-entry frameless legend, drawn inside the axes at "upper left" as
   in the original, directly overlapped the BIND band/median curves and
   the Omega_b/Omega_m dashed line at this figure's narrower `ONE_COL`
   width — content/wording/order of legend entries unchanged.
3. No panel title (dropped any implicit title; this figure never had
   one).

**No data changes.** Same parquet columns (`run,snap,z,M_tot_500,
f_gas_500`), same z-cut (0.08<z<0.45), same 8 mass-bin edges
[13.0,14.4], same run-level stack + 16/84/2.5 percentiles, same
hardcoded Eckert19/Popesso24 analytic fits.
