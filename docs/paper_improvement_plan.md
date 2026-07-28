# P4c paper: grade report + agent-executable improvement plan

Companion to `docs/paper_rubric.md`. Two independent graders (one neutral, one
adversarial 2026-bar referee) scored `examples/paper_p4c_tsz_ksz.ipynb` (via
its builder `examples/_build_p4c_paper_nb.py`) + the decision registry +
hardening plan.

## Reconciled grade: **70 / 100** (graders: 71.2, 68.7)

| Category | weight | achieved | verdict |
|---|---|---|---|
| A. Claim & framing | 12 | 70–80% | strong abstract; **title fails the niche convention**, no novelty stake |
| B. Measurement rigor | 22 | 69–75% | Hartlap/satellites/anchor/validation strong; **nulls lack σ/PTE, bootstrap never surfaced, no per-leg GoF table** |
| C. Model confrontation | 22 | **87.5%** (both graders) | near exemplar — threshold, 2-halo, GP coverage, look-elsewhere, family scoping all present |
| D. Cosmological stakes | 14 | **50%** (both) | **no suppression-curve figure (D1=0, both graders — biggest single miss)**; S8 closure qualitative only |
| E. Figures | 15 | 67–75% | scale markers/tables/reference lines good; **no residual sub-panels anywhere (E2=0, both)** |
| F. Reusability | 15 | 50–60% | everything exists, **nothing released/packaged**; all products on private ceph paths |

Consensus reading: *a paper whose statistics would survive a hostile referee,
attached to a presentation that undersells it and a product story that leaves
the citation engines (data release, drop-in constraint, suppression figure)
unbuilt.* Ceiling after this plan ≈ 90.

---

## Improvement plan — tasks executable by Sonnet/Haiku agents

Conventions for every task:
- **The notebook is edited ONLY via its builder** `examples/_build_p4c_paper_nb.py`;
  after any change run
  `python examples/_build_p4c_paper_nb.py && jupyter nbconvert --to notebook --execute --inplace examples/paper_p4c_tsz_ksz.ipynb`
  (kernel `bind_env` is declared in metadata — do NOT pass a kernel_name
  override; the generic python3 kernel lacks scipy) and verify zero error
  outputs. That command is the acceptance gate for every notebook task.
- Products root: `KS = /mnt/home/mlee1/ceph/bind_science/ksz_confront`,
  `LC = KS/lightcone`. Never run recursive searches rooted at /mnt/home or
  /mnt/ceph; access only the exact paths given.
- Commit per phase on branch `analysis/ksz-desi-act-v2` (never main).

### Phase Q — text-only quick wins (one **Haiku** agent, single pass) ≈ +6 pts

- **Q1 (A3, +1.2–2.4)** Retitle to the field convention:
  `DESI × ACT: group gas thermodynamics from kSZ + tSZ stacking versus a 253-node IllustrisTNG feedback design`
  (keep the current title as the subtitle line). Pure edit in the title md cell.
- **Q2 (A5, +1.2)** Add one defensible novelty sentence to the abstract AND
  intro close: "the first map-level confrontation of a full (30-dimensional,
  253-node) feedback parameter design with independent kSZ and tSZ
  measurements at the same galaxy population."
- **Q3 (D3, +1.75)** In §4 Discussion, insert an itemized per-probe ledger
  (markdown table): kSZ 27/253 consistent (R6); tSZ 0/253, fid p≈3e-4 (T3);
  DES WL: no discrimination after amplitude marginalization, co-ranking
  ρ=0.78, demoted (decision #20, X.json); eROSITA/X-ray: same direction
  (lit.). All numbers already in the notebook or verdicts — cite them.
- **Q4 (C8, +1.375)** In §5 caveats, surface the measured-but-unshown splits:
  EBV<0.15 shift = 0.05–0.07σ (decision #18); z=0.45–0.9 window measured
  (T2 products) — one sentence each.
- **Q5 (E6, +1.25)** Add explicit signposts at the top of §2: "§2.1–2.4
  describe what was *measured*; §2.5 what is *modeled/assumed* on top of it."
  One sentence, mirrors the Schaan/Amodeo split.
- **Q6 (C6, +0.7)** Add the existing realization-scatter number to the §5
  shared-box caveat: cite `capmat_realization_scatter_pct` from
  `LC/verdicts/P6b.json` and Schaller et al. 2024 box-convergence.
- Acceptance: builder runs, notebook executes clean, all six edits present.

### Phase N — surfaced numbers (one **Haiku** agent, needs exact keys) ≈ +4 pts

- **N1 (B2, +1.375)** In the Fig 1 code cell, compute and print the null
  χ²/PTE inside the fit range: random null χ² = rᵀC⁻¹r with
  r=`a["random_null"]`, C=diag(`a["random_null_err"]²`) over the RAP columns
  with θ ≤ 4.5′ (and the same for `a["rotated_null"]` using the data cov
  scale); print PTE via `scipy.stats.chi2.sf`. Add the out-of-range −3σ
  footprint systematic (decision #19) to the §2.2 narrative explicitly.
- **N2 (B3, +1.375)** Surface bootstrap-vs-jackknife: `LC/T2f_lrg_z0406_cib1.7.npz`
  has both `err_boot` and `err_jk` (xb columns via `theta_kind=="xb"`).
  Print the per-column ratio in the Fig 1 cell and add one sentence in §2.2
  ("bootstrap errors agree with jackknife to X–Y%, consistent with [ratio]").
- **N3 (B7, +1.375)** Add a compact goodness-of-fit table (md or printed) to
  §3: per leg — kSZ bgs110 (fid χ² from `LC/verdicts/R6.json →
  metrics.per_sample.bgs110`, n_dof there), tSZ (fid 4.2×6, best 15.9/6,
  from T3.json), joint counts + PTEs (`scipy.stats.chi2.sf`).
- **N4 (C4, +1.375)** In §3.4, add nuisance plausibility sentences with the
  numbers: load `LC/r8_posterior.npz`, print 16/50/84 of `chain_joint[:,30]`
  (f_sat), `[:,31]` (dlogM) — compare f_sat with the DESI LRG literature
  (~0.11, Yuan+23) and dlogM with the Sailer+24 anchor uncertainty (±0.1 dex),
  mirroring the A_2h check.
- Acceptance: builder runs, notebook executes clean, printed numbers appear.

### Phase R — figure upgrades (one **Sonnet** agent) ≈ +4 pts

- **R1 (E2, +2.5)** Convert Figs 4 and 5 to two-panel layouts
  (`gridspec height_ratios=[3,1]`, shared x): bottom panel = ratio to data
  (model/data with the data's fractional error band shaded at 1). Keep all
  present styling/annotations. Watch: division by data values near zero
  outside the fit range — clip the ratio panel to the fit columns for Fig 5,
  and to θ ≤ 1.4·θ200 for Fig 4.
- **R2 (E1, +1.25)** In Fig 5, individually name ≥3 model curves: keep
  TNG fiducial + best node, and add the strongest-feedback node (lowest
  `LC/latent_constraints.npz → f_in`) as a named third curve
  ("strongest-feedback node (run NNN)"); name them in the legend.
- Acceptance: builder + clean execution; both figures regenerate with
  sub-panels; visual check that ratio panels are legible.

### Phase S — the suppression-curve figure + quantitative S8 closure
(one **Sonnet** agent — the highest-value single task) ≈ +5.25 pts

- **S1 (D1, +3.5)** New Fig 10 in a new §4.1: the WL-suppression translation.
  Data source (already computed, do NOT re-run any lightcone code):
  the suite suppression curves S(ℓ, z_s)=C_ℓ^κ(node)/C_ℓ^κ(DMO) built by
  `examples/lightcone_transfer.py` — locate its merged product by reading
  that script's output paths (non-recursive ls of the exact dirs it names;
  the paired DMO trace lives at
  `/mnt/home/mlee1/ceph/bind_science/runs/dmo/run_0000`). Plot at z_s=1:
  all 253 node curves (light), the R8 joint-posterior-weighted 68% band
  (weight nodes by importance of their U253 position under the joint chain —
  simplest correct approach: for each node compute the fraction of
  `r8_posterior.npz → chain_joint[:,:30]` samples within the node's
  Voronoi/nearest-neighbor cell in normalized parameter space, or use
  `latent_constraints.npz → w_joint` as the fallback weights and say so),
  the TNG fiducial (named), and the kSZ-consistent subset (colored).
  Mark ℓ ranges relevant to DES/LSST (ℓ~300–3000) with a shaded band.
  If the S(ℓ) product cannot be located, STOP and report — do not
  approximate it from unrelated products.
- **S2 (D2, +1.75)** Quantitative closure in §6 Conclusions: report the
  posterior-implied suppression ("the joint gas posterior maps to a
  z_s=1 κ-spectrum suppression of X% [16–84: X_lo–X_hi] at ℓ=2000")
  computed from the same weighted curves; echo the van Daalen f_gas→P(k)
  relation in prose with its citation.
- Acceptance: builder + clean execution; Fig 10 exists with named curves and
  the S2 numbers printed from data (not hardcoded).

### Phase P — products & release packaging (one **Sonnet** + one **Haiku**) ≈ +6 pts

- **P1 (F1, +1.5) [Haiku]** Write `LC/RELEASE/README_data_vector.md` +
  copy `act_ycap_lrg_real.npz` there: document every key (name, shape, units,
  convention — pull key list by loading the npz), the aperture rule, map
  variant, and the Hartlap convention. Add a 10-line python usage snippet.
- **P2 (F4, +1.5) [Haiku]** Same for the posterior:
  `LC/RELEASE/README_chains.md` documenting `r8_posterior.npz` columns
  (`chain_columns` key has the layout; names in `names`), thinning/τ numbers
  (from R8.json), and a corner-plot snippet.
- **P3 (F3, +1.5) [Sonnet]** Build the drop-in product: fit a 2-D Gaussian
  (mean + cov) to the joint gas-plane samples (`gas_fin_joint`,
  `gas_fout_joint` in `r8_posterior.npz`), save as
  `LC/RELEASE/gasplane_prior.npz` (+ 5-line usage doc in the README):
  "use as a prior on (f̃_gas(<R500), f̃_gas shell) in baryonification
  analyses." Add a short §6 sentence announcing it.
- **P4 (F2, +0.75) [Sonnet]** De-hardcode: add `--products_root` (default
  from env `BIND_KSZ_PRODUCTS`) to the two engines outsiders would run
  (`act_ycap_measure.py`, `lightcone_m2r_ycap_real.py`); add
  `docs/REPRODUCING.md` with the exact command sequence figure-by-figure.
  Do not change any default behavior for existing paths.
- **P5 (F5, +0.75) [Haiku]** Extend the notebook's reproducibility appendix
  to a figure-by-figure map: fig # → input npz/verdicts → engine script →
  release file. Pure table edit in the builder.
- Acceptance: files exist; notebook re-executes clean; README snippets run.

### Explicitly deferred (not agent-executable / needs human or new data)
- B1=2 requires an independently-produced y-map (e.g. Planck MILCA
  cross-check) — new measurement run; candidate future task, not this plan.
- True public release (Zenodo DOI, GitHub release assets) — needs the
  user's accounts; Phase P prepares everything ("release-ready").
- Multi-suite (SIMBA/Astrid) confrontation — future work §7, separate paper.

## Execution order & projected score

Q → N → R → S → P (Q/N/P1/P2/P5 are Haiku-grade; R/S/P3/P4 Sonnet-grade;
S is the single highest-value task). Phases are independent except S2
depends on S1, and P5 is best done last. Re-grade with the same two-grader
protocol after Q–S. Projected: 70 → ~86–90 (exemplar band).
