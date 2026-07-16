# Unregenerated figures — figure-regeneration agent for fig04/fig05/fig06/fig07

## fig07_latent_posterior.png — left as placeholder

**Placeholder untouched**: `figs/fig07_latent_posterior.png` (still the
original notebook-extracted PNG, byte-identical to
`examples/wl_latent_sbi_figs/f3c_latent_posterior.png`). No `.pdf` was
written; `main.tex`'s `\includegraphics{figs/fig07_latent_posterior.png}`
should NOT be changed to `.pdf` for this figure.

**Why it was not regenerated**: both panels' plotted arrays do not exist as
cached files anywhere on disk today:

- Panel (a) (`chi^2` filled-contour over the 2-D suppression latent) is the
  output of `wl_latent_sbi.py`'s `latent_chi2_grid(sim_cl, Ze, rid[vR], d,
  ng=140)`, which internally (i) instantiates `EmulatorSimulator`, whose
  constructor calls `em.predict()` ~2000 times to fit a PCA reducer on
  *clean* prior draws, then (ii) calls `sim.emu_at(Uk)` (another `em.predict`
  batch, at the 253 Sobol nodes) to fit a linear latent->data-vector map,
  then (iii) evaluates that linear fit on a 140x140 grid to produce `chi2`.
- Panel (b) (NPE posterior projected onto the latent) calls a notebook-local
  `to_latent()` closure that runs `em.predict()` on ~4000 cached NPE
  posterior samples (from `npe_cl.npz`) to get their predicted suppression
  spectra, before projecting onto the SVD basis.

Both routes require **forward-evaluating the cached GP emulator checkpoint**
(`/mnt/home/mlee1/ceph/bind_sb35/emulator/lightcone_emulator_gp.pt`) several
thousand times to produce genuinely new derived arrays (the chi2 grid, the
latent-projected posterior cloud) that are not stored anywhere as a flat
`.npz`/`.npy` on disk. I confirmed no such cache exists:
`examples/wl_latent_sbi_figs/` contains `npe_cl.npz` (raw 30-d posterior
samples, used for fig06) but no `chi2`/`latent_posterior`-named array, and no
`figs_raw/*/manifest.json` entry supplies one either.

**Judgment call** (this was explicitly flagged as ambiguous in
`fig_scripts/DATA_MAP.md`'s fig07 entry): the assignment's HARD RULES state
"Re-running analysis engines, notebooks, model training, MPI, or ANY Slurm
command is FORBIDDEN" and "No cached data -> the placeholder stays." Although
the GP checkpoint itself is not *retrained* (loading it takes ~3s per its own
docstring), invoking `EmulatorSimulator`/`latent_chi2_grid`/`to_latent` to
produce the chi2 grid and the latent-projected posterior cloud IS running
`wl_latent_sbi.py`'s analysis engine to compute new science products (the
notebook's own module docstring calls this file "the engine"; §3c is
explicitly "RETHOUGHT SBI" science, not a plotting step) — not a flat read of
an existing cached array. Per the explicit hard-rule priority ("violations =
task failure") over the softer DATA_MAP language ("flagged... for the
regenerating agent to judge"), I treated this as out of scope and left the
figure as the extracted placeholder rather than execute several thousand
model forward-passes to fabricate a chi2 grid / posterior projection that
does not exist in cached form.

**What WOULD unblock this**: if a future pass runs the (already-published,
non-Slurm, CPU-only, ~seconds-scale per the docstring) `latent_chi2_grid` /
`to_latent` computation once and caches its outputs
(`E1,E2,chi2,e_map,Zk,runs_kept` for panel a; the projected `Esamp` array for
panel b) to an `.npz` under `examples/wl_latent_sbi_figs/`, this figure
becomes a straightforward two-panel `contourf`+`scatter` plot from that cache,
exactly like fig06. That caching step was out of scope for this
figure-regeneration pass (it is model inference, not a "load an existing
file" step).

No `\todo{}` edit was made to `main.tex` — a later integration agent should
add: `\todo{regenerate fig07 — needs one CPU-only forward pass of the cached
GP emulator checkpoint to build the chi2 grid / latent-projected posterior
cache; not yet run}` near line 754 of `main.tex`.
