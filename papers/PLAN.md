# The BIND Lightcone Suite — paper production plan

Goal: turn the June-campaign results (7 git branches) into **five presentable
preprint drafts** with real figures, captions, verified citations, and complete
Intro/Methods/Results/Discussion/Conclusion structure. Drafts are for advisor
review — honest about preliminary status, never inventing content.

## Paper set (branch → paper mapping)

| # | dir | working title | source branches |
|---|-----|---------------|-----------------|
| I | `01_pipeline` | Baryon-painted weak-lensing and SZ lightcones from dark-matter-only simulations | `lightcone` + `analysis/wl-tsz-bridge` |
| II | `02_cosmo_bias` | Two nuisance parameters suffice to marginalize baryonic feedback in LSST-era cosmic shear | `analysis/wl-cosmo-bias` (+ `feature/cosmo-rescale` as Appendix) |
| III | `03_latent_sbi` | The two-dimensional latent space of baryonic feedback: what lensing×SZ statistics constrain | `analysis/sobol-sb35` (+ κ×y engines from `wl-tsz-bridge`) |
| IV | `04_ksz_gas` | Gas thermodynamics across a 30-parameter feedback space: BIND vs DESI×ACT kSZ and eROSITA | `analysis/ksz-desi-act` |
| V | `05_anisotropy` | (Letter) The anisotropy of the baryonic suppression of weak lensing | `analysis/wl-anisotropy` |

Combinations: `lightcone` core + `wl-tsz-bridge` merge into Paper I (methods +
validation + bridge is one story). `feature/cosmo-rescale` is Paper II's
"towards varying cosmology" appendix, answering that paper's main limitation.
The κ×y multiprobe results (files on `wl-tsz-bridge`) belong scientifically in
Paper III. Everything else is discrete.

## Source-material geography

- Main tree `/mnt/home/mlee1/BIND` is on branch `papers` (this branch). Write
  ONLY under `papers/<id>/`.
- Read-only worktrees of every source branch:
  `/tmp/claude-2107/-mnt-home-mlee1-BIND/08d6aa87-999a-472c-84ce-4f5d924f7238/scratchpad/wt/{lightcone,wl-tsz-bridge,wl-cosmo-bias,sobol-sb35,ksz-desi-act,wl-anisotropy,cosmo-rescale}`
- `docs/WORKLOG.md` (in this tree) = 63 dated session entries with the exact
  numbers for every result. **This is the ground truth for quantitative claims.**
- On-disk figure assets (gitignored, main tree only — NOT in worktrees):
  `examples/figures_lightcone/` (81 images, incl. subdirs A_binned216..E_fulltomo),
  `examples/figures_ksz/` (13), `examples/wl_latent_sbi_figs/` (12),
  `examples/sobol_sbi_figs/` (9), `examples/figures_field/` (8),
  `examples/sobol_latent_figs/` (5), `examples/demo_maps.png`.
- Embedded notebook figures: extract with
  `python3 papers/_tools/extract_nb_figures.py <nb.ipynb> <outdir>` (writes
  pngs + `manifest.json` with cell/heading provenance).

## Per-paper file layout (the contract)

```
papers/<id>/
  brief.md          # the assignment (already written; read first)
  dossier.md        # miner output: every result w/ numbers + provenance
  dossier.json      # same, structured
  figs_raw/         # extracted candidates (not committed)
  figs/             # final selected figures, descriptive names (committed)
  main.tex          # the paper (from papers/_tools/template.tex)
  references.bib    # verified entries only
  refs_needed.md    # writer's cite-key wishlist w/ intended reference
  leftovers.md      # in-scope-adjacent results deliberately excluded
```

## Pipeline (Sonnet agents, one chain per paper)

1. **Mine** — read brief + WORKLOG sections + notebooks/engines in the
   worktrees; write `dossier.md`/`.json`: every result with exact numbers and
   provenance (file + cell/heading, or WORKLOG date); inventory figure
   candidates (disk paths + extracted nb figures); list mandatory caveats.
2. **Draft** — write `main.tex` from template + dossier ONLY. Copy chosen
   figures into `figs/` with descriptive names; every figure gets a caption
   written from its manifest provenance; every quantitative sentence ends with
   a `% src:` comment naming its dossier entry. Cite freely with `\cite{key}`
   but log every key + intended reference in `refs_needed.md`.
3. **Cite** — resolve `refs_needed.md` into `references.bib` with entries
   verified against arXiv/ADS (alphaXiv MCP tools via ToolSearch, WebSearch).
   No entry enters the .bib without being seen in a search result. Unresolvable
   → drop the cite, leave `\todo{ref}`.
4. **Verify** (3 adversarial lenses in parallel per paper):
   numbers-vs-source · figures/captions (open the images) · citations+science
   (caveats present, abstract consistent with results).
5. **Fix** — apply confirmed findings; compile
   (`pdflatex → bibtex → pdflatex ×2`, binaries at
   `/mnt/sw/nix/store/0x72l2bb3rr6v1giw5fk1cx591xp09l7-texlive-20240312/bin/x86_64-linux/`);
   fix LaTeX errors; report page count.
6. **Coherence** (single agent, after all papers): cross-paper consistency
   (suite described identically: *256-node SB35 Sobol design, 253 usable runs*;
   shared numbers agree; series numbering/titles consistent; every paper cites
   its siblings as "in prep."), write `papers/README.md` index.

## Hard rules for every agent

- **Never invent a number, result, or reference.** If the dossier lacks it,
  write `\todo{...}` instead. Drafts with honest TODOs are the deliverable;
  fabrications are failures.
- Do NOT run analysis engines, training, sbatch/srun, or anything on GPUs.
  Figure work = copying disk files + extracting embedded nb outputs only.
- Never execute any Slurm command. Never search `/mnt/ceph` or `/mnt/home`
  outside `/mnt/home/mlee1/BIND` and the worktree dir; recursive searches get
  `timeout 60` and stay inside those roots.
- Write only inside your paper's `papers/<id>/` directory. No git commits.
- Big notebooks: don't Read whole .ipynb files; run the extract tool and read
  `manifest.json`, or list markdown headings first
  (`python3 -c "import json;[print(i,''.join(c['source'])[:80]) for i,c in enumerate(json.load(open('nb'))['cells']) if c['cell_type']=='markdown']"`).
- BIND itself (the flow-matching engine, `Lee et al. in prep.`) and the other
  suite papers are cited as *in prep.* — no fake arXiv IDs.
