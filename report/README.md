# Observable → f_b report

Advisor-facing writeup of the Observable→f_b proof-of-concept (branch
`analysis/observable-fb-map`). Source is version-controlled; the compiled PDF and
figure PDFs are gitignored and regenerated from the cached reductions on ceph.

## Build
```bash
module load texlive/20240312                 # provides pdflatex + dvipng (usetex)
python tools/report_figures.py               # -> report/figs/*.pdf (scienceplots)
cd report && pdflatex observable_fb_report.tex && pdflatex observable_fb_report.tex
```
`tools/report_figures.py` regenerates every figure from the cached `.npz`
reductions (`/mnt/home/mlee1/ceph/sobol_ss_cv/{stacked_profiles,truth_stacked_profiles,...}.npz`
and `fm_testsuite_cube`), styled with `scienceplots` (`['science']`): no figure
titles, shared axes, no hard-coded font sizes.

## Files
- `observable_fb_report.tex` — the report (math + figures + tables).
- `figs/` — generated figure PDFs (gitignored).
- Underlying analysis: notebooks `*_fb_*.ipynb` + dated notes in `docs/`.
