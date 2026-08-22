# Referee-response numerics

Every number added to `main.tex` in the 2026-08-14 revision is produced by one of
the three scripts here. All are read-only against the campaign trees on ceph and
write only into this directory (plus `../imgs/` for the figure).

```bash
PY=/mnt/home/mlee1/venvs/BIND_env/bin/python
cd analysis
$PY referee_numerics.py          # ~3 min   -> referee_numbers.json
$PY referee_nestedcv.py          # ~30 s    -> referee_nestedcv.json
$PY make_fig_detectability.py    # ~20 s    -> ../imgs/fig07_detectability.{png,pdf}
                                 #             fig_detectability_numbers.json
```

| script | referee points | what it computes |
|---|---|---|
| `referee_numerics.py` | II-1, II-2 (baseline), II-3, II-5, II-6, II-9, II-10, II-11 | detectability under three binning conventions; $\sigma_{\rm pred}$ vs survey $\sigma$; $\Delta\chi^2$; decoy path + 200-decoy best-of-$N$ null; fold-internal CV; $\theta$ baselines; replica circularity test; lightcone geometry |
| `referee_nestedcv.py` | II-4, II-2 (matched search) | nested CV of the full latent search, all seven statistics; identical search run on the 30 parameters |
| `make_fig_detectability.py` | II-1, II-9, II-13 | rebuilds the §4 opener with a self-consistent error budget and the $\sigma_{\rm pred}$ panel |

`appendixA_offsets.json` holds the mass-binned amplitude offsets of Table 4.

## Conventions that matter

- **Source plane** `ZI = 1`, i.e. $z_s = 1.0$, throughout.
- **Band-powers** $\Delta\ell/\ell = 0.15$. The native grid is 724 logarithmic
  bins ($\Delta\ln\ell = 0.0028$); mixing the two is what produced the
  inconsistent error budget the referee's II-1 exposed.
- **$S(\ell)$ denominator** is the seed-paired 50-realization DMO prefix
  (`runs/dmo/run_0000/Cl_kappa_paired.npz`), *never* the shipped 550-realization
  mean in `Cl_kappa.npz`. Using the latter breaks the mode-by-mode cosmic-variance
  cancellation and shifts the $>5\sigma$ node fraction by ~16 points.
- **Fiducial arm** is `twobound/run_0049`, the replica painted with the correct
  TNG300 cosmology, not `bind/run_0000`.
- **Covariance** comes from the `bind_n1000` TRUTH arm (1000 realizations), not
  the BIND arm, which inherits the CAMELS conditioning.

## Inputs (all read-only)

```
ceph/bind_sb35/emulator_dataset_nu05.npz            256 Sobol nodes, all statistics
ceph/bind_sb35/emulator_dataset_nu05n.npz           the LSST-Y10 shape-noise twins
ceph/bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz
ceph/bind_science/halo_atlas/{fid,truth}_snap096.npz
ceph/bind_science/runs/twobound/run_{0018,0049,0053}/Cl_kappa.npz
ceph/bind_science/runs/dmo/run_0000/Cl_kappa_paired.npz
ceph/bind_n1000/analysis/stream_stats_truth_run_0000.npz
papers/01_pipeline/figs_preview/{amplitude_sets,family_model_bundle}.npz
papers/01_pipeline/figs_preview/agnostic_lambda_results_obs.npz
```

`referee_numerics.py` asserts on entry that it reproduces the shipped per-statistic
CV $R^2$ from `papers/01_pipeline/audits/agnostic_lambda_search_obs_mf22.log` to
better than $2\times10^{-3}$; it currently agrees to $4.6\times10^{-5}$. If that
check fails, the bundle or the dataset has moved underneath the paper and nothing
downstream should be trusted.
