# NOTICE

BIND is distributed under the MIT License (see [LICENSE](LICENSE)).
This file records third-party components BIND depends on or redistributes, and
the data whose provenance carries obligations of its own.

Licences below are as declared by each project's own package metadata; consult
the upstream project for the authoritative text.

## Redistributed data (bundled in the wheel)

These files live in `src/bind/assets/` and are installed with the package.

| file | what it is | provenance |
|---|---|---|
| `CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt` | the CAMELS SB35 parameter and seed table for the 1024 IllustrisTNG L50n512 simulations | redistributed from the CAMELS public data release |
| `SB35_param_minmax.csv` | per-parameter min/max bounds and log flags derived from the table above | derived from CAMELS SB35 |
| `wlemu_gp.npz` | fitted Gaussian-process artifact for `bind.wlemu` | fitted to statistics of convergence maps raytraced through BIND-baryonified IllustrisTNG-DMO lightcones; downstream of CAMELS |

The pretrained checkpoints published at
<https://huggingface.co/mel2260/BIND> are **not** in this repository, but they
are trained on CAMELS IllustrisTNG data and are likewise derived products of the
CAMELS suite.

**Attribution required.** If you use these assets, the released weights, or any
field BIND generates, cite:

- **CAMELS** — Villaescusa-Navarro et al. 2021, ApJ 915, 71,
  [doi:10.3847/1538-4357/abf7ba](https://doi.org/10.3847/1538-4357/abf7ba),
  [arXiv:2010.00619](https://arxiv.org/abs/2010.00619). See the
  [CAMELS citation page](https://camels.readthedocs.io/en/latest/citation.html)
  for the data-release papers relevant to the products you use.
- **CAMELS SB35** (the suite BIND is trained on) — Genel et al. 2026,
  [arXiv:2606.10038](https://arxiv.org/abs/2606.10038).
- **IllustrisTNG** — Pillepich et al. 2018, MNRAS 475, 648,
  [arXiv:1707.03406](https://arxiv.org/abs/1707.03406); public data release,
  Nelson et al. 2019, ComAC 6, 2,
  [arXiv:1812.05609](https://arxiv.org/abs/1812.05609).

**Licence intent.** The MIT License covers the *code* only. The intended licence
for the CAMELS-derived data and the released weights is **CC-BY-4.0**
(<https://creativecommons.org/licenses/by/4.0/>). The Hugging Face repository
card currently declares MIT; that is to be updated to CC-BY-4.0 to match this
statement. See the "Data & attribution" section of [README.md](README.md).

## Required runtime dependencies

| component | licence | project |
|---|---|---|
| Pylians3 (`Pylians`) — `MAS_library` CIC pixelization, `Pk_library` power spectra | MIT | <https://github.com/franciscovillaescusa/Pylians3> |
| PyTorch (`torch`) | BSD-3-Clause | <https://github.com/pytorch/pytorch> |
| Lightning (`lightning`, `pytorch-lightning`) | Apache-2.0 | <https://github.com/Lightning-AI/pytorch-lightning> |
| `torch-ema` — exponential moving average of model weights | MIT | <https://github.com/fadel/pytorch_ema> |
| NumPy | BSD-3-Clause | <https://github.com/numpy/numpy> |
| SciPy | BSD-3-Clause | <https://github.com/scipy/scipy> |
| pandas | BSD-3-Clause | <https://github.com/pandas-dev/pandas> |
| h5py | BSD-3-Clause | <https://github.com/h5py/h5py> |
| `huggingface_hub` | Apache-2.0 | <https://github.com/huggingface/huggingface_hub> |
| tqdm | MPL-2.0 and MIT | <https://github.com/tqdm/tqdm> |

## Optional dependencies

| component | licence | extra | project |
|---|---|---|---|
| GPyTorch | MIT | `wlemu-fit` | <https://github.com/cornellius-gp/gpytorch> |
| scikit-learn | BSD-3-Clause | `wlemu-fit` | <https://github.com/scikit-learn/scikit-learn> |
| mpi4py | BSD-3-Clause | `mpi` | <https://github.com/mpi4py/mpi4py> |
| Ruff | MIT | `dev` | <https://github.com/astral-sh/ruff> |
| pytest | MIT | `dev` | <https://github.com/pytest-dev/pytest> |

## Method references

BIND's training objective follows the flow-matching formulation of
Lipman et al. 2023 ([arXiv:2210.02747](https://arxiv.org/abs/2210.02747)) and the
optimal-transport-coupled variant of Tong et al. 2023
([arXiv:2302.00482](https://arxiv.org/abs/2302.00482)). These are citations, not
redistributed code.
