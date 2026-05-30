# Parameters

BIND is conditioned on a **35-dim cosmology + astrophysics parameter vector**
in the CAMELS *SB35* ordering. The full table — names, fiducial values, prior
ranges, log-sampling flags, and short descriptions — is bundled with the
package at `src/bind/assets/SB35_param_minmax.csv` and loaded into `bind` as
arrays plus three convenience functions.

## Inspect the table

```python
import bind
df = bind.param_dataframe()
df.head()
```

| ParamName            | LogFlag | FiducialVal | MinVal | MaxVal | Description |
|----------------------|---------|-------------|--------|--------|-------------|
| Omega0               | 0       | 0.30        | 0.10   | 0.50   | OmegaMatter |
| sigma8               | 0       | 0.80        | 0.60   | 1.00   | sigma8 |
| WindEnergyIn1e51erg  | 1       | 3.6         | 0.9    | 14.4   | ASN1 — galactic winds energy |
| RadioFeedbackFactor  | 1       | 1.0         | 0.25   | 4.0    | AAGN1 — AGN FB kinetic mode energy |
| ...                  | ...     | ...         | ...    | ...    | ... |

`LogFlag == 1` means the parameter is sampled log10-uniformly within
`[MinVal, MaxVal]` — this matches the CAMELS Sobol design and is what
`bind.random_params` does for those entries.

## Build a parameter vector

```python
import bind

fid    = bind.fiducial_params()                                        # (35,)
random = bind.random_params(n=4, rng=0)                                # (4, 35)
hi_AGN = bind.vary_param("RadioFeedbackFactor", fraction=1.0)          # (35,)
explicit = bind.vary_params({                                          # (35,)
    "Omega0":              0.32,
    "sigma8":              0.81,
    "WindEnergyIn1e51erg": 7.2,
})
```

`vary_param(name, value=...)` overrides one parameter to a raw value;
`vary_param(name, fraction=f)` interpolates in the parameter's *native*
sampling space — log10-uniform if `LogFlag == 1`, linear-uniform otherwise —
with `f=0` → `MinVal`, `f=1` → `MaxVal`.

`random_params` accepts a `fix={name: value}` dictionary to pin some
parameters while randomizing the rest:

```python
fixed_cosmo = bind.random_params(
    n=100, rng=0,
    fix={"Omega0": 0.3, "sigma8": 0.8, "OmegaBaryon": 0.049, "HubbleParam": 0.6711},
)   # 100 random astro draws at TNG cosmology
```

## Adapting to an external cosmology

For a non-CAMELS N-body simulation, set the cosmology entries to your
simulation's cosmology — at minimum

| index | name        | typical override |
|-------|-------------|------------------|
| 0     | Omega0      | sim's Ω_m |
| 1     | sigma8      | sim's σ8 |
| 6     | OmegaBaryon | sim's Ω_b |
| 7     | HubbleParam | sim's $h$ |
| 8     | n_s         | sim's $n_s$ |

Leave the astrophysics entries at the fiducial values unless you specifically
want to explore the posterior over feedback parameters.

```python
p = bind.fiducial_params()
p[0] = 0.315          # Planck-18 Omega_m
p[1] = 0.811          # Planck-18 sigma8
p[6] = 0.0493         # Planck-18 Omega_b
p[7] = 0.6736         # Planck-18 h
p[8] = 0.9649         # Planck-18 n_s
np.save("planck18_fiducial.npy", p)
```

## Reading parameters from a CAMELS file

CAMELS ships a `CosmoAstroSeed_*.txt` file with one row per simulation. The
first column is the simulation name, the next 35 are the parameters in SB35
order, and the last is the seed.

```python
import numpy as np
params = np.loadtxt(
    "/path/to/CosmoAstroSeed_IllustrisTNG_L50n512_CV.txt",
    skiprows=1, usecols=range(1, 36),
)            # shape (N_sims, 35)
```

(The CAMELS *CV* set has a known bug: parameter index 14 is actually 0 even
though the file lists 2000 — patch it manually before passing to BIND if you
care about that entry.)
