# Small-items audit — 2026-08-03

Five self-contained investigations into the Paper I pipeline figure package
(`papers/01_pipeline/_build_figures_nb.py`). Each section gives the finding,
the file:line evidence, numeric verification, and a RECOMMENDED EDIT block
(exact old-string/new-string, ready for the orchestrator to apply — I did not
edit `_build_figures_nb.py` myself).

Scripts written for this audit (both run clean, output reproduced below):
- `papers/01_pipeline/audits/tau_constant_check.py` (item 1)
- `papers/01_pipeline/audits/map_rescale_check.py` (item 5)

---

## 1. TAU-CONVERSION PIN [§1d]

### Finding

The τ formula `tau = sigma_T * (x_e/m_p) * Sigma_gas` with `x_e = 0.88` encodes:

| assumption | value | source |
|---|---|---|
| Hydrogen mass fraction | `X_H = 0.76` | derived (see below) |
| Helium mass fraction | `Y_He = 0.24` | derived (no metals: `X_H+Y_He=1`) |
| Ionization state | **fully ionized**: H → p⁺+e⁻ (1 e⁻/proton-mass), He → α²⁺+2e⁻ (doubly ionized) | derived |
| ⇒ free electrons / proton mass | `x_e = X_H + Y_He/2 = 0.88` | matches hardcoded `X_E_PER_MASS = 0.88/M_P` |
| Thomson cross-section | `σ_T = 6.6524e-25 cm²` | `lightcone_maps.py:42` (CODATA 6.6524587e-25, 5-digit truncation) |
| proton mass | `m_p = 1.6726e-24 g` | `lightcone_maps.py:43` (CODATA 1.67262192e-24, 5-digit truncation) |
| Surface density convention | **physical** (proper-area), not comoving | `_tau_per_gas_pixel`, `lightcone_maps.py:52-57` |
| h-handling | mass `[Msun/h]→[Msun]` divides by `h`; comoving pixel length → physical multiplies by scale factor `a_l` (physical = comoving × a), then → cm; net: the *product* `K_TAU × gas[Msun/h]` is h-independent, `K_TAU` alone is not | derived, see script |

**x_e = 0.88 derivation:** for a fully ionized H+He plasma with primordial
composition (no metals), `n_e/ρ = X_H/m_p + (Y_He/4)·2/m_p = (X_H + Y_He/2)/m_p`
(He nucleus ≈4 m_p, doubly ionized ⇒ 2 e⁻ per nucleus). With `X_H=0.76,
Y_He=0.24`: `0.76 + 0.12 = 0.88`. Equivalently `μ_e = 2/(1+X_H) ≈ 1.1364`,
`x_e = 1/μ_e = 0.88` — the standard cosmological mean-molecular-weight-per-
electron convention.

**Grep results** (`src/bind/cli/paint_tauplane.py`, `src/bind/inference/lightcone_maps.py`):
```
lightcone_maps.py:42-49
    SIGMA_T = 6.6524e-25            # Thomson cross-section [cm^2]
    M_P = 1.6726e-24                # proton mass [g]
    MSUN_G = 1.989e33               # solar mass [g]
    MPC_CM = 3.0857e24              # Mpc -> cm
    PC_CM = 3.0857e18               # pc -> cm
    X_E_PER_MASS = 0.88 / M_P       # free electrons per gram (X=0.76, Y_He=0.24, ionized)
    HUBBLE_H = 0.6774               # h (manifests carry no hubble; matches tau_profiles/morphology)
    TAU_PER_DM = SIGMA_T * PC_CM    # tau = TAU_PER_DM * DM[pc/cm^3];  DM = tau / TAU_PER_DM

lightcone_maps.py:52-57  (_tau_per_gas_pixel — the actual per-plane pipeline formula)
    def _tau_per_gas_pixel(box_size, n_grid, a_l):
        pix_phys_cm = (box_size / n_grid) / HUBBLE_H * a_l * MPC_CM
        return SIGMA_T * X_E_PER_MASS * MSUN_G / HUBBLE_H / pix_phys_cm ** 2
```
`paint_tauplane.py:10-16,52-56` calls this same function per lux tau-plane (it
is the "lux-RT counterpart of the Born `tau`" already built by
`lightcone_maps.assemble_lightcone`) — i.e. **the same one formula**, not a
separate hardcode.

**IMPORTANT CORRECTION to the investigation's framing:** `2.219785e-14` is
**not** a hardcoded pipeline constant — it does not appear anywhere in
`paint_tauplane.py` or `lightcone_maps.py`. It only appears in
`papers/01_pipeline/_build_figures_nb.py` (the fig-6 cell, ~line 923,
`K_TAU = SIGMA_T*(0.88/M_P)*MSUN_G/(HH*(PIX_MPCH*a_snap/HH*MPC_CM)**2)`) and in
`FIGURE_NUMBERS.md:50`. It is `_tau_per_gas_pixel`'s formula evaluated at ONE
specific snapshot's geometry — snap 96, patch box 6.25 Mpc/h / 128 px,
`a = 0.96738` (`z = 0.033724`). Every other lens plane gets its own value from
the same formula at its own `(box_size, n_grid, a_l)`.

Also found: `TAU_PER_DM = σ_T·pc_cm = 2.0527e-6` is a **different** quantity
(τ per unit DM in pc/cm³, used by `stats.py`'s FRB-DM conversion and
`cli/lightcone_maps.py`) — not to be confused with the gas-surface-density
constant above. It is not currently referenced anywhere in the current
`_build_figures_nb.py` draft.

### Numeric verification (`audits/tau_constant_check.py`)

```
x_e derived from X_H=0.76, Y_He=0.24, H singly + He doubly ionized: x_e = X_H + Y_He/2 = 0.88
equivalent mu_e = 2/(1+X_H) = 1.136364  ->  x_e = 1/mu_e = 0.880000
K_TAU = SIGMA_T * (x_e/m_p) * MSUN_G / h / pix_phys_cm^2
      = 2.2197850641e-14
notebook-printed constant: 2.219785e-14
relative difference: 2.889e-08
VERDICT: matches to all 6 printed significant figures. PASS.
```

### RECOMMENDED EDIT (§1d markdown, next to the τ integral)

File: `papers/01_pipeline/_build_figures_nb.py`

```
OLD (lines 595-597):
$$\tau=\sigma_T\,\frac{x_e}{m_p}\,\Sigma_{\rm gas},\qquad x_e=0.88,$$

with per-plane physical-area factors applied at painting time. The Compton-$y$ channel is

NEW:
$$\tau=\sigma_T\,\frac{x_e}{m_p}\,\Sigma_{\rm gas},\qquad x_e=0.88,$$

$x_e=0.88$ is the free-electron-per-proton-mass factor for a **fully ionized, primordial
(no-metal) plasma**: hydrogen mass fraction $X_H=0.76$, helium mass fraction $Y_{\rm
He}=0.24$, hydrogen singly and helium **doubly** ionized, so $x_e=X_H+Y_{\rm He}/2=0.88$
(equivalently a mean molecular weight per electron $\mu_e=2/(1+X_H)\simeq1.136$).
$\Sigma_{\rm gas}$ is the **physical** (proper-area, not comoving) gas surface density —
the composited comoving gas mass ($M_\odot/h$ per pixel) is converted to a physical column
using that plane's own scale factor $a_l$ (physical pixel area $\propto a_l^2/h^2$) before
the Thomson factor is applied, so the released $\tau$ carries no residual $h$ dependence.
$\sigma_T=6.6524\times10^{-25}\,\mathrm{cm^2}$, $m_p=1.6726\times10^{-24}\,\mathrm{g}$
(`bind.inference.lightcone_maps.{SIGMA_T,M_P,X_E_PER_MASS}`). This one formula is
re-evaluated at every plane's own $(box\_size, n_{\rm grid}, a_l)$ — it is not a single
frozen constant — and, e.g., at snap 96 ($a=0.96738$, fig 6) reduces to
$\tau=2.219785\times10^{-14}\times\Sigma_{\rm gas}\,[M_\odot h^{-1}\,\mathrm{px}^{-1}]$.

with per-plane physical-area factors applied at painting time. The Compton-$y$ channel is
```

---

## 2. SCALING-RELATIONS SAMPLE DEFINITION [§2a, fig 3]

### Finding

Data: `bind_science/halo_atlas/fid_snap096.npz` (`M_fof` key — verified elsewhere
in the notebook, lines 723-724, to be the SO mass $M_{200c}$, a pure relabel).

```
N halos:            2933
log10 M200c range:  13.00 - 14.98   (M200c = 1.00e13 - 9.63e14  Msun/h)
N(M200c > 1e14):    149   (5.1% of the sample)
N(M200c > 3e14):    14    (0.48%)
N(M200c > 5e14):    5     (0.17%)

0.2-dex bin counts (the same bins fig 3 uses, edges = np.arange(13.0, 14.8, 0.2)):
  [13.0,13.2) N=1162   [13.2,13.4) N=732   [13.4,13.6) N=446   [13.6,13.8) N=294
  [13.8,14.0) N=150     [14.0,14.2) N=88    [14.2,14.4) N=32    [14.4,14.6) N=23
  [14.6,14.8) N=3       (above 14.8, unbinned: 3 more, up to the max 14.98)
```

**Upper-end sparsity:** the top populated 0.2-dex bin holds only 3 halos, and
only 6 halos in total lie above $\log M=14.6$ ($\sim4\times10^{14}\,M_\odot/h$)
— the highest-mass end of fig 3's binned medians is effectively a low-number
statistic, consistent with the existing markdown's own note (a few paragraphs
down) that "the top two bins hold 32 and 23 halos" for the $M_\star$ panel
specifically (those are the $[14.2,14.4)$/$[14.4,14.6)$ bins — i.e. even before
reaching the truly sparse $[14.6,14.8)$ tail).

### RECOMMENDED EDIT (§2a markdown, fig 3 intro paragraph)

File: `papers/01_pipeline/_build_figures_nb.py`

```
OLD (lines 665-666):
($\log M=13.0$–14.8); the light points are the individual painted halos. The stacked
*radial* profiles that used to occupy columns (c,d) now have their own figure (fig 6).

NEW:
($\log M=13.0$–14.8); the light points are the individual painted halos. The stacked
*radial* profiles that used to occupy columns (c,d) now have their own figure (fig 6).
The sample spans $\log_{10}M_{200c}=13.00$–$14.98$ ($10^{13}$–$9.6\times10^{14}\,M_\odot/h$);
it thins rapidly at the high-mass end (149 halos above $10^{14}$, 14 above $3\times10^{14}$,
only 6 above $4\times10^{14}\,M_\odot/h$), so the top one or two 0.2-dex bins in every panel
are low-number statistics, not populous cluster stacks.
```

---

## 3. PROFILE-WINDOW TRIM REASON [§2a′, fig 6]

### Finding

Data: `bind_science/profiles/{fid,truth}_snap096.npz`, built by
`examples/profiles.py` (branch `analysis/wl-tsz-bridge`, `git show
analysis/wl-tsz-bridge:examples/profiles.py`). Constants there: `PIX =
6.25/128.0` Mpc/h comoving per patch pixel, `R500_FAC = 0.659`, `BG_ANN =
(2.5, 3.0)` Mpc/h — a **fixed physical (transverse) annulus**, the same for
every halo regardless of mass, used to background-subtract each patch
"observer-style" before stacking (`bg = patch[bg_mask].mean()`).

**Inner cut (0.148 r200c): verified as the pixel-resolution floor.** Using
the actual per-halo `r200` from `halo_atlas/fid_snap096.npz` binned into the
low-mass bin ($2.5\times10^{13}$–$6\times10^{13}\,M_\odot/h$, $N=728$):
median $r_{200}=0.543$ Mpc/h, so 1 patch pixel = $\mathrm{PIX}/r_{200} =
0.0898\,r_{200c}$ — matches the existing markdown's "$1\,\mathrm{px}=0.09\,
r_{200c}$" claim exactly. The first plotted radial bin (0.148 r200c) is
therefore $\approx1.65$ pixels in — genuinely resolution-limited for this bin
(for the high-mass bin, median $r_{200}=1.008$ Mpc/h, 1 px $=0.048\,r_{200c}$,
so it is not the binding constraint there).

**Outer cut (1.855 r200c): confirmed as a hand-set threshold, with one
precision correction.** The cell hardcodes `rok = (r200x >= 0.145) & (r200x
<= 1.90)` against the actual 18-point log-spaced radial grid, which lands on
0.148 and 1.855 after intersection. I checked, bin-by-bin, exactly where each
of the 4 stacked fields (y, gas/τ, star, DM) first goes non-positive for both
the low-mass (N=728) and high-mass (N=59) bins:

```
low-mass bin (median r200=0.543 Mpc/h):  prof_y first negative at r200x=2.335 (idx 16)
                                          gas, star, DM: never negative in range
high-mass bin (median r200=1.008 Mpc/h): prof_y (BIND) negative already at r200x=1.855 (idx 15)
                                          prof_star (BIND+truth) negative already at r200x=1.855 (idx 15)
                                          gas, DM: first negative at r200x=2.935 (idx 17)
```

So the printed window's right edge (1.855, idx 15) is **not quite** "the last
radius where all fields stay positive on both sides" as the code comment
states — the sparse high-mass bin's $y$ and $\star$ legs are *already*
crossing zero right at that edge (the true universal-positivity edge is one
grid point earlier, $r/r_{200c}=1.474$). This is harmless in practice: the
cell's own per-panel mask (`m = rok & isfinite & (yb>0) & (yt>0)`, at line
~944) already silently drops exactly those two marginal points from the plot
— but the *code comment and markdown claim* overstate the precision. The
general physical picture is correct: the fixed background annulus (2.5–3.0
Mpc/h) sits close to the 3.125 Mpc/h patch half-width (only 0.125 Mpc/h /
2.6 px of margin), so once a halo's profile approaches that annulus the
already-background-subtracted signal is small and noisy, crossing zero
earliest for the sparser, noisier high-mass stack.

### RECOMMENDED EDIT (§2a′ markdown — precision fix, optional but accurate)

File: `papers/01_pipeline/_build_figures_nb.py`

```
OLD (lines 855-858):
concentration.) The plotted window runs $0.15\lesssim r/r_{200c}\lesssim1.9$: the inner cut
is the pixel-resolution floor of the lower mass bin (1 px $=0.09\,r_{200c}$ there) and the
outer cut is where the stacks reach the patch edge and background subtraction drives the
profiles negative. Profiles are surface densities per $0.0488\,h^{-1}$Mpc patch pixel.

NEW:
concentration.) The plotted window runs $0.15\lesssim r/r_{200c}\lesssim1.9$: the inner cut
is the pixel-resolution floor of the lower mass bin (1 px $=0.09\,r_{200c}$ there, verified:
median $r_{200}=0.543\,h^{-1}$Mpc in that bin) and the outer cut is a hand-set bound just
past where the sparser, noisier $N=59$ high-mass stack's $y$ and $\star$ legs first cross
zero (the fixed $2.5$–$3.0\,h^{-1}$Mpc background annulus leaves only $0.125\,h^{-1}$Mpc of
margin to the $3.125\,h^{-1}$Mpc patch half-width, so the already background-subtracted
signal is small and noisy there). Profiles are surface densities per $0.0488\,h^{-1}$Mpc
patch pixel.
```

Optional companion fix to the **code comment** (not user-facing, precision only):

```
OLD (lines 932-936):
# inner cut = pixel-resolution floor of the lower plotted bin (1 px = 0.090
# r200c there); outer cut = last radius where all three fields stay positive on
# both sides -- beyond ~2 r200c the stacks hit the patch edge and the
# background subtraction drives them negative.
rok = (r200x >= 0.145) & (r200x <= 1.90)

NEW:
# inner cut = pixel-resolution floor of the lower plotted bin (1 px = 0.090
# r200c there); outer cut is HAND-SET (not derived from a "last positive bin"
# query -- checked: the high-mass (N=59) bin's y/star legs are already
# marginally negative right at r200x=1.855, one grid point past the true
# all-positive edge at 1.474; those two points are silently dropped by the
# per-key (yb>0)&(yt>0) mask below, which is why the figure still looks clean)
# -- beyond ~2 r200c the stacks hit the patch edge and the background
# subtraction drives them negative.
rok = (r200x >= 0.145) & (r200x <= 1.90)
```

---

## 4. F-TILDE-BAR DEFINITION PIN [§3c, fig 20]

### Finding

**Operational definition** (equation-style):

$$
\tilde f^{\,\rm cyl}_{\rm bar,500c}(\text{node}, \text{mass bin}) \;=\;
\frac{1}{\Omega_b/\Omega_m}\;\underset{h\,\in\,\text{bin}}{\mathrm{median}}\;
\frac{m_{\rm gas,500c}^{\rm bg}(h) + m_{\star,500c}(h)}{m_{\rm tot,500c}^{\rm bg}(h)}
$$

with, per halo $h$:

- **Aperture:** a **projected cylinder**, transverse radius $R_{500c}=0.659\,r_{200}$
  (comoving, `R500_FAC=0.659`), line-of-sight depth = the **full lens-plane
  slab**, $51.25\,h^{-1}$Mpc (±$25.625\approx25.6\,h^{-1}$Mpc from the halo
  center — matches §1d's "four slabs of $51.25\,h^{-1}$Mpc depth"), applied as
  a hard circular mask on the $128\times128$ composite patch.
- **Masses entering:** `m_gas_500c_bg` (gas, background-subtracted) +
  `m_star_500c` (stellar, **not** background-subtracted — only the gas and
  total columns get a `_bg` variant) over `m_tot_500c_bg`
  (= DM+gas+star sum, background-subtracted). "Baryons" = gas + stars; DM is
  excluded from the numerator by construction.
- **Background subtraction:** a fixed **transverse** annulus $2.5$–$3.0\,h^{-1}$Mpc
  on the same projected patch (`BG_ANN`), mean per-pixel $\Sigma$ there ×
  aperture pixel count, subtracted "observer-style" — the *same* annulus used
  by fig 6's profile reduction and by fig 3's halo atlas (same reducer).
- **Normalization:** $\Omega_b/\Omega_m = 0.0486/0.3089$ (TNG/Planck-2015
  cosmology), `_build_figures_nb.py:729` (`OB_OM`), applied identically at
  fig 3 (line 729) and fig 20 (line 3387/3488/3530).
- **Mass binning:** by $\log_{10}m_{\rm tot,500c}^{\rm bg}$ (the
  background-subtracted total projected mass, **not** $M_{200c}$/`M_fof`),
  edges `MEDG = [13.0, 13.3, 13.6, 13.9, 15.5]` (`_build_figures_nb.py:3376`).
  A node/bin cell requires $\geq5$ halos to enter (`binned_ftilde`, line
  ~3379-3386) — otherwise NaN.
- **Sample:** the shared 2933-halo atlas (all 256 Sobol nodes' masses at the
  *same* halos, `bind_sb35/analysis_cache/atlas_cubes/atlas_cube_snap096.npz`
  → `sobol_m_{tot,gas}_500c_bg`, `sobol_m_star_500c`).

**Provenance/evidence:**
```
_build_figures_nb.py:3369-3373   mt5/mg5/ms5/fbar5 construction (this exact combination)
_build_figures_nb.py:3376        MEDG mass-bin edges (log10 m_tot_500c_bg)
_build_figures_nb.py:3379-3390   binned_ftilde() -- per-node median, /OB_OM, min-5-halo guard
_build_figures_nb.py:729         OB_OM = 0.0486/0.3089
_build_figures_nb.py:3354-3357   cell header: "same reducer as fig 3's atlases"
_build_figures_nb.py:3658-3659   printed caption line: "aperture = projected R500c,
                                  +-25.6 Mpc/h slab, annulus-subtracted"
analysis/sobol-sb35:examples/halo_atlas.py:42-90   the reducer itself (R500_FAC, BG_ANN,
                                  aperture mask, per-halo bg-subtracted sums; this is what
                                  built halo_atlas/*.npz and, by the fig-20 cell's own
                                  provenance note, the atlas_cube_snap096.npz Sobol cube)
```
Key excerpt (`analysis/sobol-sb35:examples/halo_atlas.py`):
```python
R500_FAC = 0.659
BG_ANN = (2.5, 3.0)          # LOS background annulus [Mpc/h]
...
ann = (rr >= BG_ANN[0]) & (rr < BG_ANN[1])
tot_pix = gen.sum(1)
sig_tot_bg = tot_pix[:, ann].mean(1)
sig_gas_bg = gen[:, 1][:, ann].mean(1)
for tag, rad in (("500c", R500_FAC * r200), ("200c", r200)):
    m = rr[None] <= rad[:, None, None]
    out[f"m_gas_{tag}"] = (gen[:, 1] * m).sum((1, 2))
    out[f"m_star_{tag}"] = (gen[:, 2] * m).sum((1, 2))
    out[f"m_tot_{tag}_bg"] = (out[f"m_dm_{tag}"]+out[f"m_gas_{tag}"]+out[f"m_star_{tag}"]
                              - sig_tot_bg * npix)
    out[f"m_gas_{tag}_bg"] = out[f"m_gas_{tag}"] - sig_gas_bg * npix
```
**Minor provenance note (not a bug):** the builder script itself locally
defines `F_B_COSMIC = 0.0490/0.3089` for its own internal print statements
(a slightly different $\Omega_b=0.0490$ vs the paper's $0.0486$) — but that
constant is never baked into the saved `.npz` columns (which are raw masses),
so it has no effect on the paper's $\tilde f^{\,\rm cyl}_{\rm bar}$, which is
computed entirely from the notebook's own `OB_OM=0.0486/0.3089` at plot time.
Flagged only for awareness if anyone re-derives $\tilde f$ from that script's
own printed diagnostics.

### RECOMMENDED EDIT (§3c ¶1 markdown)

File: `papers/01_pipeline/_build_figures_nb.py`

```
OLD (lines 3301-3304):
five geometric $\ell$-bands ($\ell=300$–$1.5\times10^4$) and $\tilde f_{\rm bar,500c}$ in
four halo-mass bins at $z\simeq0.03$ (background-subtracted projected-500c apertures, the
same reducer as fig 3's atlases; the halo side's per-bin node ranking is redshift-stable —
rank-corr $\simeq0.99$ against $z=0.18$, printed by the cell — so the low-$z$ bins proxy

NEW:
five geometric $\ell$-bands ($\ell=300$–$1.5\times10^4$) and $\tilde f_{\rm bar,500c}$ in
four halo-mass bins at $z\simeq0.03$ (background-subtracted projected-500c apertures, the
same reducer as fig 3's atlases — precisely,
$\tilde f^{\,\rm cyl}_{\rm bar,500c}=\frac{1}{\Omega_b/\Omega_m}\,\mathrm{median}_{h}
\frac{m_{\rm gas,500c}^{\rm bg}(h)+m_{\star,500c}(h)}{m_{\rm tot,500c}^{\rm bg}(h)}$
over halos in a $\log_{10}m_{\rm tot,500c}^{\rm bg}$ bin, with $m^{\rm bg}$ background-
subtracted using a fixed transverse $2.5$–$3.0\,h^{-1}$Mpc annulus on a projected cylinder
of radius $R_{500c}=0.659\,r_{200}$ and full-slab ($\pm25.6\,h^{-1}$Mpc) line-of-sight
depth, and $\Omega_b/\Omega_m=0.0486/0.3089$; the halo side's per-bin node ranking is
redshift-stable — rank-corr $\simeq0.99$ against $z=0.18$, printed by the cell — so the
low-$z$ bins proxy
```

---

## 5. MAP-RESCALING CONSISTENCY CHECK [§2b]

### Finding

Script: `papers/01_pipeline/audits/map_rescale_check.py` (run output below).
Uses `bind_science/field_cache/field_stats_fid.npz`'s `yy_bind`/`yy_truth`
(50 realizations × 724 ℓ), **total column** (`[:, -1]`) — the exact convention
the fig-4 cell itself uses (verified: the diagonal χ²/dof this script computes,
177.3, reproduces the notebook's own printed "177" for `yy` to the digit).

A uniform rescale of the BIND $y$ map by $1/1.054$ divides $C_\ell^{yy}$
everywhere by $c^2=1.054^2=1.1109$ ($+11.09\%$ flat prediction). **This does
not flatten the measured residual — it makes it worse overall:**

```
diagonal chi2/dof, full trusted range (ell=100-15000), AS MEASURED: 177.3
diagonal chi2/dof, AFTER dividing out c^2=1.1109:                  249.6
fraction of trusted ell-bins where |residual| shrinks after rescale: 44%

full-range residual (BIND/truth - 1): median +4.0%  mean +3.1%  (min -9.7%, max +13.6%)

ell in [  300,  3000]  (38 bins):  median residual BEFORE = -6.15%   AFTER /c^2 = -15.52%
                                    chi2/dof BEFORE=46.5  AFTER=353.0
ell in [ 5000, 15000]  (139 bins): median residual BEFORE = +7.60%   AFTER /c^2 = -3.14%
                                    chi2/dof BEFORE=240.5  AFTER=139.1

residual sign changes at ell ~ [238, 5504, 5577, 5648]
```

**Interpretation.** The mid-ℓ (300–3000) residual is *negative* — the
opposite sign from what a uniform +11% amplitude excess predicts — so dividing
it out makes the mismatch **worse** (chi2/dof 46 → 353), not better. The
high-ℓ (5000–15000) excess (+7.6%) is only partially aligned with the
constant-offset prediction: dividing by $c^2$ **overshoots past zero** to
−3.1% rather than flattening to ~0% (chi2/dof still 139, not ≲1). Over the
full trusted range the rescale *increases* the overall diagonal chi2/dof
(177→250) and only 44% of ℓ-bins even improve in magnitude. **The halo-level
Y-normalization offset is not the dominant driver of the map-level $C_{yy}$
discrepancy** — its sign doesn't even match at mid-ℓ. This is consistent with
a scale-dependent (texture) effect — over-textured halo interiors boosting
small-scale (high-ℓ) power, under-textured outskirts suppressing large-scale
(mid-ℓ) power — rather than a single multiplicative amplitude bias.

### RECOMMENDED EDIT (§2b markdown — new paragraph after the noise-accounting audit)

File: `papers/01_pipeline/_build_figures_nb.py`

```
OLD (lines 1019-1022):
would face against an independent simulation. This audit (and the Hartlap block) covers the
$\kappa y$/$yy$ legs of fig 4b as well; **every $\chi^2/\mathrm{dof}$ now lives in the printed
cell output rather than on the figure.**

**Survey context.**

NEW:
would face against an independent simulation. This audit (and the Hartlap block) covers the
$\kappa y$/$yy$ legs of fig 4b as well; **every $\chi^2/\mathrm{dof}$ now lives in the printed
cell output rather than on the figure.**

**Is the $yy$ offset just a normalization?** Fig 3 reports a halo-level $Y_{500c}$
BIND/truth offset of $1.054\pm0.007$; a uniform amplitude rescale by that factor would
predict a *flat*, $\ell$-independent $C_\ell^{yy}$ excess of $1.054^2-1=+11.1\%$
(`audits/map_rescale_check.py`). Dividing the measured $C_\ell^{yy}$ by that constant does
**not** flatten the residual: at mid $\ell$ (300–3000) the measured residual is itself
*negative* ($-6.2\%$ median — the opposite sign from the offset's prediction), so removing
it makes the mismatch worse ($\chi^2/\mathrm{dof}$ $46\to353$); at high $\ell$
(5000–15000) the $+7.6\%$ excess only partially aligns, overshooting to $-3.1\%$ rather
than flattening to zero. The overall diagonal $\chi^2/\mathrm{dof}$ across the trusted
range *increases* after the rescale ($177\to250$). The halo-integrated Y-normalization
offset is therefore not the map-level $C_{yy}$ discrepancy's dominant driver; its
$\ell$-dependence points to a scale-dependent (texture) effect instead.

**Survey context.**
```

---

## Summary of deliverables

| item | script | edit target | status |
|---|---|---|---|
| 1 τ constant | `audits/tau_constant_check.py` | §1d, lines 595-597 | verified to 6 sig figs |
| 2 fig-3 sample | (inline, numbers above) | §2a, lines 665-666 | N=2933, logM 13.00-14.98, top bin N=3 |
| 3 fig-6 window | (inline, numbers above) | §2a′, lines 855-858 (+ optional code comment 932-936) | inner cut confirmed; outer cut precision-corrected |
| 4 f̃_bar def | (inline, git-show evidence) | §3c, lines 3301-3304 | fully pinned, equation-style |
| 5 map rescale | `audits/map_rescale_check.py` | §2b, insert after line 1021 | rescale does not flatten; worsens overall χ² |
