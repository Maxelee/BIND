# Data-covariance convergence audit -- fiducial lightcone retrace

Realizations with all 21 files (incl. `config.dat`) at time of this audit: **N=477** complete out of the planned 550 (`/mnt/home/mlee1/ceph/bind_lightcone_tng/rt_output/run001..run477`; runs beyond that were still in flight / empty). Statistic: isotropic $C_\ell^{\kappa\kappa}$ of the $z_s=1$ plane (`kappa45.dat`, plane 45 -> $z_s{=}1.0$ per `PLANE_TO_ZS`), 1024$^2$ px over a 5 deg field ($\ell_{\rm fund}=72$, $\ell_{\rm Nyq}=36864$), rfft2 power binned onto 25 log bins over $\ell=100$-$1.5\times10^4$ (primary vector) and 42 log bins over the same range (secondary, for the $p{\sim}45$ context question -- a handful of the finest low-$\ell$ log bins were merged to avoid zero-mode bins against the discrete mode grid). Absolute normalization is arbitrary; only convergence *ratios* below are meaningful, which is why that's fine.

## Convergence table (p=25 primary data vector)

| N | diag drift vs N=ALL (median \|ratio-1\|) | mean \|off-diag corr\| | off-diag corr - ALL | Hartlap h(N,25) | sigma_A (GLS, Hartlap-corr.) | sigma_A drift vs ALL | bootstrap diag scatter (N=50,ALL only) |
|---|---|---|---|---|---|---|---|
| 25 | 18.0% | 0.433 | +0.019 | -0.083 (invalid, N<=p+2) | singular | -- | -- |
| 50 | 19.7% | 0.419 | +0.005 | 0.469 | 0.02118 | -14.1% | 19.4% |
| 100 | 10.8% | 0.403 | -0.011 | 0.737 | 0.02258 | -8.4% | -- |
| 200 | 5.9% | 0.422 | +0.008 | 0.869 | 0.02503 | +1.5% | -- |
| 300 | 6.7% | 0.427 | +0.013 | 0.913 | 0.02537 | +2.9% | -- |
| 400 | 3.1% | 0.423 | +0.009 | 0.935 | 0.02469 | +0.1% | -- |
| 477 | 0.0% | 0.414 | +0.000 | 0.945 | 0.02466 | +0.0% | 7.0% |

*Template for the GLS amplitude fit is the mean spectrum over the full N=477 sample (fixed across rows, so the table isolates how the **covariance estimate** alone changes with N, not template noise); precision matrix is Hartlap-debiased, $\hat C^{-1} \to h(N,p)\,\hat C^{-1}$. N=25 is flagged singular because a 25-realization sample covariance of a 25-bin vector has rank <=24 (ddof=1) -- it is exactly the boundary case where GLS is not yet defined; this is a real illustration of why N must exceed p by a healthy margin, not a bug.*

## Secondary check: p=45 data vector

| N | diag drift vs N=ALL | mean \|off-diag corr\| | Hartlap h(N,45) | sigma_A |
|---|---|---|---|---|
| 25 | 18.4% | 0.397 | -0.792 (invalid, N<=p+2) | singular |
| 50 | 19.0% | 0.384 | 0.122 | 0.01947 |
| 100 | 10.0% | 0.378 | 0.566 | 0.02188 |
| 200 | 5.2% | 0.399 | 0.784 | 0.02494 |
| 300 | 5.7% | 0.402 | 0.856 | 0.02531 |
| 400 | 2.4% | 0.398 | 0.892 | 0.02453 |
| 477 | 0.0% | 0.390 | 0.910 | 0.02453 |

## Context: N=50 (paper) -> N=477 (now) -> N=550 (final)

- Hartlap at N=50, p=25 (as previously used): **h=0.469**. This matches what's already recorded elsewhere in this pipeline (`FIGURE_NUMBERS.md` fig10: "Hartlap factor 0.47") -- the author's recollection of "~0.39" is a bit low; the formula $(N-p-2)/(N-1)$ at N=50,p=25 gives 0.469, i.e. 0.47.

- Hartlap at N=477 (current), p=25: **h=0.945** (p=45: 0.910).

- Hartlap at N=550 (final), p=25: **h=0.953** (p=45: 0.922) -- formula-only projection, no new data needed since Hartlap depends only on (N,p).

## Verdict

At N=477 the covariance is already close to its converged shape for both compressions: the diagonal at N=400 differs from the current N=477 diagonal by only 3.1% (p=25, median over bins) and 2.4% (p=45) -- i.e. the extra bins do not obviously cost convergence speed in the raw diagonal, since after merging away the empty low-ell bins both vectors sit on Fourier modes that are already well sampled by N~400. What *does* cost more for the finer vector is degrees of freedom: the Hartlap factor has climbed from h=0.47 at the paper's N=50 (p=25) to h=0.945 now and will reach h=0.953 at the final N=550 -- most of that gain is already banked, and the remaining 73 realizations buy a comparatively small further step (h=0.945->0.953). The p=45 vector trails behind at every N (h=0.910 now, h=0.922 at N=550) simply because h=(N-p-2)/(N-1) pays a bigger fixed cost per extra bin: at N=50 a naive (non-Hartlap) inverse covariance overstates the precision by 1/h~2.1x (i.e. understates sigma_A by ~31%, overconfident), and that overconfidence is essentially gone by N=477 (1/h~1.06x for p=25, 1/h~1.10x for p=45). Bootstrap resampling shows *how well we know the covariance itself* has tightened a lot less dramatically in relative terms than the raw N would suggest: the diagonal's own realization-to-realization scatter is 19% at N=50 vs 7% at N=477 (roughly the expected $1/\sqrt{N}$ scaling, so a real, non-zero floor, not noise). The practical number -- $\sigma_A$ for a fiducial-amplitude fit -- has settled to within 0.1% (p=25) and 0.0% (p=45) of its final value already by N=400, so the remaining ~73 realizations of the retrace will sharpen both covariances further but are well past the steep part of the diminishing-returns curve for either compression; the main thing 550 (over 477) still buys is a slightly less Hartlap-penalized precision matrix for whichever data vector the final analysis compresses to, not a materially different covariance shape or amplitude error.
