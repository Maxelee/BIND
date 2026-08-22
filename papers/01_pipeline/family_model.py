"""family_model.py -- the shipped family-basis model, loader + predictor.

Loads figs_preview/family_model_bundle.npz (built by family_basis_all.py)
and predicts any of the 7 WL field statistics of the BIND lightcone suite as

    stat(x; lambda) = mean(x) + sum_k a_k(lambda) B_k(x)
    a_k(lambda)     = M_k . (lambda - lambda_ref) + a_ref_k

where the B_k are the twobound shape-family basis vectors and the amplitudes
come from the LAMBDA ROUTE: a linear map from MEASURED halo numbers, fit on
the 256-run Sobol design (the GP theta-route was dropped -- it is less
accurate for every statistic).  numpy only, no sklearn/pickle.

lambda is self-describing in the bundle -- ``fm.lat_names`` gives the
ordered latent names and ``fm.lat_convention`` the exact definitions (no
hardcoded dimensionality here; ``lam`` must simply match ``len(fm.lat_names)``
entries in that order). As of the 2026-08-12 OBSERVABLE AGNOSTIC-LAMBDA
promotion (chosen by a fully agnostic, observables-only SFFS search over
97 candidate halo summaries -- agnostic_lambda_search.py, OBS_ONLY=1;
see FAMILY_BASIS_METHODS.md) that is 8 latents, every one X-ray/SZ/
optical accessible, ordered by selection (each a median over halos in
the quoted log10 M_500c,bg bin):
    lambda[0] = f_star[13.2-13.4]   = median(M_star,500c/M_tot,500c,bg) / (Ob/Om)
    lambda[1] = logT[13.0-13.2]     = log10 median T_mw,500c [K] (T>0)
    lambda[2] = logY[13.0-13.2]     = log10 median Y_500c (Y>0)
    lambda[3] = logPe[14.0-14.3]    = log10 median Pe_mw,500c (Pe>0)
    lambda[4] = c_gas[14.0-14.3]    = median(M_gas,500c,bg/M_gas,200c,bg)
    lambda[5] = logY_ss[13.4-13.6]  = log10 median Y_500c/M_tot,500c,bg^(5/3) (>0)
    lambda[6] = c_gas[13.2-13.4]    = median(M_gas,500c,bg/M_gas,200c,bg)
    lambda[7] = logPe[13.4-13.6]    = log10 median Pe_mw,500c (Pe>0)
with Ob/Om = 0.0486/0.3089 (>=5 halos required per bin, else NaN). The
baryon fraction is deliberately NOT a latent: the search picked it first
and then evicted it as redundant with the same bin's Y + T. Measure
these in YOUR simulation (TNG, SIMBA, ...) or constrain them from
observations -- no CAMELS parameters are involved. NB the map is linear:
latents far outside the Sobol design range extrapolate unphysically.
Prior bundles are preserved at figs_preview/family_model_bundle_2bin8.npz
(two-bin 8) and figs_preview/family_model_bundle_4lat.npz (original 4).

Usage:
    from family_model import FamilyModel
    fm = FamilyModel()
    print(fm.lat_names)                   # ordered latent names
    lam = fm._z["clk__lat_ref"]           # e.g. the design-mean latents
    S   = fm.predict(lam, "clk")          # S(ell) on fm.grid("clk")
    pdf = fm.predict(lam, "pdf")          # kappa-PDF(nu)
    a   = fm.amplitudes(lam, "clk")       # the family amplitudes

Statistics: clk (S(ell)), pdf (kappa-PDF), pk / mn (peak / minima counts),
v0 / v1 / v2 (Minkowski functionals, nu <= 3.75).
"""
from pathlib import Path

import numpy as np

BUNDLE_PATH = Path(__file__).parent / "figs_preview/family_model_bundle.npz"
STATS = ("clk", "pdf", "pk", "mn", "v0", "v1", "v2")


class FamilyModel:
    def __init__(self, bundle_path=BUNDLE_PATH):
        z = np.load(bundle_path, allow_pickle=True)
        self._z = {k: z[k] for k in z.files}
        self.lat_names = [str(s) for s in self._z["lat_names"]]
        self.lat_convention = str(self._z["lat_convention"])
        self.stats = [s for s in STATS if f"{s}__basis" in self._z]

    def grid(self, stat):
        return self._z[f"{stat}__x"]

    def basis(self, stat):
        return self._z[f"{stat}__basis"]

    def family_names(self, stat):
        return [str(s) for s in self._z[f"{stat}__fam_names"]]

    def metrics(self, stat):
        r2 = self._z[f"{stat}__r2"]
        return dict(r2_full=float(r2[0]), r2_span=float(r2[1]),
                    r2_model=float(r2[2]), r2_lambda_e2e_cv=float(r2[3]),
                    r2_amp_cv=self._z[f"{stat}__r2_amp_cv"])

    def _check(self, lam, stat):
        if stat not in self.stats:
            raise ValueError(f"unknown statistic {stat!r}; "
                             f"available: {self.stats}")
        lam = np.atleast_2d(np.asarray(lam, float))
        n_lat = len(self.lat_names)
        if lam.shape[1] != n_lat:
            raise ValueError(f"lambda must have {n_lat} entries "
                             f"({', '.join(self.lat_names)}); see "
                             "self.lat_convention")
        return lam

    def amplitudes(self, lam, stat):
        """(n, n_lat) measured halo latents -> (n, K) family amplitudes."""
        lam = self._check(lam, stat)
        Lc = lam - self._z[f"{stat}__lat_ref"]
        return np.c_[Lc, np.ones(len(Lc))] @ self._z[f"{stat}__lat_M"]

    def predict(self, lam, stat):
        """(n, n_lat) measured halo latents -> (n, n_bins) statistic curves."""
        a = self.amplitudes(lam, stat)
        return self._z[f"{stat}__mean"] + a @ self._z[f"{stat}__basis"]

    def predictive_sigma(self, stat):
        """per-bin 1-sigma predictive uncertainty of the lambda route (the
        CV residual scatter across the design).  Scale-dependent: for
        S(ell) it grows by ~10x from ell~10^3 to the ell>10^4 trough --
        four group-bin numbers do not pin the small-scale response.  Always
        quote predictions with this band."""
        return self._z[f"{stat}__sig_pred"]


if __name__ == "__main__":
    fm = FamilyModel()
    lam0 = fm._z["clk__lat_ref"]
    print(f"family model: {len(fm.stats)} statistics; lambda = "
          f"{fm.lat_names}, design ref {np.round(lam0, 3)}")
    for st in fm.stats:
        m = fm.metrics(st)
        pred = fm.predict(lam0, st)[0]
        print(f"  {st:>4s}: K={len(fm.family_names(st))} "
              f"{fm.family_names(st)}, span R^2 {m['r2_span']:.4f}, "
              f"lambda e2e CV {m['r2_lambda_e2e_cv']:.4f}; pred@ref "
              f"range [{pred.min():.3g}, {pred.max():.3g}]")
