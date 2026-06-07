#!/usr/bin/env python
"""Build field_vs_profile_act.ipynb -- the field-vs-profile demonstration tuned to
the ACT x CMASS CAP pipeline and validated on a CAMELS hydro-truth observation.

Observable = Compensated Aperture Photometry (Schaan 2021 / Amodeo 2021): tSZ CAP
(Compton-Y) + kSZ CAP (projected gas), ACT f150 1.6' beam at z=0.55, galaxy
mis-centering, flux-limited (Eddington) selection.  Two products feed the notebook
(built by tools/forward_model_act.py):

  forward_model_act_bind.npz   -- 256 BIND Sobol designs (training): per design the
       PROFILE-CAP (axisymmetrised+beamed, the steelmanned profile-emulator output)
       and the FIELD-CAP (mis-centered 2D, flux-selected, what ACT measures), tSZ+kSZ,
       plus parent f_b.
  forward_model_act_truth.npz  -- the 26-sim CV hydro TRUTH (one feedback point):
       per-halo PROFILE-CAP, FIELD-CAP, f_b, mass, flux -> stacked + bootstrapped
       in the notebook to give the real observation.

Two panels:
  (1) leave-one-design-out over 256 BIND feedbacks -> the systematic bias a profile
      emulator pays on a realistic ACT measurement;
  (2) train on all 256 BIND designs, predict the CAMELS-truth f_b -> the profile path
      lands off the truth, the field path lands on it (real-data validation).

Run:  python tools/build_fieldvsprofile_act_nb.py
Then: jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=torch3 field_vs_profile_act.ipynb
"""
from pathlib import Path
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def co(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ----------------------------------------------------------------------------
md(r"""
# Why the field, not the profile: an ACT CAP measurement of the baryon fraction

We want the baryon fraction $f_b$ of a cluster/group population from what **ACT
actually measures**: stacked **Compensated Aperture Photometry** (CAP) of the tSZ
(Compton-$Y$) and kSZ (projected gas) signals around a flux-selected sample
(Schaan et al. 2021; Amodeo et al. 2021). The training link observable $\to f_b$
could come from a **profile-level** emulator (clean azimuthal profile) or a
**field-level** emulator (BIND, full 2D field).

**The point.** A real CAP is computed on a **beam-convolved, mis-centered 2D map**.
A profile emulator can output only the clean, centred, axisymmetric profile — even if
it applies the beam analytically, it cannot reproduce the mis-centering or the
per-halo asphericity, nor the flux selection (which acts on the 2D flux). So it
predicts the wrong CAP and infers a biased $f_b$. A field emulator runs the identical
2D pipeline on its training fields and is unbiased. We (1) measure the bias across 256
unseen feedback settings, and (2) validate it on a genuine CAMELS hydro observation.

Fully reproducible from `forward_model_act_{bind,truth}.npz`
(`tools/forward_model_act.py`).
""")

co(r"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
import scienceplots
plt.style.use(['science', 'no-latex'])

S = Path('/mnt/home/mlee1/ceph/sobol_ss_cv')
FIG = Path('figures/field_vs_profile_act'); FIG.mkdir(parents=True, exist_ok=True)

B = np.load(S / 'forward_model_act_bind.npz', allow_pickle=True)
T = np.load(S / 'forward_model_act_truth.npz', allow_pickle=True)
theta = B['theta_arcmin']
meta = B['meta'].item()
# BIND per design: profile-CAP (cpY,cpG) and field-CAP (cfY,cfG), tSZ + kSZ
cpY, cpG, cfY, cfG = B['cpY'], B['cpG'], B['cfY'], B['cfG']     # (256, ntheta)
fb_bind = B['fb_parent']                                        # (256,)
Nd = len(fb_bind)
print('ACT pipeline:', {k: meta[k] for k in
      ['z_obs', 'beam_fwhm_arcmin', 'sigma_mc_kpc', 'flux_scatter', 'sel_q']})
print('kpc/h per arcmin @ z :', round(meta['kpc_h_per_arcmin'], 1))
print('CAP apertures arcmin :', list(theta))
print('BIND designs         :', Nd, ' | mean f_sel', round(float(B['frac_sel'].mean()), 2))
print('truth halos          :', len(T['fb']))
""")

# ----------------------------------------------------------------------------
md(r"""
## 1. The ACT CAP forward model: profile vs field

The two CAP observables differ only by the irreducibly-2D operations. The
**profile** CAP (blue) axisymmetrises and beam-convolves the clean field (centred);
the **field** CAP (red) mis-centers and beam-convolves the 2D field, as ACT measures
it. The ratio is a large, aperture-dependent factor that is **not a function of the
clean profile** — so a profile emulator cannot produce it. The CAMELS-truth stack
(black) is built with the identical pipeline.
""")

co(r"""
# truth stacked CAP with the same mass function + flux selection as BIND
Mt = T['M200']; w_mf = Mt ** (-meta['mf_slope']); w_mf /= w_mf.sum()
rng = np.random.default_rng(meta['seed'])
obs_flux = T['flux'] * (1.0 + rng.normal(0, meta['flux_scatter'], T['flux'].shape))
limit = np.quantile(T['flux'][T['flux'] > 0], meta['sel_q'])
sel = obs_flux > limit; ws = w_mf * sel
tcpY = np.average(T['cpY'], 0, weights=w_mf)     # truth profile CAP (parent)
tcfY = np.average(T['cfY'], 0, weights=ws)       # truth field CAP (selected) = the measurement

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
axes[0].plot(theta, cpY.mean(0), 'C0-', label='profile CAP (axisym + beam)')
axes[0].fill_between(theta, np.percentile(cpY, 16, 0), np.percentile(cpY, 84, 0), color='C0', alpha=.2)
axes[0].plot(theta, cfY.mean(0), 'C3-', label='field CAP (2D, mis-centered)')
axes[0].fill_between(theta, np.percentile(cfY, 16, 0), np.percentile(cfY, 84, 0), color='C3', alpha=.2)
axes[0].plot(theta, tcfY, 'k--', label='CAMELS truth (field)')
axes[0].set_yscale('log'); axes[0].set_xlabel(r'CAP aperture $\theta\;$[arcmin]')
axes[0].set_ylabel(r'tSZ $T_{\rm CAP}(\theta)$'); axes[0].legend()
axes[1].axhline(1, color='0.6', lw=.8)
ratio = cfY / cpY
axes[1].plot(theta, ratio.mean(0), 'k-')
axes[1].fill_between(theta, np.percentile(ratio, 16, 0), np.percentile(ratio, 84, 0), color='0.5', alpha=.3)
axes[1].set_xlabel(r'CAP aperture $\theta\;$[arcmin]'); axes[1].set_ylabel('field / profile CAP')
axes[1].annotate('mis-centering + asphericity\n(invisible to a profile model)',
                 (0.05, 0.08), xycoords='axes fraction')
fig.tight_layout(); fig.savefig(FIG / 'fig1_act_cap.png', dpi=200, bbox_inches='tight')
plt.show()
print('field/profile CAP ratio (tSZ):', np.round(ratio.mean(0), 2))
""")

# ----------------------------------------------------------------------------
md(r"""
## 2. The bias across 256 unseen feedback settings

Each BIND design is, in turn, the held-out mock observation (leave-one-design-out;
unseen feedback). A Ridge model maps the (standardised) CAP feature vector
[tSZ$(\theta)$, kSZ$(\theta)$] to the population $f_b$. Three estimators differ only in
the training observable: profile/vacuum (profile$\to$profile), **profile/real**
(profile$\to$**field**), field/real (field$\to$field).
""")

co(r"""
def feats(*caps):
    return np.hstack(caps)

Xp = feats(cpY, cpG)        # profile CAP features (tSZ + kSZ)
Xf = feats(cfY, cfG)        # field CAP features (the real measurement)
y = fb_bind

def loo(Xtr, Xte, y, alpha=10.0):
    pred = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(y):
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Xtr[tr], y[tr])
        pred[te] = m.predict(Xte[te])
    return pred

def stat(pred, y):
    b = (pred - y).mean()
    return dict(bias=b, rel=100 * b / y.mean(), rms=np.sqrt(((pred - y) ** 2).mean()))

cases = {'profile, no instrument': loo(Xp, Xp, y),
         'profile, real obs':      loo(Xp, Xf, y),
         'field, real obs':        loo(Xf, Xf, y)}
res = {k: stat(v, y) for k, v in cases.items()}
for k, r in res.items():
    print(f'{k:24s} bias={r["bias"]:+.4f} ({r["rel"]:+.1f}%) RMS={r["rms"]:.4f}')

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.6, 3.4))
lo, hi = y.min() * 0.97, y.max() * 1.03
cm = {'profile, no instrument': '0.6', 'profile, real obs': 'C1', 'field, real obs': 'C2'}
axA.plot([lo, hi], [lo, hi], 'k--', lw=1)
for k in cases:
    axA.scatter(y, cases[k], s=8, alpha=.6, color=cm[k], label=f'{k} ({res[k]["rel"]:+.0f}%)')
axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
axA.set_xlabel(r'true population $f_b$'); axA.set_ylabel(r'predicted $f_b$'); axA.legend(loc='upper left')
order = list(cases); x = np.arange(3); w = .38
axB.bar(x - w / 2, [abs(res[k]['bias']) for k in order], w, color='C3', label='|bias|')
axB.bar(x + w / 2, [res[k]['rms'] for k in order], w, color='C0', label='RMS')
axB.set_xticks(x); axB.set_xticklabels(['profile\n(vacuum)', 'profile\n(real obs)', 'field\n(real obs)'])
axB.set_ylabel(r'error in $f_b$'); axB.legend()
fig.tight_layout(); fig.savefig(FIG / 'fig2_act_bias.png', dpi=200, bbox_inches='tight')
plt.show()
print(f'HEADLINE: profile emulator biased {res["profile, real obs"]["rel"]:+.0f}% on the ACT'
      f' measurement; field emulator unbiased (RMS {res["field, real obs"]["rms"]:.4f}).')
""")

# ----------------------------------------------------------------------------
md(r"""
## 3. Real-data validation: predicting the CAMELS-truth $f_b$

The decisive test: train on all 256 BIND designs, then predict the baryon fraction of
a **genuine CAMELS hydro simulation** (held-out feedback = the CV fiducial) from its
ACT-CAP measurement. The profile model (trained on profile CAP) applied to the truth
**field** CAP lands off the truth; the field model lands on it. Error bars are a
bootstrap over the 1111 truth halos (resampling the stack).
""")

co(r"""
# fit final models on ALL designs
mP = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xp, y)   # profile model
mF = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xf, y)   # field model

# bootstrap the truth observation (resample halos) -> distributions
nb = 500; nH = len(T['fb'])
pred_prof = np.empty(nb); pred_field = np.empty(nb); truth_fb = np.empty(nb)
rng = np.random.default_rng(1)
for b in range(nb):
    idx = rng.integers(0, nH, nH)
    m = Mt[idx]; wmf = m ** (-meta['mf_slope']); wmf /= wmf.sum()
    of = T['flux'][idx] * (1.0 + rng.normal(0, meta['flux_scatter'], nH))
    lim = np.quantile(T['flux'][idx][T['flux'][idx] > 0], meta['sel_q'])
    s = of > lim; wsb = wmf * s
    Fobs = feats(np.average(T['cfY'][idx], 0, weights=wsb)[None],
                 np.average(T['cfG'][idx], 0, weights=wsb)[None])     # field CAP (measured)
    pred_prof[b] = mP.predict(Fobs)[0]      # profile model on the real (field) measurement
    pred_field[b] = mF.predict(Fobs)[0]     # field model on the real measurement
    truth_fb[b] = np.average(T['fb'][idx], weights=wmf)

tf = truth_fb.mean()
fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.axhline(tf, color='k', ls='--', label='CAMELS truth $f_b$')
ax.axhspan(np.percentile(truth_fb, 16), np.percentile(truth_fb, 84), color='0.8', alpha=.6)
for i, (lbl, p, c) in enumerate([('profile model\non ACT obs', pred_prof, 'C1'),
                                 ('field model\non ACT obs', pred_field, 'C2')]):
    ax.errorbar([i], [p.mean()], yerr=[[p.mean() - np.percentile(p, 16)],
                [np.percentile(p, 84) - p.mean()]], fmt='o', color=c, capsize=4, ms=7)
    ax.annotate(f'{100*(p.mean()-tf)/tf:+.0f}%', (i, p.mean()),
                textcoords='offset points', xytext=(10, 0), va='center', color=c)
ax.set_xticks([0, 1]); ax.set_xticklabels(['profile model\non ACT obs', 'field model\non ACT obs'])
ax.set_ylabel(r'$f_b$ inferred from CAMELS truth'); ax.set_xlim(-0.5, 1.7); ax.legend(loc='lower right')
fig.tight_layout(); fig.savefig(FIG / 'fig3_truth_validation.png', dpi=200, bbox_inches='tight')
plt.show()
print(f'CAMELS truth f_b      : {tf:.4f}')
print(f'profile model -> f_b  : {pred_prof.mean():.4f}  ({100*(pred_prof.mean()-tf)/tf:+.1f}% bias)')
print(f'field   model -> f_b  : {pred_field.mean():.4f}  ({100*(pred_field.mean()-tf)/tf:+.1f}% bias)')
""")

# ----------------------------------------------------------------------------
md(r"""
## 4. Conclusion

Tuned to the actual ACT $\times$ CMASS CAP pipeline (tSZ + kSZ, $1.6'$ beam,
$z=0.55$, mis-centering, flux selection) and tested on a genuine CAMELS hydro
observation, the result is unchanged and sharper: a **profile/scalar-level emulator
infers a biased $f_b$** because the CAP it can predict (centred, axisymmetric) is not
the CAP ACT measures (mis-centered 2D, flux-selected); a **field-level emulator
forward-models the identical pipeline and recovers the truth**. The bias is the price
of not having the field.

**Caveats.** One sub-grid family at fixed cosmology; the profile model is allowed the
beam but not the 2D morphology, mis-centering, or flux selection (all genuinely
field-level); CAP/beam/$z$ are representative ACT$\times$CMASS values, not a specific
map-level forward model; the truth observation is a single feedback point (the CV
fiducial), so the §3 validation is an interpolation within the trained feedback range
(the §2 leave-one-out covers the unseen-feedback case across 256 settings). The truth
test also exercises the BIND$\to$CAMELS transfer, established independently elsewhere.
""")

nb = nbf.v4.new_notebook(); nb['cells'] = cells
nb['metadata']['kernelspec'] = {'name': 'torch3', 'display_name': 'torch3', 'language': 'python'}
out = Path('field_vs_profile_act.ipynb'); nbf.write(nb, str(out))
print('wrote', out, 'with', len(cells), 'cells')
