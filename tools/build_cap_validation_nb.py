"""Build profile_fb_cap_validation.ipynb -- re-run the real-data (CAMELS-truth)
validation of the Observable -> f_b map using **CAP-filtered observables** (the
background-nulled estimator a real tSZ/X-ray stack actually uses), and compare to the
raw-profile result.

Execute:
  jupyter nbconvert --execute --inplace \
    --ExecutePreprocessor.kernel_name=torch3 --ExecutePreprocessor.timeout=1800 \
    profile_fb_cap_validation.ipynb
"""
import nbformat as nbf
nb=nbf.v4.new_notebook(); cells=[]
md=lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co=lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Real-data validation with **CAP-filtered observables**

**⚠️ Provisional.** Follow-up to `profile_fb_realdata.ipynb` + `projection_fb_check.ipynb`.

The earlier validation fed the map the **raw** stacked Y(r), SX(r) profiles. But a real
tSZ/X-ray stack cannot measure the raw profile — the line-of-sight background is unknown.
The standard estimator is **Compensated Aperture Photometry (CAP)**: at aperture θ,
`CAP(θ) = mean(disk r<θ) − mean(annulus θ<r<√2θ)`, which **cancels any uniform
background** (and large-scale modes) exactly. tSZ Y is the LOS-contaminated observable
(§3 of the projection notebook); CAP is how that contamination is removed.

Here we **re-run the BIND→CAMELS-truth validation using CAP-filtered Y and SX** as the
features (target f_b(r) unchanged — it is the LOS-robust mass-based profile), and ask:
1. does the map still recover truth f_b(r) from the *realistic* (background-nulled)
   observables? 2. is it **more robust to observational noise** than the raw profile?

**Key fact:** CAP is a linear, radius-only functional, so we compute it exactly from the
stacked azimuthal profiles + the per-bin pixel counts — no re-reduction.""")

co(r"""import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, importlib.util
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
spec=importlib.util.spec_from_file_location('spr','tools/stack_profiles_reduce.py')
spr=importlib.util.module_from_spec(spec); spec.loader.exec_module(spr)
r=spr.R_CEN; NR=spr.NR; Nr=np.array([np.sum(spr._RBIN==b) for b in range(NR)],float)
SOBOL=Path('/mnt/home/mlee1/ceph/sobol_ss_cv'); FIG=Path('figures/observable_fb'); RNG=np.random.default_rng(2)
B=np.load(SOBOL/'stacked_profiles.npz',allow_pickle=True); MLBL=list(B['mass_lbl'])
Bp={k:B[k] for k in ['Y','SX','fb']}
T=np.load(SOBOL/'truth_stacked_profiles.npz',allow_pickle=True)

def cap_one(O):
    '''CAP(theta) from a radial profile O(r); NaN where the sqrt2 annulus does not fit.'''
    out=np.full(NR,np.nan)
    for j in range(NR):
        din=r<=r[j]; ann=(r>r[j])&(r<=np.sqrt(2)*r[j])
        if din.sum() and ann.sum():
            out[j]=np.sum(O[din]*Nr[din])/Nr[din].sum()-np.sum(O[ann]*Nr[ann])/Nr[ann].sum()
    return out
VALID=np.isfinite(cap_one(Bp['Y'][0,1]))     # which apertures are usable (first 11)
def cap(O): return cap_one(O)[VALID]
print('CAP valid apertures:',VALID.sum(),'of',NR)""")

md(r"""## 1. Aggregate the CAMELS truth (all-CV + cosmic-variance bootstrap)""")

co(r"""KEYS=['y_sum','SX_sum','gas_sum','baryon_sum','tot_sum']
S={k:T[k] for k in KEYS}; cnt_sb=T['counts']; nsim=cnt_sb.shape[0]
def aggregate(sel):
    acc={k:np.zeros((3,NR)) for k in KEYS}; w=np.zeros(3)
    for s in sel:
        for k in KEYS: acc[k]+=np.nan_to_num(S[k][s])
        w+=cnt_sb[s]
    return dict(Y=acc['y_sum']/w[:,None], SX=acc['SX_sum']/w[:,None],
                fb=acc['baryon_sum']/acc['tot_sum'])
truth=aggregate(np.arange(nsim))
boot=[aggregate(RNG.integers(0,nsim,nsim)) for _ in range(300)]
print('truth aggregated; f_b mid bin:',truth['fb'][1].round(3))""")

md(r"""## 2. CAP profiles of the truth observables (what a survey would measure)""")

co(r"""fig,ax=plt.subplots(1,2,figsize=(11,4))
rv=r[VALID]
for j,(nm,scale) in enumerate([('Y',1),('SX',1)]):
    cv=cap(truth[nm][1])
    bb=np.array([cap(b[nm][1]) for b in boot]); lo,hi=np.percentile(bb,[16,84],0)
    ax[j].plot(rv,cv,'o-',c='crimson'); ax[j].fill_between(rv,lo,hi,color='crimson',alpha=0.2)
    ax[j].plot(r,truth[nm][1],'s--',c='0.6',label='raw profile')
    ax[j].set_xscale('log'); ax[j].set_yscale('log'); ax[j].set_xlabel('aperture θ / r [kpc/h]')
    ax[j].set_title(f'{nm}: CAP(θ) (red) vs raw (grey), mid bin'); ax[j].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG/'cap_truth_profiles.png',dpi=130); print('saved cap_truth_profiles.png'); plt.close(fig)""")

md(r"""## 3. Predict truth f_b(r): raw observables vs CAP-filtered observables
Both train the Ridge map on the 256 BIND designs and predict the CAMELS-truth f_b(r).
Raw features = log10[Y(r),SX(r)]; CAP features = CAP_Y(θ),CAP_SX(θ). Target identical.""")

co(r"""def feat_raw(obs,b): return np.hstack([np.log10(np.clip(obs['Y'][b],1e-30,None)),
                                        np.log10(np.clip(obs['SX'][b],1e-30,None))])
def feat_cap(obs,b): return np.hstack([cap(obs['Y'][b]),cap(obs['SX'][b])])

def train_predict(featfn,b,obs_test,alpha=10.0):
    Xtr=np.array([featfn({'Y':Bp['Y'][d],'SX':Bp['SX'][d]},b) for d in range(Bp['Y'].shape[0])])
    ytr=Bp['fb'][:,b,:]
    xs=StandardScaler().fit(Xtr); ys=StandardScaler().fit(ytr)
    reg=Ridge(alpha=alpha).fit(xs.transform(Xtr),ys.transform(ytr))
    return ys.inverse_transform(reg.predict(xs.transform(featfn(obs_test,b)[None])))[0]

print('NOISE-FREE RMS|pred-truth| in f_b (x1e3): RAW vs CAP observables')
print(f'{"bin":14s} {"RAW":>8s} {"CAP":>8s}')
pred={}
for b,l in enumerate(MLBL):
    pr=train_predict(feat_raw,b,truth); pc=train_predict(feat_cap,b,truth)
    pred[('raw',l)]=pr; pred[('cap',l)]=pc
    rr=np.sqrt(np.nanmean((pr-truth['fb'][b])**2)); rc=np.sqrt(np.nanmean((pc-truth['fb'][b])**2))
    print(f'{l:14s} {rr*1e3:8.2f} {rc*1e3:8.2f}')""")

co(r"""# predicted vs truth, CAP, with truth cosmic-variance band
fbb=np.array([b['fb'] for b in boot]); flo,fhi=np.percentile(fbb,[16,84],0)
fig,axes=plt.subplots(1,3,figsize=(14,4.2))
for b,(ax,l) in enumerate(zip(axes,MLBL)):
    ax.fill_between(r,flo[b],fhi[b],color='crimson',alpha=0.2,label='truth ±cosmic var')
    ax.plot(r,truth['fb'][b],'-',c='crimson',lw=2,label='CAMELS truth')
    ax.plot(r,pred[('raw',l)],'--',c='k',lw=1.3,label='pred (raw obs)')
    ax.plot(r,pred[('cap',l)],'-',c='C0',lw=1.6,label='pred (CAP obs)')
    ax.set_xscale('log'); ax.set_xlabel('r [kpc/h]'); ax.set_title(l); ax.set_ylim(0.06,0.20)
axes[0].set_ylabel('$f_b(r)$'); axes[0].legend(fontsize=7.5)
fig.suptitle('Predict CAMELS-truth $f_b(r)$: raw vs CAP-filtered observables')
fig.tight_layout(); fig.savefig(FIG/'cap_pred_vs_truth.png',dpi=130); print('saved cap_pred_vs_truth.png'); plt.close(fig)""")

md(r"""## 4. Noise robustness (honest per-measurement metric)
Add multiplicative log-normal noise per radial bin to the truth Y, SX **before**
feature extraction, and report the **median over 300 draws of the per-measurement
RMS|pred−truth|** — i.e. the typical error of a *single* noisy measurement. (We
deliberately avoid RMS-of-the-mean-prediction, which understates error at high noise
because the mean regresses toward the training mean.) The Ridge map is fit **once** per
(bin, method); only the noisy test profile changes.""")

co(r"""import json
def fit_model(featfn,b):
    Xtr=np.array([featfn({'Y':Bp['Y'][d],'SX':Bp['SX'][d]},b) for d in range(Bp['Y'].shape[0])])
    ytr=Bp['fb'][:,b,:]; xs=StandardScaler().fit(Xtr); ys=StandardScaler().fit(ytr)
    reg=Ridge(alpha=10.0).fit(xs.transform(Xtr),ys.transform(ytr))
    return lambda obs,bb: ys.inverse_transform(reg.predict(xs.transform(featfn(obs,bb)[None])))[0]
def noisy(obs,f):
    o={k:obs[k].copy() for k in obs}
    for k in ['Y','SX']: o[k]=obs[k]*np.exp(RNG.normal(0,f,obs[k].shape))
    return o
NOISE=[0.0,0.05,0.10,0.20,0.40]; nmc=300
res={'noise':NOISE,'raw':{},'cap':{}}
for b,l in enumerate(MLBL):
    mraw=fit_model(feat_raw,b); mcap=fit_model(feat_cap,b)
    for nm,mdl,ff in [('raw',mraw,feat_raw),('cap',mcap,feat_cap)]:
        med=[]
        for f in NOISE:
            e=[np.sqrt(np.nanmean((mdl(noisy(truth,f),b)-truth['fb'][b])**2)) for _ in range(nmc if f>0 else 1)]
            med.append(float(np.median(e)))
        res[nm][l]=med
print('median per-measurement RMS|pred-truth| (x1e3) vs noise  [RAW / CAP]')
print(f'{"bin":12s}'+''.join(f'{int(f*100):>9d}%' for f in NOISE))
for b,l in enumerate(MLBL):
    print(f'{l:12s} raw '+''.join(f'{x*1e3:9.1f}' for x in res['raw'][l]))
    print(f'{"":12s} cap '+''.join(f'{x*1e3:9.1f}' for x in res['cap'][l]))

fig,axes=plt.subplots(1,3,figsize=(14,4))
for b,(ax,l) in enumerate(zip(axes,MLBL)):
    ax.plot(np.array(NOISE)*100,np.array(res['raw'][l])*1e3,'o--k',label='raw obs')
    ax.plot(np.array(NOISE)*100,np.array(res['cap'][l])*1e3,'o-C0',label='CAP obs')
    ax.set_xlabel('obs noise [%]'); ax.set_title(l)
axes[0].set_ylabel('median per-meas. RMS $f_b$ (x1e3)'); axes[0].legend(fontsize=8)
fig.suptitle('Noise robustness (per-measurement): CAP vs raw observables')
fig.tight_layout(); fig.savefig(FIG/'cap_noise_robustness.png',dpi=130); print('saved cap_noise_robustness.png'); plt.close(fig)
json.dump(res,open(FIG/'cap_results.json','w'),indent=2); print('wrote cap_results.json')""")

md(r"""## 5. Reading the result
- **§3:** if CAP RMS ≈ raw RMS, the map still recovers truth f_b(r) from the realistic
  background-nulled observable — i.e. the result survives the estimator a real survey
  must use (CAP discards the unmeasurable raw zero-point, so being ~as good is the win).
- **§4:** per-measurement error grows ~linearly with noise for **both** raw and CAP —
  CAP is **not** more noise-robust here (slightly worse throughout). The mass-f_b target
  is already LOS-robust and the whole-profile fit averages noise, so CAP's
  background-nulling buys robustness against the *background*, not against per-bin noise.
  At ≳20% per-bin noise the error approaches σ_marg (the prediction loses most skill).
- **Caveats unchanged:** single feedback point (interpolation), shared DMO halos,
  projected/mass-based f_b target, uncalibrated SX, uniform toy noise. And CAP here is
  computed from the 50 Mpc/h projection's profile — a fully realistic test still wants
  6.25 Mpc/h thermo cubes to set the true Y LOS background that CAP removes.""")

nb['cells']=cells
nb.metadata['kernelspec']={'name':'python3','display_name':'torch3','language':'python'}
nbf.write(nb,'profile_fb_cap_validation.ipynb')
print('wrote profile_fb_cap_validation.ipynb with',len(cells),'cells')
