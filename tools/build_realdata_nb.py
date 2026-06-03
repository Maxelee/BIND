"""Build profile_fb_realdata.ipynb -- REAL-DATA validation of the Observable -> f_b
map: train the map on the BIND/Sobol cube, apply it to CAMELS *hydro-truth* stacked
profiles (with realistic observational noise), compare predicted vs truth f_b(r).

Execute with:
  jupyter nbconvert --execute --inplace \
    --ExecutePreprocessor.kernel_name=torch3 --ExecutePreprocessor.timeout=1800 \
    profile_fb_realdata.ipynb
"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Observable → f_b — **real-data validation** (CAMELS hydro truth)

**⚠️ Provisional.** This is the test that matters: does the observable→f_b map that
BIND learned (trained on emulator profiles across 256 feedback designs) recover the
**actual CAMELS hydro f_b(r)** when fed the **actual hydro-truth** Y(r), SX(r) — with
realistic observational noise? Up to now obs *and* f_b were both BIND outputs
(self-consistency). Here the test profiles and the target are 100% hydro truth.

### What exactly is being tested, and the honest caveats
- **Training set:** 256 Sobol feedback designs, BIND-painted stacked profiles
  (`stacked_profiles.npz`). The map = `f_b(r) ← (Y,SX,…)(r)`, multi-output Ridge.
- **Test set:** the **27-sim CV** set = CAMELS IllustrisTNG at **fiducial cosmology
  AND fiducial feedback**, 1111 halos. We stack the **hydro-truth** fields
  (`truth_maps`=[DM,Gas,Stars], `truth_thermo`=[y,T,S,P]) into truth Y(r), SX(r) and
  the truth f_b(r) (`tools/stack_profiles_truth.py`), built with the *identical*
  geometry/stacking as training.
- **Caveat 1 (the user's "sketchy"):** these 1111 halos are the *same* DMO halos
  BIND was trained/painted on, so the DMO structure is shared. But the **truth
  observables and truth f_b were never used to fit anything** — only BIND's own
  f_b was. A fully clean test would also hold out halo identity; at the *population
  stack* level this matters less, but we flag it.
- **Caveat 2:** CV is a **single feedback point** (the fiducial). So this validates
  the map as an **interpolation** to the centre of the design cloud (we verify the
  truth observables land inside the cloud), not its behaviour at the feedback
  extremes, and not off-grid (SIMBA/real data) — still out of scope.
- **Caveat 3:** truth f_b(r) is the same **projected** ratio as training; SX the same
  uncalibrated ρ²√T proxy.""")

co(r"""import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
SOBOL=Path('/mnt/home/mlee1/ceph/sobol_ss_cv'); FIG=Path('figures/observable_fb')
RNG=np.random.default_rng(1)
B=np.load(SOBOL/'stacked_profiles.npz',allow_pickle=True)          # BIND training
T=np.load(SOBOL/'truth_stacked_profiles.npz',allow_pickle=True)    # CAMELS truth
r=B['r_kpc']; MLBL=list(B['mass_lbl']); nr=len(r)
Bp={k:B[k] for k in ['Y','SX','T','S','P','fb']}                   # (256,3,nr)
print('BIND designs',Bp['fb'].shape[0],'| truth sims',len(T['sims']),
      '| halos/bin',T['counts'].sum(0))""")

md(r"""## 1. Aggregate the truth stack (all-CV) and its cosmic-variance band
The truth stores, per (sim, bin), azimuthal profiles of the **summed** maps, so the
all-CV stack is an exact sum over sims: `Y=Σy_sum/Σcount`, `f_b=Σbaryon/Σtot`, etc.
(`nansum` skips sims with no halos in a bin). The **cosmic-variance** uncertainty on
the truth stack comes from **bootstrapping the 26 sims** (the independent phase
realisations) — this is the error bar on "what the universe's stacked profile is".""")

co(r"""KEYS=['y_sum','SX_sum','gas_sum','baryon_sum','tot_sum','Tgas_sum','Sgas_sum','Pgas_sum']
S={k:T[k] for k in KEYS}; cnt_sb=T['counts']                       # (nsim,3,nr),(nsim,3)
nsim=cnt_sb.shape[0]

def aggregate(sim_sel):
    '''all-CV (or bootstrap-subset) truth profiles -> dict of (3,nr).'''
    w=np.zeros(3);
    acc={k:np.zeros((3,nr)) for k in KEYS}
    for s in sim_sel:
        for k in KEYS: acc[k]+=np.nan_to_num(S[k][s])
        w+=cnt_sb[s]
    Y=acc['y_sum']/w[:,None]; SX=acc['SX_sum']/w[:,None]
    Tp=acc['Tgas_sum']/acc['gas_sum']; Sp=acc['Sgas_sum']/acc['gas_sum']
    Pp=acc['Pgas_sum']/acc['gas_sum']; fb=acc['baryon_sum']/acc['tot_sum']
    return dict(Y=Y,SX=SX,T=Tp,S=Sp,P=Pp,fb=fb)

truth=aggregate(np.arange(nsim))
# cosmic-variance bootstrap over sims
boot=[aggregate(RNG.integers(0,nsim,nsim)) for _ in range(300)]
def band(name):
    A=np.array([b[name] for b in boot]); return np.percentile(A,[16,84],0)  # (2,3,nr)
print('truth f_b(r) all-CV by bin:')
for b,l in enumerate(MLBL): print(f'  {l}',truth['fb'][b].round(3))""")

md(r"""## 2. Does the truth land on the BIND-learned joint relation?
The decisive domain check. Overlay the **truth** (Y(r*), f_b(r*)) point on the BIND
joint p(f_b,Y) (the 256 designs) at the peak-information radius. If truth sits on the
BIND conditional mean E[f_b|Y], the learned relation transfers to hydro truth.""")

co(r"""from scipy.stats import spearmanr
b=1; lbl=MLBL[b]; ri=5            # ~191 kpc/h, near peak reduction
x=np.log10(Bp['Y'][:,b,ri]); y=Bp['fb'][:,b,ri]
fig,ax=plt.subplots(figsize=(6,5))
ax.scatter(x,y,s=14,c='0.6',label='BIND designs (256)')
qe=np.quantile(x,np.linspace(0,1,11)); xc=0.5*(qe[1:]+qe[:-1])
mu=[np.median(y[(x>=qe[k])&(x<=qe[k+1])]) for k in range(10)]
ax.plot(xc,mu,'k-',lw=2,label='E[$f_b$|Y] (BIND)')
# truth point + cosmic-variance bars
tx=np.log10(truth['Y'][b,ri]); ty=truth['fb'][b,ri]
yb=band('fb')[:,b,ri]; xb=np.log10(band('Y')[:,b,ri])
ax.errorbar([tx],[ty],yerr=[[ty-yb[0]],[yb[1]-ty]],xerr=[[tx-xb[0]],[xb[1]-tx]],
            fmt='*',ms=20,c='crimson',label='CAMELS truth (±cosmic var)',zorder=5)
ax.set_xlabel('$\\log_{10}Y(r)$'); ax.set_ylabel('$f_b(r)$')
ax.set_title(f'Truth on the BIND joint  r={r[ri]:.0f} kpc/h, {lbl}')
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG/'real_truth_on_joint.png',dpi=130); print('saved real_truth_on_joint.png'); plt.close(fig)""")

md(r"""## 3. Predict truth f_b(r) from truth observables (noise-free)
Train the Ridge map on **all 256 BIND designs** (the test point is truth, so no fold
needed), then feed the **truth** observable profiles. Compare predicted vs truth
f_b(r) per mass bin. This is the headline real-data result before adding noise.""")

co(r"""LADDERS={'Y':['Y'],'Y+SX':['Y','SX'],'Y+SX+T+S+P':['Y','SX','T','S','P']}
LOGF={'Y','SX','P','T','S'}
def logf(a,name): return np.log10(np.clip(a,1e-30,None)) if name in LOGF else a
def feat_bind(cols,b): return np.hstack([logf(Bp[c][:,b,:],c) for c in cols])
def feat_obs(cols,b,obs): return np.hstack([logf(obs[c][b][None,:],c) for c in cols])

def predict(cols,b,obs,alpha=10.0):
    Xtr=feat_bind(cols,b); ytr=Bp['fb'][:,b,:]
    xs=StandardScaler().fit(Xtr); ys=StandardScaler().fit(ytr)
    reg=Ridge(alpha=alpha).fit(xs.transform(Xtr),ys.transform(ytr))
    return ys.inverse_transform(reg.predict(xs.transform(feat_obs(cols,b,obs))))[0]

print('NOISE-FREE: RMS|pred-truth| in f_b (x1e3), over radii, per bin & ladder')
print(f'{"ladder":14s}'+''.join(f'{l:>16s}' for l in MLBL))
pred_nf={}
for L,cols in LADDERS.items():
    row=f'{L:14s}'
    for b,l in enumerate(MLBL):
        p=predict(cols,b,truth); pred_nf[(L,l)]=p
        rms=np.sqrt(np.nanmean((p-truth['fb'][b])**2))
        row+=f'{rms*1e3:15.2f} '
    print(row)""")

md(r"""## 4. Add realistic observational noise to the "observations"
A real stacked measurement of Y(r), SX(r) is noisy. We inject **multiplicative
log-normal** noise per radial bin (fractional level `f`), independently on Y and SX,
and Monte-Carlo the prediction (300 draws). We sweep `f ∈ {0, 5, 10, 20}%` and show
the predicted f_b(r) ± noise band against truth (with its cosmic-variance band).
(This is deliberately simple and uniform-in-radius; a real analysis would use the
per-bin S/N of the stack — the outskirts are noisier — so treat these as indicative.)""")

co(r"""def noisy_obs(obs,f):
    o={k:obs[k].copy() for k in obs}
    for k in ['Y','SX']:
        o[k]=obs[k]*np.exp(RNG.normal(0,f,obs[k].shape))
    return o

NOISE=[0.0,0.05,0.10,0.20]; cols=['Y','SX']; nmc=300
fig,axes=plt.subplots(1,len(MLBL),figsize=(14,4.2))
for b,(ax,l) in enumerate(zip(axes,MLBL)):
    tb=band('fb')[:,b,:]
    ax.fill_between(r,tb[0],tb[1],color='crimson',alpha=0.2,label='truth ±cosmic var')
    ax.plot(r,truth['fb'][b],'-',c='crimson',lw=2,label='CAMELS truth $f_b$')
    for f,c in zip(NOISE,['C0','C1','C2','C3']):
        draws=np.array([predict(cols,b,noisy_obs(truth,f)) for _ in range(nmc if f>0 else 1)])
        if f==0:
            ax.plot(r,draws[0],'--',c='k',lw=1.5,label='predicted (noise-free)')
        else:
            lo,hi=np.percentile(draws,[16,84],0)
            ax.fill_between(r,lo,hi,color=c,alpha=0.25)
            ax.plot(r,draws.mean(0),'-',c=c,lw=1,label=f'pred, {int(f*100)}% noise')
    ax.set_xscale('log'); ax.set_xlabel('r [kpc/h]'); ax.set_title(l); ax.set_ylim(0.06,0.20)
axes[0].set_ylabel('$f_b(r)$'); axes[0].legend(fontsize=7)
fig.suptitle('Real-data validation: predict CAMELS-truth $f_b(r)$ from truth (Y,SX) profiles + obs noise')
fig.tight_layout(); fig.savefig(FIG/'real_pred_vs_truth.png',dpi=130); print('saved real_pred_vs_truth.png'); plt.close(fig)""")

md("## 5. Error vs noise level, and verdict")

co(r"""import json
print('RMS|pred-truth| in f_b (x1e3) vs obs-noise level, ladder Y+SX')
print(f'{"bin":14s}'+''.join(f'{int(f*100):>8d}%' for f in NOISE))
out={'mass_bins':MLBL,'r_kpc':r.tolist(),'noise_levels':NOISE,'rms_vs_noise':{}}
for b,l in enumerate(MLBL):
    row=f'{l:14s}'; rec=[]
    for f in NOISE:
        draws=np.array([predict(['Y','SX'],b,noisy_obs(truth,f)) for _ in range(300 if f>0 else 1)])
        rms=np.sqrt(np.nanmean((draws.mean(0)-truth['fb'][b])**2))
        row+=f'{rms*1e3:8.2f} '; rec.append(float(rms))
    print(row); out['rms_vs_noise'][l]=rec
# reference scales
out['truth_fb']={l:truth['fb'][b].tolist() for b,l in enumerate(MLBL)}
out['pred_noisefree_YSX']={l:pred_nf[('Y+SX',l)].tolist() for l in MLBL}
json.dump(out,open(FIG/'realdata_results.json','w'),indent=2)
print('\nFor scale: truth f_b spans ~0.08-0.18; BIND design scatter sigma_marg(r) ~ 0.02-0.03.')
print('wrote',FIG/'realdata_results.json')""")

md(r"""## 6. Reading the result
- **§2** tells you whether the relation transfers at all: truth on the BIND joint.
- **§3/§4** quantify the recovered f_b(r): if `RMS|pred−truth|` is small compared to
  the truth f_b range (~0.08–0.18) and to the BIND feedback scatter σ_marg(r)
  (~0.02–0.03), the map predicts real hydro f_b from real hydro observables. Noise
  inflates the band; the noise level at which the prediction stops tracking the truth
  core/outskirt shape is the practical S/N requirement for a real measurement.
- **Caveats (carry to any write-up):** single feedback point (interpolation), shared
  DMO halos, projected f_b, uncalibrated SX, simple uniform noise. A clean off-grid /
  held-out-halo / real-instrument test remains the next step.""")

nb['cells']=cells
nb.metadata['kernelspec']={'name':'python3','display_name':'torch3','language':'python'}
nbf.write(nb,'profile_fb_realdata.ipynb')
print('wrote profile_fb_realdata.ipynb with',len(cells),'cells')
