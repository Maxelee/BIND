"""Build observable_fb_map.ipynb (the Observable -> f_b PoC notebook).

After this writes the .ipynb, the notebook is authoritative; execute with:
  jupyter nbconvert --execute --inplace \
    --ExecutePreprocessor.kernel_name=torch3 --ExecutePreprocessor.timeout=1800 \
    observable_fb_map.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Observable → f_b — non-parametric mapping (proof-of-concept)

**⚠️ Provisional.** All numbers below are hypotheses pending review of raw outputs.
One subgrid model (CAMELS-TNG), fixed cosmology, scalar integrated quantities only.

**Question.** BIND paints, per cluster-scale halo, both *observables* (tSZ
Compton-Y, X-ray surface brightness SX, gas T/entropy/pressure) and the
*unmeasurable* baryon fraction f_b. Can observables pin down f_b **at fixed mass,
on feedback settings (θ) never seen in training** — i.e. without estimating θ?

**Design.** 256 Sobol θ-designs × 1111 fixed halos (true factorial; same halos
repainted under different feedback) at fixed cosmology. Held-out test =
`GroupKFold(groups=design)` → the parameter-free regime. See
`docs/observable_fb_poc.md`.

Deliverable: σ(f_b | observable set) per mass bin on held-out θ, the
scatter-reduction fraction, a degeneracy diagnostic, and conditional MI.""")

co(r"""import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import mutual_info_regression

SOBOL = Path('/mnt/home/mlee1/ceph/sobol_ss_cv')
FIGDIR = Path('figures/observable_fb'); FIGDIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
F_COSMIC = 0.049/0.30  # Omega_b/Omega_m, fixed cosmology -> 0.1633
print('f_cosmic =', round(F_COSMIC,4))""")

md("## 1. Load & assemble the (design, halo) table")

co(r"""cube = np.load(SOBOL/'cube.npz', allow_pickle=True)
obs = cube['obs']                       # (256,1111,8) Y200,T,S,P,f_gas,m_gen,supp_k10,supp_prof
onames = list(cube['obs_names'])
M200 = np.asarray(cube['M200'], float)  # (1111,) Msun/h
logM = np.log10(M200)
nD, nH, _ = obs.shape
print('designs', nD, 'halos', nH, 'logM range', logM.min().round(3), logM.max().round(3))

extra = np.load(SOBOL/'obs_fb_extra.npz', allow_pickle=True)
ex = extra['extra']                     # (256,1111,4) tau_ksz, SX, f_star, f_b
enames = list(extra['extra_names'])
print('extra cols:', enames, 'shape', ex.shape)

# columns
Y   = obs[:,:,onames.index('Y200')]
T   = obs[:,:,onames.index('T')]
S   = obs[:,:,onames.index('S')]
P   = obs[:,:,onames.index('P')]
fg  = obs[:,:,onames.index('f_gas')]
tau = ex[:,:,enames.index('tau_ksz')]
SX  = ex[:,:,enames.index('SX')]
fstar = ex[:,:,enames.index('f_star')]
fb  = ex[:,:,enames.index('f_b')]

# broadcast per-halo / per-design indexers
logM2  = np.broadcast_to(logM[None,:], (nD,nH))
design = np.broadcast_to(np.arange(nD)[:,None], (nD,nH))

# flatten
def flat(a): return np.asarray(a).reshape(-1)
tab = dict(Y=flat(Y), SX=flat(SX), T=flat(T), S=flat(S), P=flat(P), tau=flat(tau),
           M200=flat(np.broadcast_to(M200[None,:],(nD,nH))), logM=flat(logM2),
           fb=flat(fb), f_gas=flat(fg), f_star=flat(fstar), design=flat(design))
good = np.isfinite(tab['fb']) & np.isfinite(tab['Y']) & np.isfinite(tab['SX'])
for k in tab: tab[k] = tab[k][good]
print('rows', tab['fb'].size, 'dropped', good.size-good.sum())
print('f_b: med %.3f  f_gas med %.3f  f_star med %.4f  f_b/f_cosmic med %.3f'
      % (np.median(tab['fb']), np.median(tab['f_gas']), np.median(tab['f_star']),
         np.median(tab['fb'])/F_COSMIC))""")

md(r"""**Confirm the factorial premise** (same halos across designs → f_b varies for a
fixed halo only through feedback):""")

co(r"""# per-fixed-halo spread of f_b across designs vs across halos
fb_2d = fb  # (256,1111) with nan where mtot<=0
across_design = np.nanstd(fb_2d, axis=0)   # per-halo, over feedback
across_halo   = np.nanstd(fb_2d, axis=1)   # per-design, over halos
print('median std(f_b) across designs (feedback) for a fixed halo:', np.nanmedian(across_design).round(4))
print('median std(f_b) across halos for a fixed design (mass+assembly):', np.nanmedian(across_halo).round(4))
print('-> both non-zero: feedback AND mass/assembly drive f_b, as expected.')""")

md(r"""## 2. Definitions
Mass bins (log10 M200, Msun/h). Conditional scatter σ(f_b|𝒪) from held-out-θ
out-of-fold predictions (`GroupKFold(design)`, 5 folds, `HistGradientBoostingRegressor`).
Marginal σ = in-bin std. Reduction = 1 − σ_cond/σ_marg. Reported with fold spread
and a design-bootstrap.""")

co(r"""MASS_BINS = [(13.0,13.5),(13.5,14.0),(14.0,np.inf)]
MASS_LBL  = ['[13.0,13.5)','[13.5,14.0)','[14.0,+)']

def bin_mask(lo,hi):
    return (tab['logM']>=lo) & (tab['logM']<hi)

for (lo,hi),lbl in zip(MASS_BINS,MASS_LBL):
    m=bin_mask(lo,hi)
    print(f'{lbl}: N={m.sum():6d}  f_b med={np.median(tab["fb"][m]):.3f} '
          f'marg-sigma={tab["fb"][m].std():.4f}')

def oof_predict(X, y, groups, seed=0):
    '''Out-of-fold held-out predictions via GroupKFold on designs.'''
    pred = np.full(y.shape, np.nan)
    gkf = GroupKFold(n_splits=5)
    fold_sigma=[]
    for tr,te in gkf.split(X,y,groups):
        reg = HistGradientBoostingRegressor(random_state=seed, max_iter=200,
                                            learning_rate=0.08)
        reg.fit(X[tr],y[tr])
        pred[te]=reg.predict(X[te])
        fold_sigma.append(np.sqrt(np.mean((y[te]-pred[te])**2)))
    return pred, np.array(fold_sigma)

def design_bootstrap_sigma(resid, groups, nboot=300):
    '''Bootstrap RMS(resid) by resampling whole designs (the independent unit).'''
    uq = np.unique(groups)
    by = {g: resid[groups==g] for g in uq}
    out=[]
    for _ in range(nboot):
        pick = RNG.choice(uq, size=uq.size, replace=True)
        r = np.concatenate([by[g] for g in pick])
        out.append(np.sqrt(np.mean(r**2)))
    return np.percentile(out,[16,50,84])""")

md("## 3. Phase 1 — single observable (Y), held-out-θ scatter")

co(r"""LADDERS = {
 'Y'            : ['Y'],
 'Y+SX'         : ['Y','SX'],
 'Y+SX+T+S+P'   : ['Y','SX','T','S','P'],
 'kSZ-anchor(Y+tau)': ['Y','tau'],   # near-circular: tau = gas mass ~ f_b numerator
}

def features(cols, m, logfeat=True):
    X=np.column_stack([tab[c][m] for c in cols]).astype(float)
    if logfeat:  # observables span decades -> log helps the tree splits little but is robust
        X=np.sign(X)*np.log10(np.abs(X)+1e-30)
    return X

results={}  # (ladder, bin) -> dict
for lbl_l,cols in LADDERS.items():
    for (lo,hi),lbl_b in zip(MASS_BINS,MASS_LBL):
        m=bin_mask(lo,hi)
        y=tab['fb'][m]; g=tab['design'][m]
        X=features(cols,m)
        pred,fold_sig=oof_predict(X,y,g)
        resid=y-pred
        sig_cond=np.sqrt(np.mean(resid**2))
        sig_marg=y.std()
        lo16,med,hi84=design_bootstrap_sigma(resid,g)
        # dex
        dex=np.sqrt(np.mean((np.log10(y)-np.log10(np.clip(pred,1e-6,None)))**2))
        results[(lbl_l,lbl_b)]=dict(sig_cond=sig_cond,sig_marg=sig_marg,
            reduction=1-sig_cond/sig_marg, fold_sig=fold_sig, dex=dex,
            boot=(lo16,med,hi84), pred=pred, y=y, g=g, n=m.sum())

# Phase-1 print
print('PHASE 1 — Y only, held-out theta')
print(f'{"bin":12s} {"N":>7s} {"sig_marg":>9s} {"sig(fb|Y)":>10s} {"reduction":>10s} {"dex":>7s}')
for lbl_b in MASS_LBL:
    r=results[('Y',lbl_b)]
    print(f'{lbl_b:12s} {r["n"]:7d} {r["sig_marg"]:9.4f} {r["sig_cond"]:10.4f} '
          f'{r["reduction"]*100:9.1f}% {r["dex"]:7.3f}')""")

co(r"""# Phase-1 predicted-vs-true (held-out), colored by mass bin
fig,ax=plt.subplots(figsize=(5,5))
colors=['tab:blue','tab:orange','tab:red']
for lbl_b,c in zip(MASS_LBL,colors):
    r=results[('Y',lbl_b)]
    ax.scatter(r['y'],r['pred'],s=2,alpha=0.15,color=c,label=lbl_b)
lim=[tab['fb'].min(),tab['fb'].max()]
ax.plot(lim,lim,'k--',lw=1)
ax.set_xlabel('true $f_b$'); ax.set_ylabel('predicted $f_b$ (held-out θ, from Y)')
ax.set_title('Phase 1: recover $f_b$ from tSZ-Y alone'); ax.legend(markerscale=4)
fig.tight_layout(); fig.savefig(FIGDIR/'phase1_pred_vs_true.png',dpi=130)
print('saved', FIGDIR/'phase1_pred_vs_true.png'); plt.close(fig)""")

md("## 4. Phase 2 — observable ladder + degeneracy diagnostic")

co(r"""print('PHASE 2 — sigma(f_b | obs set), held-out theta  [reduction %]')
hdr=f'{"bin":12s}'+''.join(f'{l:>20s}' for l in LADDERS)
print(hdr)
for lbl_b in MASS_LBL:
    row=f'{lbl_b:12s}'
    for lbl_l in LADDERS:
        r=results[(lbl_l,lbl_b)]
        row+=f'{r["sig_cond"]:.4f}({r["reduction"]*100:4.0f}%)'.rjust(20)
    print(row)
print('\nNote: kSZ-anchor (Y+tau) is near-circular (tau∝gas mass∝f_b numerator);'
      ' it is the upper-bound reference, not an independent-observable result.')""")

co(r"""# Degeneracy diagnostic: residual std(f_b) at fixed (mass, Y) vs fixed (mass, Y, SX)
def cell_resid_std(m, conds, nq=10, min_n=8):
    '''mean over cells of within-cell std(f_b); cells = quantile bins of cond vars.'''
    y=tab['fb'][m]
    digs=[]
    for c in conds:
        v=tab[c][m]
        q=np.quantile(v,np.linspace(0,1,nq+1)); q[-1]+=1e-9
        digs.append(np.clip(np.digitize(v,q[1:-1]),0,nq-1))
    key=digs[0]
    for d in digs[1:]: key=key*nq+d
    stds=[]
    for k in np.unique(key):
        yy=y[key==k]
        if yy.size>=min_n: stds.append(yy.std())
    return np.mean(stds), len(stds)

print('Degeneracy: residual std(f_b) at fixed conditioning (lower = more determined)')
print(f'{"bin":12s} {"marginal":>9s} {"|(mass,Y)":>10s} {"|(mass,Y,SX)":>13s}')
deg={}
for (lo,hi),lbl_b in zip(MASS_BINS,MASS_LBL):
    m=bin_mask(lo,hi)
    marg=tab['fb'][m].std()
    sY ,_=cell_resid_std(m,['Y'],nq=10)
    sYS,_=cell_resid_std(m,['Y','SX'],nq=6)   # 6x6 cells to keep counts up
    deg[lbl_b]=(marg,sY,sYS)
    print(f'{lbl_b:12s} {marg:9.4f} {sY:10.4f} {sYS:13.4f}')""")

co(r"""# Visual: f_b vs Y colored by SX, middle mass bin
lo,hi=MASS_BINS[1]; m=bin_mask(lo,hi)
fig,ax=plt.subplots(figsize=(5.5,4.5))
sc=ax.scatter(tab['Y'][m],tab['fb'][m],c=np.log10(tab['SX'][m]),s=4,alpha=0.4,cmap='viridis')
ax.set_xscale('log'); ax.set_xlabel('$Y_{200}$ (tSZ)'); ax.set_ylabel('$f_b$')
ax.set_title(f'{MASS_LBL[1]}: scatter in $f_b$ at fixed Y, colored by $\\log S_X$')
fig.colorbar(sc,label='$\\log_{10} S_X$'); fig.tight_layout()
fig.savefig(FIGDIR/'phase2_fb_Y_coloredSX.png',dpi=130)
print('saved', FIGDIR/'phase2_fb_Y_coloredSX.png'); plt.close(fig)""")

md("## 5. Phase 3 — conditional information")

co(r"""# R2_info from held-out predictive skill: R2_info = 1 - (sig_cond/sig_marg)^2,
# I = -0.5 ln(1-R2_info).  Plus sklearn per-feature MI for single observables.
print('PHASE 3 — conditional R^2_info and I (nats) per ladder x mass bin')
print(f'{"bin":12s}'+''.join(f'{l:>20s}' for l in LADDERS))
for lbl_b in MASS_LBL:
    row=f'{lbl_b:12s}'
    for lbl_l in LADDERS:
        r=results[(lbl_l,lbl_b)]
        R2=max(0.0,1-(r['sig_cond']/r['sig_marg'])**2)
        I=-0.5*np.log(max(1-R2,1e-9))
        row+=f'R2={R2:.2f} I={I:.2f}'.rjust(20)
    print(row)

print('\nsklearn per-feature MI(f_b; obs | mass bin)  [nats]')
for (lo,hi),lbl_b in zip(MASS_BINS,MASS_LBL):
    m=bin_mask(lo,hi)
    cols=['Y','SX','T','S','P','tau']
    X=features(cols,m); y=tab['fb'][m]
    mi=mutual_info_regression(X,y,random_state=0)
    print(lbl_b, {c:round(float(v),3) for c,v in zip(cols,mi)})""")

md(r"""## 6. Stretch — does the map generalise to other unmeasurable state? (target = entropy K)
Quick check that observables also pin down the gas entropy (another quantity an
observer cannot read off directly). Reported, not headline.""")

co(r"""# target = entropy S (call it K), predict from observables that are NOT S
for tgt,exclude in [('S (entropy)','S')]:
    print(f'target = {tgt}, held-out theta, features Y+SX+T+P')
    cols=['Y','SX','T','P']
    print(f'{"bin":12s} {"sig_marg":>9s} {"sig_cond":>9s} {"reduction":>10s}')
    for (lo,hi),lbl_b in zip(MASS_BINS,MASS_LBL):
        m=bin_mask(lo,hi)
        y=np.log10(tab['S'][m]); g=tab['design'][m]   # entropy spans decades -> log target
        X=features(cols,m)
        pred,_=oof_predict(X,y,g)
        sc=np.sqrt(np.mean((y-pred)**2)); sm=y.std()
        print(f'{lbl_b:12s} {sm:9.4f} {sc:9.4f} {(1-sc/sm)*100:9.1f}%')""")

md("## 7. Save machine-readable results")

co(r"""import json
out={'ladders':{k:v for k,v in LADDERS.items()},
     'mass_bins':MASS_LBL,'f_cosmic':F_COSMIC,
     'phase2':{}, 'degeneracy':{}, 'phase1_dex':{}}
for lbl_l in LADDERS:
    for lbl_b in MASS_LBL:
        r=results[(lbl_l,lbl_b)]
        out['phase2'][f'{lbl_l}|{lbl_b}']=dict(
            n=int(r['n']), sig_marg=float(r['sig_marg']), sig_cond=float(r['sig_cond']),
            reduction=float(r['reduction']), boot16_50_84=[float(x) for x in r['boot']],
            fold_sig=[float(x) for x in r['fold_sig']])
for lbl_b in MASS_LBL:
    out['degeneracy'][lbl_b]=dict(zip(['marginal','at_fixed_Y','at_fixed_Y_SX'],
                                      [float(x) for x in deg[lbl_b]]))
    out['phase1_dex'][lbl_b]=float(results[('Y',lbl_b)]['dex'])
Path('figures/observable_fb').mkdir(parents=True,exist_ok=True)
with open('figures/observable_fb/results.json','w') as f: json.dump(out,f,indent=2)
print('wrote figures/observable_fb/results.json')
print(json.dumps(out['phase2'],indent=1)[:1500])""")

nb['cells']=cells
nb.metadata['kernelspec']={'name':'python3','display_name':'torch3','language':'python'}
nbf.write(nb,'observable_fb_map.ipynb')
print('wrote observable_fb_map.ipynb with',len(cells),'cells')
