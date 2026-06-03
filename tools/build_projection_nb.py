"""Build projection_fb_check.ipynb -- is the >cosmic f_b(r) at ~400 kpc a projection
artifact?  Compares the 50 Mpc/h LOS projection (used in the PoC) against the
6.25 Mpc/h cluster-depth cube, separates differential vs enclosed f_b, and discusses
how line-of-sight contamination of tSZ Y is handled observationally.

Execute:  jupyter nbconvert --execute --inplace
  --ExecutePreprocessor.kernel_name=torch3 --ExecutePreprocessor.timeout=1800
  projection_fb_check.ipynb
"""
import nbformat as nbf
nb=nbf.v4.new_notebook(); cells=[]
md=lambda s: cells.append(nbf.v4.new_markdown_cell(s))
co=lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Is f_b(r) > cosmic at ~400 kpc a projection effect?  (sanity / LOS check)

**The worry.** Our maps are 2-D projections over a **50 Mpc/h** line of sight with the
halo at the centre. The stacked baryon-fraction profile rises above the cosmic value
`f_cosmic = Ω_b/Ω_m = 0.163` at r ≳ 400 kpc/h. A halo cannot contain *more* baryons
than cosmic, so is this a **projection artifact** (uncorrelated LOS structure inflating
the ratio), or real?

**Two things were being conflated, and the test settles it:**
1. **Differential** f_b(r) = Σ_b/Σ_tot in an *annulus* at r. This **can** exceed cosmic
   at intermediate radii because feedback moves gas *outward* from the core — it piles
   up at large r. It is a local enhancement, not a violation of baryon conservation.
2. **Enclosed** f_b(<R200) = the cumulative ratio inside R200 — the actual "missing
   baryons" metric. This must be ≤ cosmic for a baryon-poor halo.

**The lever.** `/mnt/ceph .../fm_testsuite_cube/CV/.../truth_halos_cube.npz` holds the
*same halos* projected over only **6.25 Mpc/h** (cluster depth, same 48.83 kpc/h pixel,
8× shorter LOS). Comparing 6.25 vs 50 Mpc/h isolates the uncorrelated-LOS contribution.
(The cube has no thermo, so this tests the **mass-based f_b**; the tSZ-Y LOS question is
discussed in §3.)""")

co(r"""import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('spr','tools/stack_profiles_reduce.py')
spr=importlib.util.module_from_spec(spec); spec.loader.exec_module(spr)
MASS_BINS,MLBL,azim=spr.MASS_BINS,spr.MASS_LBL,spr.azim
r=spr.R_CEN; PIX_KPC=spr.PIX_KPC; nr=len(r)
FCOS=0.049/0.30; FIG=Path('figures/observable_fb')
CVc=Path('/mnt/home/mlee1/ceph/fm_testsuite_cube/CV')
CV50=Path('/mnt/home/mlee1/ceph/fm_testsuite/CV')
yy,xx=np.mgrid[0:128,0:128]; Rkpc=np.sqrt((xx-64)**2+(yy-64)**2)*PIX_KPC
print('f_cosmic =',round(FCOS,4),'| pixel',round(PIX_KPC,2),'kpc/h')""")

md(r"""## 1. Stack the 6.25 Mpc/h cube truth (same halos, cluster depth)
Accumulate the stacked mass maps [DM,Gas,Stars] per mass bin → differential f_b(r),
enclosed f_b(<r), and the median R200. The uncorrelated cosmic background for a
6.25 Mpc/h depth is `Σ_bg = ρ̄_m·6.25` (= the 50 Mpc/h box-mean column × 6.25/50).""")

co(r"""acc={b:dict(dm=0.0,gas=0.0,star=0.0,n=0,r200=[]) for b in range(3)}
for sd in sorted(CVc.iterdir()):
    md_=sd/'snap_090/mass_threshold_1p000e13'
    if not (md_/'truth_halos_cube.npz').exists(): continue
    c=np.load(md_/'halo_catalog.npz')
    if 'masses' not in c.files or len(c['masses'])==0: continue
    th=np.load(md_/'truth_halos_cube.npz')['truth_halos']        # (n,3,128,128)
    logM=np.log10(np.asarray(c['masses'],float)); r200=np.asarray(c['r200s'],float)*1000  # ->kpc/h
    for b,(lo,hi) in enumerate(MASS_BINS):
        idx=np.where((logM>=lo)&(logM<hi))[0]
        if len(idx)==0: continue
        acc[b]['dm']+=th[idx,0].sum(0); acc[b]['gas']+=th[idx,1].sum(0)
        acc[b]['star']+=th[idx,2].sum(0); acc[b]['n']+=len(idx); acc[b]['r200']+=list(r200[idx])
# 50 Mpc/h box-mean cosmic column
cm=[np.load(sd/'snap_090/full_maps.npz')['truth_maps'].sum(0).mean()
    for sd in sorted(CV50.iterdir())[:8] if (sd/'snap_090/full_maps.npz').exists()]
SIG_BG_50=float(np.mean(cm)); SIG_BG_CUBE=SIG_BG_50*6.25/50
print('cosmic column 50Mpc=%.3e cube=%.3e'%(SIG_BG_50,SIG_BG_CUBE))

cube={}   # per bin: differential fb(r), enclosed fb(<r), R200
for b in range(3):
    if acc[b]['n']==0: continue
    n=acc[b]['n']; gas=acc[b]['gas']/n; star=acc[b]['star']/n; dm=acc[b]['dm']/n
    bar=gas+star; tot=dm+gas+star
    fb_diff=azim(bar)/azim(tot)
    enc=np.array([(bar[Rkpc<=rr].sum())/(tot[Rkpc<=rr].sum()) for rr in r])
    enc_sub=np.array([((bar-FCOS*SIG_BG_CUBE)[Rkpc<=rr].sum())/((tot-SIG_BG_CUBE)[Rkpc<=rr].sum()) for rr in r])
    cube[b]=dict(fb_diff=fb_diff,enc=enc,enc_sub=enc_sub,R200=np.median(acc[b]['r200']),
                 fb_diff_sub=azim(bar-FCOS*SIG_BG_CUBE)/azim(tot-SIG_BG_CUBE))

# 50 Mpc/h differential f_b from the saved truth stack
T=np.load('/mnt/home/mlee1/ceph/sobol_ss_cv/truth_stacked_profiles.npz',allow_pickle=True)
cnt=T['counts'].sum(0); bar50=np.nansum(T['baryon_sum'],0); tot50=np.nansum(T['tot_sum'],0)
fb50={b: bar50[b]/tot50[b] for b in range(3)}
print('enclosed f_b(<R200) cube vs f_cosmic=%.3f:'%FCOS)
for b,l in enumerate(MLBL):
    if b in cube:
        iR=np.argmin(abs(r-cube[b]['R200']))
        print(f'  {l}: R200={cube[b]["R200"]:.0f} kpc  fb(<R200)={cube[b]["enc"][iR]:.3f}  (diff@R200={cube[b]["fb_diff"][iR]:.3f})')""")

md("### 1.1 Differential vs enclosed, and 6.25 vs 50 Mpc/h")

co(r"""fig,axes=plt.subplots(1,3,figsize=(15,4.3))
for b,(ax,l) in enumerate(zip(axes,MLBL)):
    if b not in cube: continue
    ax.plot(r,cube[b]['fb_diff'],'o-',c='C0',ms=4,label='differential, 6.25 Mpc (cube)')
    ax.plot(r,fb50[b],'s--',c='C1',ms=4,label='differential, 50 Mpc (PoC)')
    ax.plot(r,cube[b]['enc'],'^-',c='C2',ms=4,label='enclosed $f_b(<r)$, cube')
    ax.axhline(FCOS,ls=':',c='k',lw=1,label='$f_{cosmic}$')
    ax.axvline(cube[b]['R200'],ls='-',c='0.7',lw=1); ax.text(cube[b]['R200']*1.02,0.07,'R200',color='0.5',fontsize=8)
    ax.set_xscale('log'); ax.set_xlabel('r [kpc/h]'); ax.set_title(l); ax.set_ylim(0.06,0.21)
axes[0].set_ylabel('$f_b$'); axes[0].legend(fontsize=7.5,loc='lower right')
fig.suptitle('Differential $f_b(r)$ overshoots cosmic (real, feedback push-out); enclosed $f_b(<R200)$ stays below cosmic (missing baryons)')
fig.tight_layout(); fig.savefig(FIG/'proj_fb_differential_enclosed.png',dpi=130)
print('saved proj_fb_differential_enclosed.png'); plt.close(fig)

# projection-effect magnitude
print('\nLOS (50 vs 6.25 Mpc) effect on differential f_b: max|Δ| over r')
for b,l in enumerate(MLBL):
    if b in cube:
        d=np.nanmax(np.abs(fb50[b]-cube[b]['fb_diff']))
        print(f'  {l}: max|fb_50-fb_cube|={d:.3f}  (and 50Mpc is pulled TOWARD cosmic at large r)')""")

md(r"""## 2. Verdict on the projection question
- **The >cosmic differential f_b at ~400–600 kpc is real, not projection.** It is present
  at 6.25 Mpc/h cluster depth and after background subtraction; the 50 Mpc/h LOS actually
  pulls the profile *toward* cosmic at large r (adds a cosmic-f_b background), it does not
  inflate it. Max LOS effect on the mass-based f_b is < 0.02, only at r ≳ 600 kpc.
- **It is the differential profile, not the enclosed fraction.** Enclosed f_b(<R200) is
  **below cosmic** (≈0.115/0.147/0.159 → deficit largest at low mass = missing baryons).
  Feedback evacuates the core and deposits gas at intermediate/large radii, so the
  *local* f_b overshoots cosmic there while the *cumulative* stays sub-cosmic. No
  baryon-conservation problem.
- **Implication for the PoC:** the f_b(r) target is a projected differential profile; it is
  robust to LOS depth (mass maps), so the predicted profiles are not a projection
  artifact. If one wants the "missing-baryon" number, report enclosed f_b(<R200).""")

md(r"""## 3. Line-of-sight contamination of tSZ Y — how it's handled, and what we can do

The mass-based f_b above is LOS-robust because mass is concentrated. **tSZ Y is not**:
`Y ∝ ∫ P dl ∝ ∫ n_e T dl` integrates the *diffuse, extended* electron pressure over the
whole LOS, so it picks up (i) the **2-halo / correlated** large-scale pressure and (ii)
**uncorrelated** foreground/background structure. X-ray `S_X ∝ ∫ n_e² Λ(T) dl` is ρ²-
weighted, hence dominated by the dense core and far less LOS-contaminated; mass-f_b least
of all.

**How observers remove it:**
- **Compensated Aperture Photometry (CAP / AP filter):** signal in a disk of radius θ
  *minus* an equal-area surrounding annulus [θ, √2 θ]. A spatially uniform background
  (mean LOS, large-scale modes) cancels exactly → the estimator is insensitive to the
  uncorrelated column. This is the canonical stacked-tSZ/kSZ estimator (e.g. ACT×BOSS).
- **Matched filters** tuned to the expected cluster pressure profile (down-weight scales
  where the background dominates).
- **Mean-background + 2-halo modelling:** subtract the modelled mean-y background and fit
  the 1-halo+2-halo decomposition.
- **X-ray:** blank-sky / local-annulus background subtraction, then **deprojection**
  (Abel inversion or forward-fitting a 3-D model) to recover 3-D n_e, T → 3-D gas mass.

**The profile-level analog we already used** (§1) is **mean-column subtraction**:
remove `Σ_bg` (and `f_cosmic·Σ_bg` from baryons) — the same idea as CAP (null the uniform
background). For the mass maps it barely moved f_b because the halo dominates the column.
For **Y** it would matter much more.""")

co(r"""# Demonstrate a CAP (disk - equal-area annulus) estimator on the stacked TRUTH y-map,
# and a background-subtracted Y profile, to show the mechanic (50 Mpc/h, middle bin).
# Stack truth_thermo y-maps for the middle bin.
ysum=0.0; ntot=0
for sd in sorted(CV50.iterdir()):
    md_=sd/'snap_090/mass_threshold_1p000e13'
    if not (md_/'truth_thermo_patches.npz').exists(): continue
    c=np.load(md_/'halo_catalog.npz')
    if 'masses' not in c.files or len(c['masses'])==0: continue
    logM=np.log10(np.asarray(c['masses'],float))
    idx=np.where((logM>=13.5)&(logM<14.0))[0]
    if len(idx)==0: continue
    ysum=ysum+np.load(md_/'truth_thermo_patches.npz')['truth_thermo'][idx,0].sum(0); ntot+=len(idx)
ymap=ysum/ntot                       # mean stacked y-map (50 Mpc/h depth)
ybg=np.median(ymap[Rkpc>2500])       # uniform-background estimate from far pixels
# CAP(theta): mean(disk<theta) - mean(annulus theta..sqrt2 theta)
th=np.linspace(60,1500,16)
cap=[ymap[Rkpc<t].mean()-ymap[(Rkpc>=t)&(Rkpc<np.sqrt(2)*t)].mean() for t in th]
yprof=spr.azim(ymap); yprof_sub=spr.azim(ymap-ybg)
print('stacked-y background level (far-field median): %.3e ; central y: %.3e'%(ybg,yprof[0]))
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(r,yprof,'o-',label='raw $Y(r)$'); ax[0].plot(r,yprof_sub,'s--',label='background-subtracted')
ax[0].axhline(ybg,ls=':',c='k',lw=1,label='uniform background'); ax[0].set_yscale('log')
ax[0].set_xscale('log'); ax[0].set_xlabel('r [kpc/h]'); ax[0].set_ylabel('stacked y'); ax[0].legend(fontsize=8)
ax[0].set_title('tSZ y profile: raw vs background-subtracted')
ax[1].plot(th,cap,'o-'); ax[1].axhline(0,c='k',lw=0.6); ax[1].set_xlabel(r'aperture $\theta$ [kpc/h]')
ax[1].set_ylabel('CAP$(\\theta)$ = disk $-$ annulus'); ax[1].set_title('Compensated aperture photometry (nulls uniform LOS)')
fig.tight_layout(); fig.savefig(FIG/'proj_tsz_cap_demo.png',dpi=130); print('saved proj_tsz_cap_demo.png'); plt.close(fig)""")

md(r"""## 4. What this means for the project + a concrete next step
- The **target** (mass-based f_b) is LOS-robust, so the PoC's predicted f_b(r) is not a
  projection artifact — good.
- The **observables** Y(r), SX(r) used as *inputs* are themselves LOS-integrated; Y in
  particular carries 2-halo + uncorrelated pressure. In a real application the observables
  should be **CAP-filtered / background-subtracted** before feeding the map. SX is far less
  affected; mass-f_b least.
- **We cannot yet quantify Y's LOS contamination** because the 6.25 Mpc/h cube products
  carry no thermo. **Concrete next step:** generate **6.25 Mpc/h thermo cubes** (compton_y,
  T, S, P) for the CV halos, then repeat the 6.25-vs-50 comparison on Y/SX and re-run the
  map with CAP-filtered observables. Until then, treat the absolute Y normalisation as
  LOS-dependent (the PoC is internally consistent because train and test use the same
  50 Mpc/h projection).""")

nb['cells']=cells
nb.metadata['kernelspec']={'name':'python3','display_name':'torch3','language':'python'}
nbf.write(nb,'projection_fb_check.ipynb')
print('wrote projection_fb_check.ipynb with',len(cells),'cells')
