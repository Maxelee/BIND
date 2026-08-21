"""Per-halo posterior calibration figure (referee I-4) — fig_pit_perhalo.

Reads the spine 200x100 ensemble and renders a 2x3 panel:
top = PIT histograms (raw grey filled + suite-debiased colored step),
bottom = coverage curves (raw + debiased vs nominal).

Run:  python f9_pit_calibration.py
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kstest

import os
NPZ=os.environ.get('PIT_NPZ','/mnt/home/mlee1/ceph/paper_cache/fm_two_head/posterior_calibration/ensemble_200x100.npz')
OUT='/mnt/home/mlee1/vdm_bind2/examples/paper_figures/fig_pit_perhalo'
d=np.load(NPZ, allow_pickle=True)
names=[str(x) for x in d['names']]; S=d['draws']; T=d['truth']
suites=np.array([str(s) for s in d['suite']])
rng=np.random.default_rng(0)

CH=[('M_DM_hydro_r200','DM (hydro)','#4C72B0'),('M_Gas_r200','Gas','#55A868'),('M_Stars_r200','Stars','#DD8452')]
def pit(s,tr):
    return (np.sum(s<tr[:,None],axis=1)+rng.random(len(tr))*(1+np.sum(s==tr[:,None],axis=1)))/(s.shape[1]+1)
def debias(s,tr):
    s2=s.copy()
    for su in np.unique(suites):
        m=suites==su; f=np.median(np.median(s[m],axis=1)/tr[m]); s2[m]=s[m]/f
    return s2
def coverage(s,tr,qs):
    cov=[]
    for q in qs:
        lo,hi=np.percentile(s,[50-50*q,50+50*q],axis=1)
        cov.append(np.mean((tr>=lo)&(tr<=hi)))
    return np.array(cov)

qs=np.linspace(0.05,0.99,25)
fig,axes=plt.subplots(2,3,figsize=(10.5,6.2))
for j,(ch,lab,col) in enumerate(CH):
    i=names.index(ch); s=S[:,:,i]; tr=T[:,i]; s2=debias(s,tr)
    r_raw=pit(s,tr); r_deb=pit(s2,tr)
    D1=kstest(r_raw,'uniform'); D2=kstest(r_deb,'uniform')
    ax=axes[0,j]
    ax.hist(r_raw,bins=12,range=(0,1),density=True,color='0.75',alpha=.8,label='as generated')
    ax.hist(r_deb,bins=12,range=(0,1),density=True,histtype='step',lw=2.2,color=col,label='suite-debiased')
    ax.axhline(1,ls='--',c='k',lw=1)
    ax.set_title(lab); ax.set_xlim(0,1); ax.set_ylim(0,2.6)
    ax.text(.03,.95,f"raw $D={D1.statistic:.2f}$\ndebiased $D={D2.statistic:.2f}$ ($p={D2.pvalue:.2f}$)",
            transform=ax.transAxes,va='top',fontsize=8.5)
    if j==0: ax.set_ylabel('PIT density'); ax.legend(fontsize=8,loc='lower center')
    ax.set_xlabel('PIT quantile')
    ax=axes[1,j]
    ax.plot(qs,coverage(s,tr,qs),color='0.6',lw=2,label='as generated')
    ax.plot(qs,coverage(s2,tr,qs),color=col,lw=2.2,label='suite-debiased')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('nominal central probability')
    if j==0: ax.set_ylabel('empirical coverage'); ax.legend(fontsize=8,loc='upper left')
fig.tight_layout()
fig.savefig(OUT+'.pdf'); fig.savefig(OUT+'.png',dpi=300)
print('wrote',OUT)
