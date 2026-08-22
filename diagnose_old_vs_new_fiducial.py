"""Why did the WRONG (CAMELS-conditioned) fiducial produce a near-perfect S(ell)?

Three-way patch-level power comparison at identical halos, snap096:
  old = bind_lightcone_tng          (CAMELS cosmology conditioning, retired)
  new = twobound/run_0049            (TNG300 cosmology, canonical)
  tru = runs/truth/run_0000          (hydro-pasted TNG300 particles)

Hypothesis: the old conditioning (sigma8 0.80 vs 0.816, Om 0.30 vs 0.309)
painted puffier halos whose missing small-scale power accidentally cancelled
the sampler-texture excess, while its high Ob/Om overfilled the gas channel.
"""
import numpy as np
from pathlib import Path
CEPH=Path("/mnt/home/mlee1/ceph")
DIRS={"old":CEPH/"bind_lightcone_tng","new":CEPH/"bind_science/runs/twobound/run_0049",
      "tru":CEPH/"bind_science/runs/truth/run_0000"}
NPIX,L=128,6.25
kx=np.fft.fftfreq(NPIX,d=L/NPIX)*2*np.pi
kk=np.sqrt(kx[:,None]**2+kx[None,:]**2)
KEDGE=np.linspace(1.0,55.0,28); KC=0.5*(KEDGE[1:]+KEDGE[:-1])
KIDX=np.digitize(kk.ravel(),KEDGE)-1
def pk(f):
    F=np.fft.fft2(f,axes=(-2,-1)); p=(F*np.conj(F)).real.reshape(len(f),-1)
    return np.array([p[:,KIDX==b].mean(1) if (KIDX==b).any() else np.full(len(f),np.nan)
                     for b in range(len(KC))]).T
P={t:{c:[] for c in("dm","gas","st","tot")} for t in DIRS}
MASS={t:{c:0.0 for c in("dm","gas","st")} for t in DIRS}
for sl in range(4):
    ref=None
    for t,base in DIRS.items():
        sub = "snap_096" if t!="old" else "snap_096"
        d=np.load(base/sub/f"composite_slab{sl:02d}.npz")
        if ref is None: ref=d["halo_centers"]
        assert np.allclose(d["halo_centers"],ref),f"{t} slab{sl} halo mismatch"
        a=np.asarray(d["generated_patches"],np.float64)
        for i,c in enumerate(("dm","gas","st")):
            P[t][c].append(pk(a[:,i])); MASS[t][c]+=a[:,i].sum()
        P[t]["tot"].append(pk(a.sum(1)))
    print(f"slab {sl} ok")
ks=[int(np.argmin(np.abs(KC-k))) for k in (2.5,5,8,12,20,30)]
print("\npatch mass ratios (over all 2933 halos):")
for c in ("dm","gas","st"):
    print(f"  {c:4s}: old/tru {MASS['old'][c]/MASS['tru'][c]:.4f}   new/tru {MASS['new'][c]/MASS['tru'][c]:.4f}")
print(f"\n{'':26s}"+"".join(f"  k={KC[i]:5.1f}" for i in ks))
for c in ("dm","gas","st","tot"):
    A={t:np.nanmean(np.concatenate(P[t][c]),0) for t in DIRS}
    print(f"[{c}]")
    for lab,num in (("old/tru","old"),("new/tru","new"),("old/new","old")):
        den=A["tru"] if lab!="old/new" else A["new"]
        print(f"  {lab:24s}"+"".join(f"  {(A[num]/den)[i]:7.4f}" for i in ks))
np.savez("old_vs_new_fiducial_pk.npz",k=KC,**{f"{t}_{c}":np.nanmean(np.concatenate(P[t][c]),0)
        for t in DIRS for c in ("dm","gas","st","tot")})
print("\nwrote old_vs_new_fiducial_pk.npz")
