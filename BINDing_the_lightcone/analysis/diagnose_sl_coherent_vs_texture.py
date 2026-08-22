"""Split BIND's excess small-scale power into coherent vs stochastic parts.

The twobound replicas 0018/0049/0053 are independent sampler draws at the
identical fiducial parameters on the identical halos. For two draws a, b of
the same halo:

    P_auto  = <|F_a|^2>            (what enters the maps and S(ell))
    P_cross = <Re F_a F_b^*>       (the coherent, draw-independent part)
    P_stoch = P_auto - P_cross     (single-draw texture power)

If BIND's excess over the truth lives in P_stoch, the model's *mean* structure
is right and the under-suppression is sampler texture; if it lives in P_cross,
the mean structure itself is insufficiently suppressed.

Run:  /mnt/home/mlee1/venvs/BIND_env/bin/python diagnose_sl_coherent_vs_texture.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CEPH = Path("/mnt/home/mlee1/ceph")
RUNS = {r: CEPH / f"bind_science/runs/twobound/run_{r:04d}/snap_096" for r in (18, 49, 53)}
TRU = CEPH / "bind_science/runs/truth/run_0000/snap_096"
OUT = Path(__file__).resolve().parent
NPIX, LPATCH = 128, 6.25

kx = np.fft.fftfreq(NPIX, d=LPATCH / NPIX) * 2 * np.pi
kk = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
KEDGE = np.linspace(1.0, 55.0, 28)
KC = 0.5 * (KEDGE[1:] + KEDGE[:-1])
KIDX = np.digitize(kk.ravel(), KEDGE) - 1


def bin_p(p):
    out = np.empty(len(KC))
    for b in range(len(KC)):
        m = KIDX == b
        out[b] = p[:, m].mean() if m.any() else np.nan
    return out


CH = {"dm": 0, "gas": 1, "st": 2, "tot": None}
res = {}
for sl in range(4):
    F = {}
    for r, path in RUNS.items():
        d = np.load(path / f"composite_slab{sl:02d}.npz")
        arr = np.asarray(d["generated_patches"], np.float64)
        F[r] = {"dm": arr[:, 0], "gas": arr[:, 1], "st": arr[:, 2],
                "tot": arr.sum(1)}
    t = np.load(TRU / f"composite_slab{sl:02d}.npz")
    ta = np.asarray(t["generated_patches"], np.float64)
    T = {"dm": ta[:, 0], "gas": ta[:, 1], "st": ta[:, 2], "tot": ta.sum(1)}
    for ch in CH:
        fa = {r: np.fft.fft2(F[r][ch]) for r in RUNS}
        ft = np.fft.fft2(T[ch])
        auto = np.mean([(fa[r] * np.conj(fa[r])).real for r in RUNS], 0)
        cross = np.mean([(fa[a] * np.conj(fa[b])).real
                         for a in RUNS for b in RUNS if a < b], 0)
        ptru = (ft * np.conj(ft)).real
        d = res.setdefault(ch, {"auto": [], "cross": [], "tru": []})
        d["auto"].append(auto.reshape(len(auto), -1))
        d["cross"].append(cross.reshape(len(cross), -1))
        d["tru"].append(ptru.reshape(len(ptru), -1))
    print(f"slab {sl} done ({len(ta)} halos)")

print(f"\n{'':8s}" + "".join(f"  k={k:5.1f}" for k in (2.5, 5, 8, 12, 20, 30)))
ks = [int(np.argmin(np.abs(KC - k))) for k in (2.5, 5, 8, 12, 20, 30)]
store = {"k": KC}
for ch in CH:
    A = bin_p(np.concatenate(res[ch]["auto"]))
    X = bin_p(np.concatenate(res[ch]["cross"]))
    T_ = bin_p(np.concatenate(res[ch]["tru"]))
    store[f"{ch}_auto"], store[f"{ch}_cross"], store[f"{ch}_tru"] = A, X, T_
    print(f"[{ch}]")
    print(f"{'auto/tru':>10s}" + "".join(f"  {A[i]/T_[i]:7.4f}" for i in ks))
    print(f"{'cross/tru':>10s}" + "".join(f"  {X[i]/T_[i]:7.4f}" for i in ks))
    print(f"{'stoch/auto':>10s}" + "".join(f"  {(A[i]-X[i])/A[i]:7.4f}" for i in ks))
np.savez(OUT / "sl_coherent_vs_texture.npz", **store)
print(f"\nwrote {OUT / 'sl_coherent_vs_texture.npz'}")
