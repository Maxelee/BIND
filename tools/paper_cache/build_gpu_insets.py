#!/usr/bin/env python3
"""GPU inset caches for the BIND2 paper figures (Figs 8-11 / appendices).

These four figures each need live `Model.generate` calls; this script runs them
once on a GPU and saves compact npz caches the load-only notebooks read. Small
and fast (seconds-minutes each on an A100). Lifts the generate logic from the
original paper_figures2 cells verbatim (plotting removed).

    python build_gpu_insets.py --which all
    python build_gpu_insets.py --which covering        # Fig 8 low-mass covering + Rc
    python build_gpu_insets.py --which redshift        # Fig 9 redshift response (fm_redshift)
    python build_gpu_insets.py --which vdm             # Fig 10 VDM vs FM
    python build_gpu_insets.py --which obs             # Fig 11 observable-conditioned

Outputs land in PAPER_CACHE_DIR: covering.npz, redshift_evo.npz, vdm_fm.npz,
observables.npz.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_config as C  # noqa: E402

# Data source for the single-sim insets (the 1e12 low-mass eval of CV/sim_0).
_CVSIM = "/mnt/home/mlee1/ceph/fm_lowmass/CV/sim_0/snap_090"
_MT = f"{_CVSIM}/mass_threshold_1p000e12"
_FOF = "/mnt/ceph/users/camels/FOF_Subfind/IllustrisTNG_DM/L50n512/CV/CV_0/fof_subhalo_tab_090.hdf5"

RUNS = "/mnt/home/mlee1/ceph/fm_runs"
CKPT = {
    "fm_thermo": f"{RUNS}/fm_thermo/checkpoints/kept/keep_epoch064_ema.ckpt",
    "fm_redshift": sorted(glob.glob(f"{RUNS}/fm_redshift/checkpoints/*.ckpt")),
    "vdm": sorted(glob.glob(f"{RUNS}/vdm/checkpoints/*.ckpt")),
    "fm_observables_masked": sorted(glob.glob(f"{RUNS}/fm_observables_masked/checkpoints/*.ckpt")),
}


# ════════════════════════════════════════════════════════════════════════════
# Fig 8 — low-mass covering paint + generated closure-radius aperture
# ════════════════════════════════════════════════════════════════════════════
def build_covering(floor=1e10):
    import h5py
    from scipy.ndimage import gaussian_filter
    from bind.inference.artifacts import load_full_maps, load_halo_catalog
    from bind.inference.pipeline import extract_multiscale, circular_taper_weight
    from bind.inference.paint import Model
    from bind.metrics import power_spectrum_pylians_2d

    L, NPIX, PP = 50.0, 1024, 128
    MPP = L / NPIX
    PPM = NPIX / L
    HALF = (PP / 2) * MPP
    RF, TAP, FCOS = 4.0, 0.15, 0.049 / 0.30
    FGAS, SIG = 0.15, 2.0

    dmo, truth = load_full_maps(f"{_CVSIM}/full_maps.npz")
    hc = load_halo_catalog(f"{_MT}/halo_catalog.npz")
    sim_params = np.asarray(hc[0][0]["params"], np.float32)
    with h5py.File(_FOF, "r") as h:
        M200 = h["Group/Group_M_Crit200"][:] * 1e10
        POS = h["Group/GroupPos"][:] / 1e3
        R200 = h["Group/Group_R_Crit200"][:] / 1e3
    GASBG = FGAS * gaussian_filter(dmo, SIG, mode="wrap").astype(np.float32)
    model = Model.from_files(CKPT["fm_thermo"], f"{RUNS}/fm_thermo/norm_stats.npz", device="cuda")

    def plan(fl):
        sel = np.where(M200 > fl)[0]
        Cc = POS[sel, :2].astype(float)
        R = R200[sel]
        ap = RF * R
        order = np.argsort(-M200[sel])
        cov = np.zeros(len(sel), bool)
        ci = []
        box = np.full(len(sel), -1)
        for oi in order:
            if cov[oi]:
                continue
            ci.append(oi)
            unc = np.where(~cov)[0]
            d = Cc[unc] - Cc[oi]
            d -= L * np.round(d / L)
            f = (np.abs(d[:, 0]) + ap[unc] < HALF) & (np.abs(d[:, 1]) + ap[unc] < HALF)
            f[unc == oi] = True
            box[unc[f]] = len(ci) - 1
            cov[unc[f]] = True
        return dict(M=M200[sel], R=R, ci=np.array(ci), box=box,
                    px=(Cc[:, 0] * PPM).astype(int) % NPIX, py=(Cc[:, 1] * PPM).astype(int) % NPIX)

    def generate(P):
        cut = [{"condition": extract_multiscale(dmo, P["px"][i], P["py"][i], PP, MPP)[0],
                "large_scale": extract_multiscale(dmo, P["px"][i], P["py"][i], PP, MPP)[1]} for i in P["ci"]]
        return model.generate(cut, sim_params, n_steps=20, batch_size=32, use_amp=True, progress=False)[:, :3].astype(np.float32)

    yy, xx = np.mgrid[0:PP, 0:PP]
    RRi = np.round(np.hypot(xx - PP // 2, yy - PP // 2)).astype(int).ravel()
    NRi = RRi.max() + 1

    def rolled(P, G, j):
        g = P["box"][j]
        c = P["ci"][g]
        dx = (P["px"][j] - P["px"][c] + NPIX // 2) % NPIX - NPIX // 2
        dy = (P["py"][j] - P["py"][c] + NPIX // 2) % NPIX - NPIX // 2
        return np.roll(G[g], shift=(-dx, -dy), axis=(1, 2)), max(3, PP // 2 - 2 - max(abs(dx), abs(dy)))

    def aperture(P, G, mode):
        a = np.zeros(len(P["M"]))
        for j in range(len(P["M"])):
            if P["box"][j] < 0:
                continue
            if mode == "4R200":
                a[j] = np.clip(RF * P["R"][j] * PPM, 1.5, PP // 2 - 2)
            else:
                patch, cap = rolled(P, G, j)
                bar = (patch[1] + patch[2]).ravel()
                tot = patch.sum(0).ravel()
                fb = np.cumsum(np.bincount(RRi, weights=bar, minlength=NRi)) / np.maximum(
                    np.cumsum(np.bincount(RRi, weights=tot, minlength=NRi)), 1e-30)
                hit = np.where(fb >= FCOS)[0]
                a[j] = int(np.clip(hit[0] if len(hit) else NRi - 1, 2, cap))
        return a

    def paint(P, G, aper, gas_bg=True):
        canvas = np.zeros((3, NPIX, NPIX), np.float32)
        wacc = np.zeros((NPIX, NPIX), np.float32)
        wc = {}
        ar = np.arange(PP)

        def gw(a):
            k = round(a * 2) / 2
            if k not in wc:
                wc[k] = circular_taper_weight(PP, r_pix=k, taper_frac=TAP).astype(np.float32)
            return wc[k]

        for j in range(len(P["M"])):
            if P["box"][j] < 0:
                continue
            patch, _ = rolled(P, G, j)
            w = gw(aper[j])
            ix = (P["px"][j] - PP // 2 + ar) % NPIX
            iy = (P["py"][j] - PP // 2 + ar) % NPIX
            dw = dmo[np.ix_(ix, iy)]
            patch = patch * (float((dw * w).sum()) / (float((patch.sum(0) * w).sum()) + 1e-30))
            gg = np.ix_(ix, iy)
            for ch in range(3):
                canvas[ch][gg] += patch[ch] * w
            wacc[gg] += w
        canvas /= np.where(wacc > 0, wacc, 1.0)[None]
        al = np.clip(wacc, 0, 1)
        comp = np.empty((3, NPIX, NPIX), np.float32)
        if gas_bg:
            comp[0] = (1 - al) * (1 - FGAS) * dmo + al * canvas[0]
            comp[1] = (1 - al) * GASBG + al * canvas[1]
            comp[2] = al * canvas[2]
        else:
            comp[0] = (1 - al) * dmo + al * canvas[0]
            comp[1] = al * canvas[1]
            comp[2] = al * canvas[2]
        comp *= dmo.sum() / (comp.sum() + 1e-30)
        return comp

    def pk(f):
        k, p, _ = power_spectrum_pylians_2d(f, box_size=L, MAS="None")
        return k, p

    P = plan(floor)
    G = generate(P)
    ap4 = aperture(P, G, "4R200")
    apr = aperture(P, G, "closure")
    comp4 = paint(P, G, ap4)
    comprc = paint(P, G, apr)
    k, pt = pk(truth.sum(0))
    _, p4 = pk(comp4.sum(0))
    _, prc = pk(comprc.sum(0))

    lm = np.log10(P["M"])
    eb = np.arange(10, 14.01, 0.3)
    mids, rcM, r4M = [], [], []
    for lo, hi in zip(eb[:-1], eb[1:]):
        m = (lm >= lo) & (lm < hi) & (P["box"] >= 0)
        if m.sum() < 5:
            continue
        mids.append(10 ** (0.5 * (lo + hi)))
        rcM.append(np.median(apr[m]) * MPP)
        r4M.append(np.median(ap4[m]) * MPP)

    out = C.CACHE_DIR / "covering.npz"
    np.savez_compressed(
        out, floor=floor, n_halos=len(P["M"]), n_gens=len(P["ci"]),
        k=k, pk_truth=pt, pk_4r200=p4, pk_rc=prc,
        mids=np.array(mids), rcM=np.array(rcM), r4M=np.array(r4M),
        dmo=dmo.astype(np.float32), truth=truth.astype(np.float32), comp_rc=comprc.astype(np.float32),
    )
    print(f"  wrote {out}  ({len(P['M'])} halos -> {len(P['ci'])} gens, {len(P['M'])/len(P['ci']):.0f}x)")


# ════════════════════════════════════════════════════════════════════════════
# Fig 9 — redshift-conditioned response (fm_redshift)
# ════════════════════════════════════════════════════════════════════════════
def build_redshift(z_levels=(0.0, 0.5, 1.0, 2.0), n_stack=24):
    from bind.inference.paint import Model
    from bind.inference.artifacts import load_halo_catalog, load_halo_cutouts

    rs = Model.from_files(CKPT["fm_redshift"][0], f"{RUNS}/fm_redshift/norm_stats.npz", device="cuda")
    cuts = load_halo_cutouts(f"{_MT}/halo_cutouts.npz")
    halos, hmass, _, _ = load_halo_catalog(f"{_MT}/halo_catalog.npz")
    params = np.asarray(halos[0]["params"], np.float32)
    jmax = int(np.argmax([h["halo_mass"] for h in halos]))

    # (a) hero group z-response panels: (nz, 7, 128, 128)
    gz = np.stack([rs.generate([cuts[jmax]], params, redshift=z, n_steps=20, progress=False)[0]
                   for z in z_levels]).astype(np.float32)

    # (b) stacked BIND amplitude vs z for the n_stack most massive groups
    order = np.argsort([-h["halo_mass"] for h in halos])[:n_stack]
    cl = [cuts[i] for i in order]
    amp = np.zeros((len(z_levels), 3))  # Gas mass, mean y, mean T over positive px
    for zi, z in enumerate(z_levels):
        g = rs.generate(cl, params, redshift=z, n_steps=20, progress=False)  # (n,7,H,W)
        amp[zi, 0] = np.mean(g[:, 1].sum((1, 2)))                # Gas patch mass
        amp[zi, 1] = np.mean([im[im > 0].mean() if (im > 0).any() else np.nan for im in g[:, 3]])
        amp[zi, 2] = np.mean([im[im > 0].mean() if (im > 0).any() else np.nan for im in g[:, 4]])

    out = C.CACHE_DIR / "redshift_evo.npz"
    np.savez_compressed(out, z_levels=np.array(z_levels), hero_logM=np.log10(halos[jmax]["halo_mass"]),
                        gz=gz, amp=amp, n_stack=n_stack)
    print(f"  wrote {out}  (hero group logM={np.log10(halos[jmax]['halo_mass']):.2f}, z={list(z_levels)})")


# ════════════════════════════════════════════════════════════════════════════
# Fig 10 — VDM vs Flow-Matching (sharpness + high-k patch P(k))
# ════════════════════════════════════════════════════════════════════════════
def build_vdm(n_groups=48):
    from bind.inference.paint import Model
    from bind.inference.artifacts import load_halo_catalog, load_halo_cutouts
    from bind.metrics import power_spectrum_2d

    fm = Model.from_files(CKPT["fm_redshift"][0], f"{RUNS}/fm_redshift/norm_stats.npz", device="cuda")
    vdm = Model.from_files(CKPT["vdm"][0], f"{RUNS}/vdm/norm_stats.npz", device="cuda")
    cuts = load_halo_cutouts(f"{_MT}/halo_cutouts.npz")
    halos, _, _, _ = load_halo_catalog(f"{_MT}/halo_catalog.npz")
    params = np.asarray(halos[0]["params"], np.float32)
    sel = np.argsort([-h["halo_mass"] for h in halos])[:n_groups]
    cl = [cuts[i] for i in sel]
    gfm = fm.generate(cl, params, redshift=0.0, n_steps=20, progress=False)[:, :3]
    gvd = vdm.generate(cl, params, n_steps=20, progress=False)[:, :3]

    def stackpk(patches):
        ps = []
        k0 = None
        for p in patches:
            k, pk = power_spectrum_2d(p[1], box_size=6.25)[:2]
            ps.append(pk)
            k0 = k
        return k0, np.median(np.array(ps), axis=0)

    kf, pf = stackpk(gfm)
    kv, pv = stackpk(gvd)
    out = C.CACHE_DIR / "vdm_fm.npz"
    np.savez_compressed(out, fm_gas=gfm[0, 1].astype(np.float32), vdm_gas=gvd[0, 1].astype(np.float32),
                        kf=kf, pf=pf, kv=kv, pv=pv,
                        highk_ratio=float(np.median(pv[kv > 20] / pf[kf > 20])))
    print(f"  wrote {out}  (VDM/FM high-k Gas power ratio (k>20) = {float(np.median(pv[kv>20]/pf[kf>20])):.2f})")


# ════════════════════════════════════════════════════════════════════════════
# Fig 11 — observable-conditioned emulation (fm_observables_masked, M+Y+Tx)
# ════════════════════════════════════════════════════════════════════════════
def build_obs(n_halos=32):
    import torch
    sys.path.insert(0, "/mnt/home/mlee1/vdm_bind2/examples")
    import fb_predict as fbp
    from bind.inference.paint import Model
    from bind.inference.pipeline import build_observable_vectors, extract_periodic_cutout
    from bind.inference.artifacts import load_full_maps, load_halo_catalog, load_halo_cutouts
    from bind.data import N_OBS, log_transform

    obm = Model.from_files(CKPT["fm_observables_masked"][0],
                           f"{RUNS}/fm_observables_masked/norm_stats.npz", device="cuda")
    ns = obm.norm_stats
    dmo0, truth0 = load_full_maps(f"{_CVSIM}/full_maps.npz")
    tth = np.load(f"{_MT}/truth_thermo_patches.npz")["truth_thermo"]
    cuts = load_halo_cutouts(f"{_MT}/halo_cutouts.npz")
    halos, _, _, _ = load_halo_catalog(f"{_MT}/halo_catalog.npz")
    ppm = 1024 / 50.0
    sel = np.argsort([-h["halo_mass"] for h in halos])[:n_halos]

    def tmass(i):
        cx = int(halos[i]["halo_center"][0] * ppm) % 1024
        cy = int(halos[i]["halo_center"][1] * ppm) % 1024
        return np.stack([extract_periodic_cutout(truth0[c], cx, cy, 128) for c in range(3)])

    tmp = np.stack([tmass(int(i)) for i in sel])
    hsub = [halos[int(i)] for i in sel]
    obs = build_observable_vectors(tmp, tth[sel], hsub, ns)
    obs_unpacked = obs[:, :N_OBS] if obs.shape[1] == 2 * N_OBS else obs
    keep = fbp.subset_keep(["M_200", "Y_200", "Tx_200"])

    def norm_cl(i):
        c = (log_transform(cuts[i]["condition"])[None] - ns.cond_mean) / (ns.cond_std + 1e-8)
        l = (log_transform(cuts[i]["large_scale"]) - ns.ls_mean[:, None, None]) / (ns.ls_std[:, None, None] + 1e-8)
        return c, l

    ct = torch.from_numpy(np.stack([norm_cl(int(i))[0] for i in sel]).astype(np.float32))
    lt = torch.from_numpy(np.stack([norm_cl(int(i))[1] for i in sel]).astype(np.float32))
    gob = fbp.generate(obm.fm, ns, ct, lt, obs_unpacked, keep, obm.device, n_steps=20)  # (N,7,128,128)

    out = C.CACHE_DIR / "observables.npz"
    np.savez_compressed(out, subset="M+Y+Tx",
                        truth_gas=tmp[:, 1].astype(np.float32), gen_gas=gob[:, 1].astype(np.float32),
                        truth_gas_mass=tmp[:, 1].sum((1, 2)), gen_gas_mass=gob[:, 1].sum((1, 2)))
    print(f"  wrote {out}  ({n_halos} halos, M+Y+Tx conditioning)")


BUILDERS = {"covering": build_covering, "redshift": build_redshift, "vdm": build_vdm, "obs": build_obs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=[*BUILDERS, "all"])
    args = ap.parse_args()
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = list(BUILDERS) if args.which == "all" else [args.which]
    for name in todo:
        print(f"=== build_gpu_insets: {name} ===")
        BUILDERS[name]()


if __name__ == "__main__":
    main()
