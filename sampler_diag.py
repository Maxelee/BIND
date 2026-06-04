"""Sampler diagnostic: is BIND's small-scale (high-k) power deficit a SAMPLER
artifact (fixable at inference) or BAKED INTO TRAINING (mean-seeking)?

Two tests, on the most massive halos of one (or more) CV sims:
  (1) n_steps sweep  -> does per-patch high-k power converge toward truth as the
      Euler ODE is refined?  (rises toward 1 => under-integrated sampler.)
  (2) multi-sample dispersion @ fixed n_steps -> do INDIVIDUAL samples carry
      truth-level small-scale power, or are they all smooth?  (single ~ mean ~ <1
      => the model is mean-seeking / under-dispersed -> needs a training fix.)

Compares per-patch P(k) (128px patches, box 6.25 Mpc/h) of generated vs truth,
for the TOTAL field and the STELLAR channel.  Saves a figure + npz and prints a
verdict.  GPU-aware (device="auto").

Env overrides: RUN, SIMS (comma sep), NMAX, NSAMP, NSTEPS (comma sep), OUTDIR.
"""
import os
import numpy as np
import torch
import Pk_library as PKL
import bind

RUN    = os.environ.get("RUN", "/mnt/home/mlee1/ceph/fm_runs/fm_two_head")
SIMS   = os.environ.get("SIMS", "CV/sim_0,CV/sim_1,CV/sim_2").split(",")
ROOT   = os.environ.get("ROOT", "/mnt/home/mlee1/ceph/fm_testsuite")
NMAX   = int(os.environ.get("NMAX", "24"))     # most-massive halos per sim
NSAMP  = int(os.environ.get("NSAMP", "8"))     # samples for the dispersion test
NSTEPS = [int(x) for x in os.environ.get("NSTEPS", "20,50,100,200").split(",")]
OUTDIR = os.environ.get("OUTDIR", "/mnt/home/mlee1/ceph/fm_diag")
BOXP   = 128 * 50.0 / 1024.0                   # patch box [Mpc/h] = 6.25

os.makedirs(OUTDIR, exist_ok=True)


def patch_pk(img):
    img = np.asarray(img, float); mean = img.mean()
    if mean <= 0:
        return None, None
    d = (img / mean - 1.0).astype(np.float32)
    r = PKL.Pk_plane(d, BOXP, "None", 2, verbose=False)
    return r.k, r.Pk


def ratio_curve(gen_list, truth_list, ch):
    """median over patches of P_gen(k)/P_truth(k) for channel ch (or summed if ch is None)."""
    ks, rr = None, []
    for g, t in zip(gen_list, truth_list):
        gi = g.sum(0) if ch is None else g[ch]
        ti = t.sum(0) if ch is None else t[ch]
        kg, pg = patch_pk(gi); kt, pt = patch_pk(ti)
        if pg is None or pt is None:
            continue
        ks = kg
        with np.errstate(divide="ignore", invalid="ignore"):
            rr.append(pg / pt)
    return ks, np.nanmedian(np.vstack(rr), axis=0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  RUN={RUN}", flush=True)
    model = bind.Model.from_files(RUN + "/checkpoints/last.ckpt",
                                  RUN + "/norm_stats.npz", device="auto")
    params = bind.fiducial_params()

    cutouts, truth_patches = [], []
    for sim in SIMS:
        base = f"{ROOT}/{sim}/snap_090"
        try:
            cut = np.load(base + "/mass_threshold_1p000e13/halo_cutouts.npz")
            cat = np.load(base + "/mass_threshold_1p000e13/halo_catalog.npz")
            truth = np.load(base + "/full_maps.npz")["truth_maps"].astype(float)
        except Exception as e:
            print(f"  skip {sim}: {e}", flush=True); continue
        order = np.argsort(cat["halo_masses"])[::-1][:NMAX]
        cond, ls, cen = cut["condition"], cut["large_scale"], cat["centers"]
        H = truth.shape[1]
        for i in order:
            cutouts.append({"condition": cond[i], "large_scale": ls[i]})
            px, py = int(round(cen[i, 0]/50.0*H)), int(round(cen[i, 1]/50.0*H))
            xs = (np.arange(px-64, px+64) % H); ys = (np.arange(py-64, py+64) % H)
            truth_patches.append(np.stack([truth[c][np.ix_(ys, xs)] for c in range(3)]))
    print(f"{len(cutouts)} halos from {len(SIMS)} sim(s)", flush=True)

    # (1) n_steps sweep (fixed seed => same noise, isolates ODE refinement)
    sweep = {}
    for ns in NSTEPS:
        torch.manual_seed(0)
        sweep[ns] = model.generate(cutouts, params, n_steps=ns,
                                    batch_size=len(cutouts), use_amp=False, progress=False)
        print(f"  generated n_steps={ns}", flush=True)

    # (2) dispersion @ n_steps=50: many independent samples
    base_ns = 50 if 50 in NSTEPS else NSTEPS[len(NSTEPS)//2]
    samples = []
    for s in range(NSAMP):
        torch.manual_seed(100 + s)
        samples.append(model.generate(cutouts, params, n_steps=base_ns,
                                       batch_size=len(cutouts), use_amp=False, progress=False))
    print(f"  generated {NSAMP} samples @ n_steps={base_ns}", flush=True)

    # ---- analysis ----
    def himean(k, r, lo, hi=200): m = (k > lo) & (k < hi); return float(np.nanmean(r[m]))
    out = {}
    print("\n=== (1) n_steps sweep: per-patch P_gen/P_truth ===")
    print(f"{'config':>12s} {'total k>20':>11s} {'total k>40':>11s} {'stars k>20':>11s}")
    for ns in NSTEPS:
        kt, rt = ratio_curve([g for g in sweep[ns]], truth_patches, None)
        ks, rs = ratio_curve([g for g in sweep[ns]], truth_patches, 2)
        out[f"tot_ns{ns}"] = rt; out[f"star_ns{ns}"] = rs; out["k"] = kt
        print(f"{'ns='+str(ns):>12s} {himean(kt,rt,20):11.2f} {himean(kt,rt,40):11.2f} {himean(ks,rs,20):11.2f}")

    # dispersion: single sample vs sample-mean
    kt, r_single = ratio_curve([s for s in samples[0]], truth_patches, None)
    smean = [np.mean([samples[j][i] for j in range(NSAMP)], axis=0) for i in range(len(cutouts))]
    _, r_mean = ratio_curve(smean, truth_patches, None)
    out["tot_single"] = r_single; out["tot_samplemean"] = r_mean
    print("\n=== (2) dispersion @ n_steps={}: single sample vs mean of {} ===".format(base_ns, NSAMP))
    print(f"  single sample  total/truth   k>20={himean(kt,r_single,20):.2f}  k>40={himean(kt,r_single,40):.2f}")
    print(f"  mean of {NSAMP:<2d}     total/truth   k>20={himean(kt,r_mean,20):.2f}  k>40={himean(kt,r_mean,40):.2f}")

    # ---- verdict ----
    hi_lo, hi_hi = himean(out["k"], out[f"tot_ns{NSTEPS[0]}"], 40), himean(out["k"], out[f"tot_ns{NSTEPS[-1]}"], 40)
    converged = abs(hi_hi - hi_lo) < 0.05
    single_hi = himean(kt, r_single, 40)
    print("\n=== VERDICT ===")
    if not converged and hi_hi > hi_lo + 0.05:
        print(f"  high-k power RISES with n_steps ({hi_lo:.2f}->{hi_hi:.2f}): under-integrated sampler -> use more steps.")
    else:
        print(f"  high-k power FLAT across n_steps ({hi_lo:.2f}->{hi_hi:.2f}): not a step-count issue.")
    if single_hi < 0.85:
        print(f"  even a SINGLE sample is power-deficient at k>40 ({single_hi:.2f}<1): the model is mean-seeking")
        print("  / under-dispersed -> small-scale power is baked into training, not the sampler.")
        print("  => fix is training-side (spectral/adversarial loss) or post-hoc sharpening, NOT more steps.")
    else:
        print(f"  single samples carry ~truth power at k>40 ({single_hi:.2f}): sampler/averaging issue, recoverable.")

    np.savez(OUTDIR + "/sampler_diag.npz", **{k: np.asarray(v) for k, v in out.items()},
             nsteps=np.array(NSTEPS), n_halos=len(cutouts))

    # ---- figure ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = out["k"]; knyq = np.pi*1024/50.0; m = k < knyq
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a in ax:
        a.axhline(1, color="gray", lw=0.6, ls="--"); a.set_xscale("log")
        a.set_xlabel(r"$k$ [$h/$Mpc]"); a.grid(which="both", alpha=0.25); a.set_ylim(0.3, 1.6)
    for ns in NSTEPS:
        ax[0].plot(k[m], out[f"tot_ns{ns}"][m], label=f"n_steps={ns}")
    ax[0].set_ylabel(r"$P_{\rm gen}(k)/P_{\rm truth}(k)$ (total)")
    ax[0].set_title("(1) n_steps sweep — does high-k power converge?"); ax[0].legend()
    ax[1].plot(k[m], out["tot_single"][m], color="tab:blue", label="single sample")
    ax[1].plot(k[m], out["tot_samplemean"][m], color="tab:red", label=f"mean of {NSAMP}")
    ax[1].set_ylabel(r"$P_{\rm gen}/P_{\rm truth}$ (total)")
    ax[1].set_title(f"(2) dispersion @ n_steps={base_ns}"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUTDIR + "/sampler_diag.png", dpi=120)
    print(f"\nsaved {OUTDIR}/sampler_diag.png and .npz", flush=True)


if __name__ == "__main__":
    main()
