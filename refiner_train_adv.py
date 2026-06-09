#!/usr/bin/env python
"""Adversarial BIND patch refiner: residual CNN generator + PatchGAN discriminator.

The L1+spectral refiner (refiner_train.py) raised the patch power spectrum but
did NOT move the composite S(k): power-matching is satisfiable by adding high-k
*texture* anywhere, whereas the composite high-k is built from truth-like *core
structure* (concentrated central peaks).  Power can't tell "sharp core" from
"spread texture" with the same spectrum -- that's a structure problem.

Fix: add an adversarial term.  An unconditional PatchGAN discriminator D learns
"is this a real hydro patch?", forcing the generator G to produce truth-like
cores (not just the right integrated power).  We KEEP L1 (small) + spectral so G
stays anchored to the BIND input (and G is residual: out = in + correction), so
the adversarial term sharpens rather than hallucinates.  Stability: hinge loss +
spectral-norm on D + a few warmup epochs (L1+spectral only) before adv kicks in.

Writes G to refiner.pt (same format as refiner_train.py, so refiner_eval.py works
unchanged).  Fresh dir; no training run / weights touched.

Smoke (CPU):  python refiner_train_adv.py --smoke
GPU:          sbatch run_refiner_adv.sh
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from refiner_train import Refiner, Norm, radial_index, power_per_bin, spectral_loss


# ----------------------------------------------------------------------------
# discriminator (unconditional PatchGAN, spectral-normed, hinge)
# ----------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, ch=3, w=64):
        super().__init__()
        def c(i, o, s):
            return spectral_norm(nn.Conv2d(i, o, 4, s, 1))
        self.net = nn.Sequential(
            c(ch, w, 2), nn.LeakyReLU(0.2, True),       # 128 -> 64
            c(w, 2 * w, 2), nn.LeakyReLU(0.2, True),    # 64 -> 32
            c(2 * w, 4 * w, 2), nn.LeakyReLU(0.2, True),  # 32 -> 16
            c(4 * w, 4 * w, 1), nn.LeakyReLU(0.2, True),  # 16 -> 15
            c(4 * w, 1, 1),                              # -> 14x14 score map
        )

    def forward(self, x, feats=False):
        if not feats:
            return self.net(x)
        f = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                f.append(x)
        return x, f


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/mnt/home/mlee1/ceph/refiner_two_head")
    ap.add_argument("--out_dir", default="/mnt/home/mlee1/ceph/refiner_two_head_adv")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_l1", type=float, default=0.3)
    ap.add_argument("--lambda_spec", type=float, default=2.0)
    ap.add_argument("--lambda_fm", type=float, default=5.0,
                    help="feature-matching weight: match frozen-D features of refined vs truth "
                         "(stable structural loss; replaces the unstable adversarial min-max)")
    ap.add_argument("--lambda_adv", type=float, default=0.0,
                    help="kept for experiments; 0 = pure feature-matching (recommended, stable)")
    ap.add_argument("--adv_warmup", type=int, default=5,
                    help="epochs training D adversarially; after this D is FROZEN and used for FM")
    ap.add_argument("--disc_channels", default="1,2",
                    help="channels D judges; default Gas,Stars (the deficit channels). DM (T~1) is "
                         "already matched, so judging it gives D no signal and it collapses to a constant.")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--d_width", type=int, default=64)
    ap.add_argument("--nbins", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Path(args.data_dir)
    tr = np.load(d / "train.npz"); va = np.load(d / "val.npz"); nm = np.load(d / "norm.npz")
    Xtr, Ytr = tr["bind"], tr["truth"]; Xva, Yva = va["bind"], va["truth"]
    if args.smoke:
        Xtr, Ytr, Xva, Yva = Xtr[:64], Ytr[:64], Xva[:64], Yva[:64]
        args.epochs, args.adv_warmup = 3, 1
    print(f"[adv] device={device} train={Xtr.shape} val={Xva.shape} smoke={args.smoke}")

    norm = Norm(nm["mean"], nm["std"], device)
    idx, kcent = radial_index(128, args.nbins, device)
    kweight = (kcent / kcent.max()).clamp(min=0.2)
    hi = (kcent > 0.55 * kcent.max())

    dch = [int(c) for c in args.disc_channels.split(",")]  # channels D judges
    G = Refiner(w=args.width).to(device)
    D = Discriminator(ch=len(dch), w=args.d_width).to(device)
    print(f"[adv] D judges channels {dch} ({[['DM','Gas','Stars'][c] for c in dch]})")
    optG = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5)
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(f"[adv] G={sum(p.numel() for p in G.parameters())/1e3:.0f}k "
          f"D={sum(p.numel() for p in D.parameters())/1e3:.0f}k  "
          f"l1={args.lambda_l1} spec={args.lambda_spec} adv={args.lambda_adv} warmup={args.adv_warmup}")

    def batches(X, Y, shuffle):
        n = len(X); order = np.random.permutation(n) if shuffle else np.arange(n)
        for i in range(0, n, args.batch_size):
            j = order[i:i + args.batch_size]
            yield torch.from_numpy(X[j]).to(device), torch.from_numpy(Y[j]).to(device)

    def stack_power(field):
        P = power_per_bin(field, idx, args.nbins)
        Pt = power_per_bin(field.sum(1, keepdim=True), idx, args.nbins)
        return torch.cat([P, Pt], 1).sum(0)

    @torch.no_grad()
    def val_T():
        G.eval(); Pr = Pb = Pt = 0.0
        for xb, yb in batches(Xva, Yva, False):
            pr = norm.inv(G(norm.fwd(xb)))
            Pr = Pr + stack_power(pr); Pb = Pb + stack_power(xb); Pt = Pt + stack_power(yb)
        Tr = torch.sqrt(Pr / Pt.clamp(min=1e-30)); Tb = torch.sqrt(Pb / Pt.clamp(min=1e-30))
        return (Tr[3][hi].mean().item(), Tr[2][hi].mean().item(),
                Tb[3][hi].mean().item(), Tb[2][hi].mean().item())

    def fm_loss(fake_ch, real_ch):
        """Feature matching: L1 between frozen-D features of refined vs truth."""
        _, ff = D(fake_ch, feats=True)
        with torch.no_grad():
            _, fr = D(real_ch, feats=True)
        return sum(F.l1_loss(a, b) for a, b in zip(ff, fr)) / len(ff)

    hist = []; frozen = False
    for ep in range(args.epochs):
        t0 = time.time()
        fm_phase = ep >= args.adv_warmup
        if fm_phase and not frozen:           # freeze D once warmup ends -> stable FM, no min-max
            for p in D.parameters():
                p.requires_grad_(False)
            D.eval(); frozen = True
            print(f"  [froze D at ep{ep}; switching G to feature-matching]")
        G.train()
        gl = dl = dreal = dfake = fmv = 0.0; nb = 0
        adv_w = args.lambda_adv if fm_phase else 0.0
        for xb, yb in batches(Xtr, Ytr, True):
            xn, yn = norm.fwd(xb), norm.fwd(yb)
            # --- D step: only during warmup (learn to tell sharp from diffuse cores) ---
            if not frozen:
                with torch.no_grad():
                    pr_n = G(xn)
                d_real = D(yn[:, dch]); d_fake = D(pr_n[:, dch])
                lossD = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
                optD.zero_grad(); lossD.backward(); optD.step()
                dl += lossD.item(); dreal += d_real.mean().item(); dfake += d_fake.mean().item()
            # --- G step ---
            pr_n = G(xn); pr_phys = norm.inv(pr_n)
            lossG = (args.lambda_l1 * F.l1_loss(pr_n, yn)
                     + args.lambda_spec * spectral_loss(pr_phys, yb, idx, args.nbins, kweight))
            if frozen:
                fm = fm_loss(pr_n[:, dch], yn[:, dch].detach())
                lossG = lossG + args.lambda_fm * fm
                if adv_w > 0:
                    lossG = lossG + adv_w * (-D(pr_n[:, dch]).mean())
                fmv += fm.item()
            optG.zero_grad(); lossG.backward(); optG.step()
            gl += lossG.item(); nb += 1
        Tt_r, Ts_r, Tt_b, Ts_b = val_T()
        hist.append((ep, gl / nb, dl / nb, Tt_r, Ts_r))
        if frozen:
            extra = f"FM={fmv/nb:.4f}"
        else:
            extra = f"D={dl/nb:.3f} D(real)={dreal/nb:+.2f} D(fake)={dfake/nb:+.2f}"
        tag = "fm  " if frozen else "warm"
        print(f"  ep{ep:03d}[{tag}] G={gl/nb:.3f} {extra} | "
              f"hi-k T_total {Tt_b:.3f}->{Tt_r:.3f}  T_stars {Ts_b:.3f}->{Ts_r:.3f} {time.time()-t0:.1f}s")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        torch.save({"model": G.state_dict(), "width": args.width,
                    "mean": nm["mean"], "std": nm["std"]}, out / "refiner.pt")
        np.savez(out / "history.npz", hist=np.array(hist))
        print(f"[adv] wrote {out}/refiner.pt")
    else:
        print("[adv] smoke OK (no checkpoint written)")


if __name__ == "__main__":
    main()
