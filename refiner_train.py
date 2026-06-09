#!/usr/bin/env python
"""Train the BIND patch refiner: small residual CNN, L1 + spectral loss.

Maps a BIND patch -> truth patch (3 channels) to restore the high-k stellar/gas
core concentration the flow under-delivers.  The spectral-loss term is essential:
a pure L1/L2 refiner would regress-to-the-mean and re-blur (the SAME failure as
the flow), so we add a term matching each channel's physical power spectrum,
weighted toward high k, which forces the CNN to spatially re-concentrate cores.

The refiner takes NO theta input (pure image->image), so improvement on the
theta-disjoint val set (refiner_data.py) is an unambiguous generalization test.
Writes refiner.pt + history to --out_dir; touches no training run / weights.

Smoke (CPU):  python refiner_train.py --smoke
GPU:          sbatch run_refiner.sh
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, w, dilation):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(w, w, 3, padding=1)
        self.n1 = nn.GroupNorm(8, w); self.n2 = nn.GroupNorm(8, w)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class Refiner(nn.Module):
    """Residual image->image refiner in normalized log-space (out = in + net)."""
    def __init__(self, ch=3, w=64, dilations=(1, 2, 4, 2, 1)):
        super().__init__()
        self.head = nn.Conv2d(ch, w, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(w, d) for d in dilations])
        self.tail = nn.Conv2d(w, ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)  # start = identity

    def forward(self, x):
        return x + self.tail(self.body(self.head(x)))


# ----------------------------------------------------------------------------
# normalization (physical <-> normalized log10(1+x)) and spectral loss
# ----------------------------------------------------------------------------
class Norm:
    def __init__(self, mean, std, device):
        self.m = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.s = torch.tensor(std, device=device).view(1, -1, 1, 1)

    def fwd(self, phys):
        return (torch.log10(1.0 + phys.clamp(min=0)) - self.m) / self.s

    def inv(self, norm):
        return torch.clamp(10.0 ** (norm * self.s + self.m) - 1.0, min=0.0)


def radial_index(npix, nbins, device):
    kf = torch.fft.fftfreq(npix) * npix
    kk = torch.sqrt(kf[:, None] ** 2 + kf[None, :] ** 2)
    kk = kk[:, : npix // 2 + 1]                      # rfft2 layout
    edges = torch.linspace(1.0, kk.max().item(), nbins + 1)
    idx = torch.clamp(torch.bucketize(kk.contiguous(), edges) - 1, 0, nbins - 1).to(device)
    kcent = (0.5 * (edges[:-1] + edges[1:])).to(device)
    return idx, kcent


def power_per_bin(field, idx, nbins):
    """Per-channel azimuthally-averaged power of the per-patch overdensity."""
    B, C, H, W = field.shape
    mu = field.mean((-2, -1), keepdim=True).clamp(min=1e-12)
    delta = field / mu - 1.0
    p2 = (torch.fft.rfft2(delta).abs() ** 2)         # (B,C,H,W//2+1)
    flat = p2.reshape(B, C, -1)
    out = torch.zeros(B, C, nbins, device=field.device)
    out.scatter_add_(2, idx.reshape(1, 1, -1).expand(B, C, -1), flat)
    counts = torch.zeros(nbins, device=field.device)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones_like(idx.reshape(-1), dtype=torch.float))
    return out / counts.clamp(min=1).view(1, 1, -1)


def spectral_loss(pred_phys, truth_phys, idx, nbins, kweight):
    pp = power_per_bin(pred_phys, idx, nbins).clamp(min=1e-20)
    pt = power_per_bin(truth_phys, idx, nbins).clamp(min=1e-20)
    d = (torch.log10(pp) - torch.log10(pt)) ** 2          # (B,C,nbins)
    return (d * kweight.view(1, 1, -1)).mean()


# ----------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/mnt/home/mlee1/ceph/refiner_two_head")
    ap.add_argument("--out_dir", default="/mnt/home/mlee1/ceph/refiner_two_head")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_spec", type=float, default=5.0)
    ap.add_argument("--lambda_l1", type=float, default=0.3,
                    help="weight on the L1 anchor; keep small so spectral drives sharpening")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--nbins", type=int, default=24)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to verify the loop")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Path(args.data_dir)
    tr = np.load(d / "train.npz"); va = np.load(d / "val.npz"); nm = np.load(d / "norm.npz")
    Xtr, Ytr = tr["bind"], tr["truth"]; Xva, Yva = va["bind"], va["truth"]
    if args.smoke:
        Xtr, Ytr, Xva, Yva = Xtr[:64], Ytr[:64], Xva[:64], Yva[:64]
        args.epochs = 2
    print(f"[refiner] device={device} train={Xtr.shape} val={Xva.shape} smoke={args.smoke}")

    norm = Norm(nm["mean"], nm["std"], device)
    idx, kcent = radial_index(128, args.nbins, device)
    kweight = (kcent / kcent.max()).clamp(min=0.2)            # emphasize high k
    model = Refiner(w=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[refiner] params={n_par/1e3:.0f}k  lambda_spec={args.lambda_spec}")

    def batches(X, Y, shuffle):
        n = len(X); order = np.random.permutation(n) if shuffle else np.arange(n)
        for i in range(0, n, args.batch_size):
            j = order[i:i + args.batch_size]
            yield (torch.from_numpy(X[j]).to(device), torch.from_numpy(Y[j]).to(device))

    hi = (kcent > 0.55 * kcent.max())  # high-k band (the deficit region)

    def _stack_power(field):
        P = power_per_bin(field, idx, args.nbins)                      # (B,3,nbins)
        Pt = power_per_bin(field.sum(1, keepdim=True), idx, args.nbins)  # (B,1,nbins)
        return torch.cat([P, Pt], 1).sum(0)                            # (4,nbins)

    def val_metrics():
        """val_spec (refined & BIND baseline) + high-k T(k) for Total/Stars."""
        model.eval()
        sp = bsp = 0.0; nb = 0
        Pr = Pb = Pt = 0.0
        with torch.no_grad():
            for xb, yb in batches(Xva, Yva, False):
                pr = norm.inv(model(norm.fwd(xb)))
                sp += spectral_loss(pr, yb, idx, args.nbins, kweight).item()
                bsp += spectral_loss(xb, yb, idx, args.nbins, kweight).item()
                Pr = Pr + _stack_power(pr); Pb = Pb + _stack_power(xb); Pt = Pt + _stack_power(yb)
                nb += 1
        Tr = torch.sqrt(Pr / Pt.clamp(min=1e-30)); Tb = torch.sqrt(Pb / Pt.clamp(min=1e-30))
        # [Total, Stars] high-k T: refined then BIND baseline
        return (sp / nb, bsp / nb,
                Tr[3][hi].mean().item(), Tr[2][hi].mean().item(),
                Tb[3][hi].mean().item(), Tb[2][hi].mean().item())

    hist = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for xb, yb in batches(Xtr, Ytr, True):
            xn, yn = norm.fwd(xb), norm.fwd(yb)
            pr_n = model(xn); pr_phys = norm.inv(pr_n)
            loss = args.lambda_l1 * F.l1_loss(pr_n, yn) + args.lambda_spec * spectral_loss(
                pr_phys, yb, idx, args.nbins, kweight)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        vsp, bsp, Tt_r, Ts_r, Tt_b, Ts_b = val_metrics()
        hist.append((ep, tot / nb, vsp, bsp, Tt_r, Ts_r))
        print(f"  ep{ep:03d} train={tot/nb:.4f} val_spec={vsp:.4f}(base {bsp:.4f}) | "
              f"hi-k T_total {Tt_b:.3f}->{Tt_r:.3f}  T_stars {Ts_b:.3f}->{Ts_r:.3f} "
              f"{time.time()-t0:.1f}s")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        torch.save({"model": model.state_dict(), "width": args.width,
                    "mean": nm["mean"], "std": nm["std"]}, out / "refiner.pt")
        np.savez(out / "history.npz", hist=np.array(hist))
        print(f"[refiner] wrote {out}/refiner.pt")
    else:
        print("[refiner] smoke OK (no checkpoint written)")


if __name__ == "__main__":
    main()
