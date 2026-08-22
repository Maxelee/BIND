"""Build examples/lowmass_extrapolation.ipynb.

The notebook characterises whether BIND (trained on M200c > 1e13 halos)
generalises down to 1e12 halos that live inside the 6.25 Mpc/h training
patches, at z=0, using the redshift-free fm_thermo model. It loads the
per-halo artifacts written by ``bind.cli.camels_suite`` (run via
``run_lowmass_suite.sh`` at ``--halo_mass_min 1e12``) and is race-safe /
partial-data tolerant, so it renders on whatever sims have completed.

Run:  python examples/_build_lowmass_nb.py
"""
from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

nb = new_notebook()
C: list = []


def md(src: str):
    C.append(new_markdown_cell(src.strip("\n")))


def code(src: str):
    C.append(new_code_cell(src.strip("\n")))


md(r"""
# Can BIND paint low-mass halos? (zero-shot extrapolation to $10^{12}\,M_\odot/h$)

BIND was trained on patches **centered on** $M_{200c} > 10^{13}\,M_\odot/h$ halos.
But each 6.25 Mpc/h training patch already *contains* lower-mass neighbours as
context — so the model has *seen* low-mass structure, it has just never been
asked to **center its prediction** on a sub-$10^{13}$ halo.

This notebook is the **zero-shot test**: take DMO patches centered on
$10^{12}$–$10^{14}\,M_\odot/h$ halos from the **CV, 1P and held-out Test (SB35)**
suites at **z=0**, run the redshift-free **`fm_thermo`** model, and ask whether
agreement with the CAMELS hydro truth **degrades toward the low-mass end** —
i.e. where (if anywhere) the extrapolation breaks.

**Caveats baked into the interpretation**
- Pixel size $= 6250/128 \approx 48.8$ kpc/h. A $10^{12}$ halo has $R_{200c}\approx 180$ kpc/h ($\approx 3.7$ px); a $10^{11}$ halo $\approx 1.7$ px; a $10^{10}$ halo $\approx 0.8$ px (sub-pixel). The $10^{12}$ floor is the smallest *resolved* scale here.
- 1e12–1e13 halos appear as neighbours inside training patches, so this is "near-extrapolation", not blind.
- CAMELS L50n512 truth is itself shot-noise-limited at the lowest masses — we validate BIND **vs CAMELS truth**, not vs ground truth.
- Per-halo truth is saved for the **thermo** channels; mass channels (DM/Gas/Stars) are checked via per-halo mass conservation + the full-box composite.
""")

code(r"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import BadZipFile

plt.rcParams.update({"figure.dpi": 110, "font.size": 11})

# --- config -----------------------------------------------------------------
OUTPUT_ROOT = Path("/mnt/home/mlee1/ceph/fm_lowmass")   # full-run output
if not any(OUTPUT_ROOT.glob("*/sim_*")):                # fall back to the dry-run
    OUTPUT_ROOT = Path("/mnt/home/mlee1/ceph/fm_lowmass_test")
MODEL_NAME  = "fm_thermo_ema"
SNAP        = 90                 # z = 0
TRAIN_CUT   = 1e13               # BIND training mass floor
PIX_KPCH    = 50000 / 1024       # pixel size in kpc/h (full-box grid)
THERMO_KEYS = ["compton_y", "T", "entropy", "P_e"]
MASS_CH     = ["DM_hydro", "Gas", "Stars"]
print("Reading from:", OUTPUT_ROOT)
""")

md("## Load per-halo artifacts (race-safe; renders on partial data)")

code(r"""
def _safe_load(p):
    try:
        return np.load(p, allow_pickle=True)
    except (FileNotFoundError, BadZipFile, OSError, EOFError):
        return None

def discover():
    # Find every (suite, sim, mass_threshold) leaf that has a generated set.
    leaves = []
    for gen in OUTPUT_ROOT.glob(f"*/sim_*/snap_{SNAP:03d}/mass_threshold_*/{MODEL_NAME}/generated_halos.npz"):
        mt = gen.parent.parent
        leaves.append({
            "suite": gen.parents[4].name,
            "sim":   gen.parents[3].name,
            "gen":   gen,
            "cat":   mt / "halo_catalog.npz",
            "tth":   mt / "truth_thermo_patches.npz",
            "cut":   mt / "halo_cutouts.npz",
        })
    return sorted(leaves, key=lambda d: (d["suite"], d["sim"]))

leaves = discover()
print(f"Found {len(leaves)} completed (suite, sim) sets:",
      sorted({(l['suite'], l['sim']) for l in leaves})[:8], "...")

def dex_bias(g, t):
    m = (t > 0) & (g > 0)
    return np.median(np.log10(g[m]) - np.log10(t[m])) if m.sum() >= 10 else np.nan

rows = []
for L in leaves:
    cat = _safe_load(L["cat"]); gen = _safe_load(L["gen"])
    if cat is None or gen is None:
        continue
    g = gen["generated"]                          # (N, 7, 128, 128)
    m = cat["halo_masses"]; r = cat["halo_r200s"]  # Msun/h, Mpc/h
    tth = _safe_load(L["tth"])
    cut = _safe_load(L["cut"])
    truth = tth["truth_thermo"] if tth is not None else None     # (N,4,128,128)
    cond  = cut["condition"]    if cut is not None else None      # (N,128,128) DMO
    n = min(len(m), len(g))
    for i in range(n):
        row = {"suite": L["suite"], "sim": L["sim"], "M200c": float(m[i]),
               "r200_px": float(r[i]) * 1000 / PIX_KPCH}
        # per-halo mass conservation: generated (DM+Gas+Stars) vs DMO condition
        if cond is not None:
            cmass = float(cond[i].sum())
            row["mass_ratio"] = float(g[i, :3].sum() / cmass) if cmass > 0 else np.nan
        # per-halo thermo dex bias
        if truth is not None:
            for c, key in enumerate(THERMO_KEYS):
                row[f"bias_{key}"] = dex_bias(g[i, 3 + c], truth[i, c])
        rows.append(row)

df = pd.DataFrame(rows)
df["logM"] = np.log10(df["M200c"])
print(f"{len(df)} halos loaded across {df['suite'].nunique()} suites")
df.groupby("suite")["M200c"].describe()[["count", "min", "max"]]
""")

md("""
## 1. Coverage — how many more halos become accessible at the 1e12 floor

The steep halo mass function means dropping the floor from $10^{13}$ to $10^{12}$
unlocks roughly an order of magnitude more halos per simulation. This is the
prize: a far denser, lower-mass sample for downstream science.
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
edges = np.arange(11.8, 14.6, 0.2)
for s, g in df.groupby("suite"):
    ax[0].hist(g["logM"], bins=edges, histtype="step", lw=2, label=f"{s} (n={len(g)})")
ax[0].axvline(np.log10(TRAIN_CUT), color="k", ls="--", label="training cut $10^{13}$")
ax[0].set(xlabel=r"$\log_{10} M_{200c}\ [M_\odot/h]$", ylabel="N halos", title="Mass distribution of evaluated halos")
ax[0].legend(fontsize=8)

below = (df["M200c"] < TRAIN_CUT).sum(); above = (df["M200c"] >= TRAIN_CUT).sum()
ax[1].bar(["<1e13\n(new)", ">=1e13\n(train regime)"], [below, above],
          color=["#c0392b", "#2c3e50"])
ax[1].set(ylabel="N halos", title=f"{below}/{below+above} evaluated halos are NEW (below training cut)")
for i, v in enumerate([below, above]):
    ax[1].text(i, v, str(v), ha="center", va="bottom")
plt.tight_layout(); plt.show()
print(f"Fold increase in accessible halos at 1e12 vs 1e13: {(below+above)/max(above,1):.1f}x")
""")

md("""
## 2. Extrapolation quality — thermo bias & scatter vs halo mass

The key plot. For each gas-thermodynamic channel we show the median log10 bias
(generated − truth) and its halo-to-halo scatter, **binned by $M_{200c}$**. If
BIND extrapolates, the curves stay flat and near zero as we cross below the
$10^{13}$ training cut. A blow-up at low mass would localize the floor.
""")

code(r"""
mbins = np.array([1e12, 3.16e12, 1e13, 3.16e13, 1e15])
centers = np.sqrt(mbins[:-1] * mbins[1:])
df["mbin"] = np.digitize(df["M200c"], mbins) - 1

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for ax, key in zip(axes.ravel(), THERMO_KEYS):
    col = f"bias_{key}"
    if col not in df:
        ax.set_title(f"{key} (no truth)"); continue
    for s, gs in df.groupby("suite"):
        med, lo, hi, xs = [], [], [], []
        for b in range(len(centers)):
            v = gs.loc[gs["mbin"] == b, col].dropna()
            if len(v) < 3: continue
            xs.append(centers[b]); med.append(v.median())
            lo.append(v.quantile(0.16)); hi.append(v.quantile(0.84))
        if xs:
            ax.plot(xs, med, "o-", label=s)
            ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(TRAIN_CUT, color="k", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_title(key)
    ax.set_ylabel("log10(gen) - log10(truth)")
axes[0, 0].legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel(r"$M_{200c}\ [M_\odot/h]$")
fig.suptitle("Thermo bias vs mass (band = 16-84%); dashed = training cut", y=1.01)
plt.tight_layout(); plt.show()
""")

md("""
## 3. Mass conservation vs halo mass

The mass channels carry no per-halo truth patch, but BIND's per-halo
mass-matching should hold across the mass range. We compare the generated
$(\\mathrm{DM}+\\mathrm{Gas}+\\mathrm{Stars})$ patch mass to the DMO condition
patch mass — a ratio that should sit near 1 with no mass-dependent drift.
""")

code(r"""
if "mass_ratio" in df:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, gs in df.groupby("suite"):
        ax.scatter(gs["M200c"], gs["mass_ratio"], s=6, alpha=0.3, label=s)
    # binned median
    med = df.groupby("mbin")["mass_ratio"].median()
    ax.plot(centers[med.index], med.values, "k-o", lw=2, label="median")
    ax.axhline(1.0, color="r", ls="--"); ax.axvline(TRAIN_CUT, color="k", ls=":")
    ax.set(xscale="log", xlabel=r"$M_{200c}\ [M_\odot/h]$",
           ylabel="generated / DMO patch mass", ylim=(0.8, 1.2),
           title="Per-halo mass conservation across mass")
    ax.legend(fontsize=8); plt.tight_layout(); plt.show()
else:
    print("halo_cutouts (condition) not found — skip mass-conservation panel")
""")

md("""
## 4. A low-mass halo, painted

Pick a halo near the $10^{12}$ floor and show the DMO condition alongside every
generated channel (and the thermo truth) — visual evidence the model produces a
structured, non-degenerate prediction well below its training regime.
""")

code(r"""
# choose a completed leaf with cutouts + truth, then a ~1e12 halo within it
pick = None
for L in discover():
    cat = _safe_load(L["cat"]); cut = _safe_load(L["cut"])
    gen = _safe_load(L["gen"]); tth = _safe_load(L["tth"])
    if None in (cat, cut, gen, tth):
        continue
    m = cat["halo_masses"]
    j = int(np.argmin(np.abs(np.log10(m) - 12.1)))   # closest to 1.3e12
    pick = (L, j, m[j]); break

if pick is None:
    print("No leaf with full cutouts+truth yet.")
else:
    L, j, mj = pick
    cond = _safe_load(L["cut"])["condition"][j]
    g    = _safe_load(L["gen"])["generated"][j]
    tth  = _safe_load(L["tth"])["truth_thermo"][j]
    panels = [("DMO cond", cond)] + [(f"gen {MASS_CH[c]}", g[c]) for c in range(3)]
    panels += [(f"gen {THERMO_KEYS[c]}", g[3 + c]) for c in range(4)]
    panels += [(f"truth {THERMO_KEYS[c]}", tth[c]) for c in range(4)]
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for ax, (title, img) in zip(axes.ravel(), panels):
        a = np.log10(np.clip(img, img[img > 0].min() if (img > 0).any() else 1e-30, None))
        ax.imshow(a, cmap="magma"); ax.set_title(title, fontsize=9); ax.axis("off")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{L['suite']} {L['sim']}  halo M200c={mj:.2e} (log10(field) shown)", y=1.0)
    plt.tight_layout(); plt.show()
""")

md("""
## 5. Summary table

Per-suite, per-mass-bin counts and median thermo bias — the headline numbers for
"how far down does BIND hold up".
""")

code(r"""
agg = {f"bias_{k}": "median" for k in THERMO_KEYS if f"bias_{k}" in df}
lbls = ["1e12-3e12", "3e12-1e13", "1e13-3e13", ">3e13"]
out = (df.assign(bin=[lbls[b] if 0 <= b < len(lbls) else "oob" for b in df["mbin"]])
         .groupby(["suite", "bin"])
         .agg(n=("M200c", "size"), **{k: (k, "median") for k in agg})
         .round(3))
out
""")

nb["cells"] = C
out = "examples/lowmass_extrapolation.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(C), "cells")
