"""scatter/fill_outline_numbers.py
Read the completed J_mean_and_scatter.npz and update PAPER_OUTLINE.md with
top-mover parameter names, cosine angle between mean/scatter vectors, and SNR.

Run after scatter_jacobian.py completes:
    python scatter/fill_outline_numbers.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scatter.scatter_jacobian import PARAM_NAMES
from scatter.measure_scatter import ALL_OBS_NAMES

JAC_PATH    = ROOT / "scatter" / "J_mean_and_scatter.npz"
OUTLINE_PATH = ROOT / "scatter" / "PAPER_OUTLINE.md"


def top_n(arr: np.ndarray, param_names: list[str], n: int = 5) -> str:
    idxs = np.argsort(np.abs(arr))[::-1][:n]
    parts = [f"{param_names[j]}={arr[j]:+.3f}" for j in idxs if np.isfinite(arr[j])]
    return ", ".join(parts)


def cosine_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine of angle between two vectors (ignoring NaNs)."""
    mask = np.isfinite(v1) & np.isfinite(v2)
    if mask.sum() < 2:
        return float("nan")
    a = v1[mask]; b = v2[mask]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    if not JAC_PATH.exists():
        print(f"ERROR: {JAC_PATH} not found — run scatter_jacobian.py first")
        sys.exit(1)

    d = np.load(JAC_PATH, allow_pickle=True)
    J_mean      = d["J_mean"]         # (N_obs, N_params)
    J_log_sigma = d["J_log_sigma"]    # (N_obs, N_params)
    J_mean_se   = d.get("J_mean_se",   np.zeros_like(J_mean))
    J_lsig_se   = d.get("J_log_sigma_se", np.zeros_like(J_log_sigma))
    obs_names   = list(d["obs_names"])
    n_obs, n_params = J_mean.shape

    # --- Build replacement text ------------------------------------------------
    lines_42 = []
    lines_43 = []
    lines_sanity = []

    for o_name in ["M_gas", "M_star", "dq_DM"]:
        if o_name not in obs_names:
            continue
        o = obs_names.index(o_name)
        lines_42.append(f"For `{o_name}`:")
        lines_42.append(f"  Top-5 |J_mean|: {top_n(J_mean[o], PARAM_NAMES)}")
        lines_43.append(f"For `{o_name}`:")
        lines_43.append(f"  Top-5 |J_log_sigma|: {top_n(J_log_sigma[o], PARAM_NAMES)}")

    # Sanity: Omega_m in top-3 for M_gas mean
    if "M_gas" in obs_names:
        o = obs_names.index("M_gas")
        top3 = set(np.argsort(np.abs(J_mean[o]))[::-1][:3])
        om_pass = 0 in top3
        lines_sanity.append(
            f"Omega_m in top-3 M_gas mean movers: {'YES ✓' if om_pass else 'NO — check Jacobian'}"
        )

    # SNR for AGN/SN params
    snr_lines = []
    for j in [2, 3, 4, 5]:
        for o_name in ["M_gas", "M_star"]:
            if o_name not in obs_names:
                continue
            o = obs_names.index(o_name)
            sig = abs(J_log_sigma[o, j])
            se  = J_lsig_se[o, j]
            snr = sig / se if se > 0 and np.isfinite(se) else float("nan")
            snr_lines.append(
                f"  {o_name} p{j}({PARAM_NAMES[j]}): |J_lsig|={sig:.3f} SE={se:.3f} "
                f"SNR={snr:.1f} {'✓' if np.isfinite(snr) and snr > 3.3 else '⚠ LOW'}"
            )

    # Cosine angles between mean and scatter vectors per obs
    angle_lines = []
    for o_name in ["M_gas", "M_star", "dq_DM"]:
        if o_name not in obs_names:
            continue
        o = obs_names.index(o_name)
        cos_theta = cosine_angle(J_mean[o], J_log_sigma[o])
        angle_deg = np.degrees(np.arccos(np.clip(abs(cos_theta), 0, 1)))
        angle_lines.append(
            f"  {o_name}: cos θ = {cos_theta:+.3f}  |θ| = {angle_deg:.1f}°"
        )

    # --- Print summary --------------------------------------------------------
    print("\n" + "="*60)
    print("=== Jacobian Summary for PAPER_OUTLINE ===")
    print("="*60)
    print("\n[Section 4.2 — Top mean-movers]")
    for l in lines_42: print(l)
    print("\n[Section 4.3 — Top scatter-movers]")
    for l in lines_43: print(l)
    print("\n[Sanity checks]")
    for l in lines_sanity: print(l)
    print("\n[SNR checks for AGN/SN params]")
    for l in snr_lines: print(l)
    print("\n[Section 5.2 — Mean/scatter angle]")
    for l in angle_lines: print(l)

    # --- Patch PAPER_OUTLINE.md ------------------------------------------------
    outline = OUTLINE_PATH.read_text()

    # 4.2 block
    block_42 = "\n".join(lines_42)
    outline = re.sub(
        r"\[TO FILL after Jacobian\] For \$\\log M_{\\rm gas}\$.*?passed: \[YES/NO\]",
        block_42 + "\n\nSanity (" + lines_sanity[0] + ")",
        outline,
        flags=re.DOTALL,
    )

    # 4.3 block
    block_43 = "\n".join(lines_43)
    outline = re.sub(
        r"For \$\\sigma_{\\rm inter}\(\\log M_{\\rm gas}\)\$:.*?For \$\\sigma_{\\rm inter}\(\\log M_\\star\)\$:.*?\[TO FILL\]",
        block_43,
        outline,
        flags=re.DOTALL,
    )

    # 5.2 cosine
    cos_text = "\n".join(angle_lines)
    outline = re.sub(
        r"Quantify: the angle between the mean-response and scatter-response vectors in 35-D: cos θ = \[TO FILL\]\.",
        "Quantify: the angle between the mean-response and scatter-response vectors in 35-D:\n" + cos_text,
        outline,
    )

    # Numbers reference
    jm_gas = top_n(J_mean[obs_names.index("M_gas")] if "M_gas" in obs_names else np.array([]), PARAM_NAMES)
    jm_star = top_n(J_mean[obs_names.index("M_star")] if "M_star" in obs_names else np.array([]), PARAM_NAMES)
    js_gas = top_n(J_log_sigma[obs_names.index("M_gas")] if "M_gas" in obs_names else np.array([]), PARAM_NAMES)
    outline = re.sub(
        r"- Top-5 J_mean param names and values for M_gas, M_star, dq_DM: \[TO FILL after Jacobian\]",
        f"- Top-5 J_mean param names: M_gas: {jm_gas}; M_star: {jm_star}",
        outline,
    )
    outline = re.sub(
        r"- Top-5 J_log_sigma param names and values: \[TO FILL after Jacobian\]",
        f"- Top-5 J_log_sigma: M_gas: {js_gas}",
        outline,
    )

    cos_str = "; ".join(angle_lines).strip()
    outline = re.sub(
        r"- Angle between mean and scatter direction in 35-D: \[TO FILL after Jacobian\]",
        f"- Angle between mean and scatter directions: {cos_str}",
        outline,
    )

    OUTLINE_PATH.write_text(outline)
    print(f"\n[fill_outline] Updated {OUTLINE_PATH}")


if __name__ == "__main__":
    main()
