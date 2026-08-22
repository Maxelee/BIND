"""Standalone unit-check for the piece-4 (1P/two-bound OOD validation) index-
matching logic: _zi_ood's z_s-slicing branch, and the "meas_tb / idx_per_head
built in lockstep, subset via idxs" indexing pattern used in the fig15b cell.
Synthetic data only.

Run: /mnt/home/mlee1/venvs/BIND_env/bin/python audits/unit_check_ood_indexing.py
"""
import numpy as np

ZI = 1  # matches the notebook's z_s=1 working index


def _zi_ood(key, arr, zi_heads):
    arr = np.asarray(arr)
    return arr[:, ZI, :] if key in zi_heads else arr


def check_zi_slicing():
    zi_heads = {"suppression"}
    tomo = np.arange(4 * 5 * 3).reshape(4, 5, 3)          # (N_TB, n_zs, n_bin)
    flat = np.arange(4 * 3).reshape(4, 3)                  # (N_TB, n_bin), no z_s axis
    out_tomo = _zi_ood("suppression", tomo, zi_heads)
    out_flat = _zi_ood("cl_yy", flat, zi_heads)
    assert out_tomo.shape == (4, 3)
    assert np.array_equal(out_tomo, tomo[:, ZI, :])
    assert out_flat.shape == (4, 3)
    assert np.array_equal(out_flat, flat)
    print("PASS: _zi_ood slices the z_s axis only for tomographic heads, "
          "passes non-tomographic heads through unchanged.")


def check_matched_index_alignment():
    """The fig15b cell appends to meas_tb[s] and idx_per_head[s] IN LOCKSTEP
    inside a single ascending loop over idx_meas, then later does
    `pred_arr = _zi_ood(s, pred_tb[s])[idxs]` where idxs = idx_per_head[s].
    Reproduce that exact pattern on synthetic data and check the measured/
    predicted rows end up correctly paired by run id, even when some runs are
    skipped (a head not available for every run -- the realistic case)."""
    n_tb = 10
    rng = np.random.default_rng(0)
    pred_all = rng.normal(size=(n_tb, 3))          # "emulator" prediction, ALL runs
    truth_all = pred_all * (1 + rng.normal(0, 0.05, size=(n_tb, 3)))  # close to pred + noise

    # simulate: only even-indexed runs have a usable cache for this head
    available = [i for i in range(n_tb) if i % 2 == 0]
    meas, idx_per_head = [], []
    for i in available:
        meas.append(truth_all[i])          # what the cell would np.load() for run i
        idx_per_head.append(i)

    meas_arr = np.asarray(meas)
    pred_arr = pred_all[idx_per_head]      # the cell's exact indexing pattern

    # ground truth: pred_arr[k] must be pred_all[available[k]], not shuffled
    for k, i in enumerate(available):
        assert np.array_equal(pred_arr[k], pred_all[i]), \
            f"misaligned row: pred_arr[{k}] != pred_all[{i}]"
        assert np.array_equal(meas_arr[k], truth_all[i]), \
            f"misaligned row: meas_arr[{k}] != truth_all[{i}]"
    fe = np.median(np.abs(pred_arr / meas_arr - 1))
    print(f"PASS: matched-index alignment holds for {len(available)}/{n_tb} "
          f"available runs (skip pattern: every other run); "
          f"median |frac err| on the synthetic near-identity pair = {fe:.4f} "
          "(should be small, consistent with the ~5% injected noise)")
    assert fe < 0.15


if __name__ == "__main__":
    check_zi_slicing()
    print()
    check_matched_index_alignment()
    print("\nALL CHECKS PASSED")
