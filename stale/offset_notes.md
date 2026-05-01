# Stars Channel Bias: Conclusions and Path Forward

## What the notebook shows

Five independent tests all point to the same diagnosis: the ~28% stellar mass under-prediction is a **norm_stats book-keeping error**, not a model failure.

| Test | Finding |
|------|---------|
| 1 — norm_stats vs. training data | `ns.target_mean[2]` stored as 1.5461, but recomputed from the actual training patches it is 1.3991 — a **−0.147 dex** mismatch. DM/Gas offsets are −0.046 and −0.036 dex (small). |
| 2 — model bias in normalized space | Stars gen mean ≈ truth mean ≈ 0 in latent space. The model learned the correct distribution; the bias appears only after denormalization through the wrong `target_mean`. |
| 3 — ratio vs. halo mass | `m_truth / m_gen` is flat across the full halo mass range (Spearman ρ ≈ 0). A model error would produce a mass-dependent slope. |
| 4 — per-sim consistency | Per-CV-sim correction factors cluster tightly (CoV ~2–3%) with no trend across cosmologies. A cosmology-dependent model failure would scatter this. |
| 5 — channel selectivity | Only Stars is biased (median error ~−28%). DM_hydro and Gas medians are <1%. Shared network weights would bias all channels together; only a per-channel norm offset isolates one channel. |

**Measured correction:** Stars factor = **1.3853×** (+0.1415 dex), equivalent to shifting `ns.target_mean[2]` from 1.5461 → **1.6876**.

---

## Path forward

There are three options, in increasing order of cleanliness:

### Option A — Post-hoc scalar correction (quickest, no retraining)
Multiply every generated Stars map by 1.3853 after inference. Apply this at evaluation / test-suite time. No checkpoint changes needed.

```python
correction = np.array([1.0002, 0.9949, 1.3853])   # DM, Gas, Stars
generated *= correction[None, :, None, None]
```

**Upside:** Zero cost.  
**Downside:** The stored `norm_stats.npz` stays wrong; anyone who re-runs inference without the correction will get biased results. Leaks into uncertainty estimates (the model's internal confidence is calibrated to the wrong mean).

### Option B — Fix norm_stats.npz and re-run inference (recommended next step)
Update `ns.target_mean[2]` in place and regenerate `generated_halos.npz` for the test suite without retraining.

```python
ns_data = dict(np.load('path/to/norm_stats.npz'))
ns_data['target_mean'][2] += np.log10(1.3853)   # +0.1415 dex
np.savez('path/to/norm_stats.npz', **ns_data)
# then re-run test suite inference
```

**Upside:** All downstream consumers of the checkpoint automatically get the right output. No retraining needed (~hours of inference vs. ~days of training).  
**Downside:** The checkpoint was trained with the wrong norm; the correction is post-hoc at the denormalization layer, which is fine given Test 2, but it is not a clean slate.

### Option C — Recompute norm_stats from scratch and retrain (cleanest)
Rerun `compute_norm_stats.py` on the current training set and launch a fresh training run. This eliminates the root cause entirely.

**Upside:** No residual inconsistency between training distribution and norm_stats.  
**Downside:** Full training cost (~days). Only warranted if other hyperparameter changes are also queued.

---

## Recommendation

**Do Option B now.** It is a one-line patch to `norm_stats.npz` and a test-suite re-run. The five tests give strong evidence that the model is correctly calibrated in latent space, so fixing the denormalization offset is sufficient to recover unbiased physical predictions. Schedule Option C alongside the next planned retraining run.
