# Transformer SIM Controller — K=1 Development

First validation stage of the Transformer-based SIM phase-shift controller: a single-user (K=1)
scenario, following the step-by-step plan agreed with the professor (generate channels → implement
SIM model → classical optimization → dataset → train Transformer → compare).

## Contents

- `generate_dataset.py` — generates the channel dataset (8000 Rician channels, K_factor=1) and the
  classical-optimizer phase solutions (used by Method 1, and as a source of channels for Method 2).
- `sim_dataset_K1_L5_N49.mat` — the generated dataset (H, theta, rate, metadata).
- `sim_dataset_convergence_K1_L5_N49*.png` — sanity-check plots for the classical optimizer.
- `Dataset_Report.pdf` — report describing the dataset generation and validation.
- `Transformer_K1_Progress_Report.pdf` — final progress report with validated results.

## Two training methods tested

### Methode 1 — Supervised (imitation of the classical optimizer)
`train_transformer_K1.py` — trains the Transformer to reproduce the classical optimizer's phase
output (theta*) via a supervised loss. **Finding**: the classical optimizer itself turned out to be
an unreliable teacher (same channel, different random restarts, gives very different results), which
limits what this method can learn. Result: ~8.0 bit/s/Hz average, no reliable per-channel correlation
with the classical optimizer's result.

### Methode 2 — End-to-End (direct rate optimization)
`train_transformer_K1_endtoend.py` — trains the Transformer directly on the physical objective
(maximize the achieved rate), without needing theta* labels at all. This is the professor's
"second method". **Result: ~17.16 bit/s/Hz on unseen validation channels**, independently confirmed
two ways: (1) recomputed with a separate NumPy implementation of the physics, and (2) matched against
an independently-tuned Adam-based reference optimizer (multiple restarts), which converges to the
same level (~17.1 bit/s/Hz) on the same channels.

**Conclusion**: Method 2 (end-to-end) is the validated, working approach for this stage.

## How to reproduce

```bash
# 1. Generate the dataset (channels + classical optimizer solutions)
python generate_dataset.py

# 2a. Train with Method 1 (supervised)
python train_transformer_K1.py

# 2b. Train with Method 2 (end-to-end, recommended)
python train_transformer_K1_endtoend.py
```

Both training scripts expect `sim_dataset_K1_L5_N49.mat` in the same directory, and import shared
physics functions from `SIM_NN/functions.py` (one level up).

## Next steps

- Extend validation to the full 800-channel validation set.
- Extend to K>1 (multi-user, with interference) — reusing Method 2.
- Integrate the power-amplifier hardware-impairment model (Bussgang).
