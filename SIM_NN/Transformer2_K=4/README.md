# K=4 Multi-User Generalization (Method 2, end-to-end)

Extends the validated K=1 end-to-end Transformer to K=4 simultaneous users.
Dataset: fixed K_factor=1, fixed user positions (`channels_K4_N49.mat`, 8,000 Rician
channel realizations, 7,200 train / 800 validation). No classical-optimizer labels
needed (Method 2 only requires H).

Three stages, each fixing a problem found by diagnosis in the previous one:

## 1. `train_transformer_K4.py` -- naive sum-rate, continuous-trained
First attempt: train on continuous phases, quantize only at evaluation, optimize
raw sum rate. Result: continuous sum rate 15.50 bit/s/Hz, but **collapsed to
2.38 bit/s/Hz at 2-bit quantization** (85% loss) -- multi-user interference
nulling is far more sensitive to phase precision than single-user beamforming.
Diagnosed with `diagnose_K4.py`.

Also found (independent of quantization): the sum-rate objective, with no
fairness constraint, monopolizes ONE user 100% of the time under severe
near-far imbalance (Betak ratio up to x43 between users) -- confirmed with
`diagnose_K4_fairness.py` (dominant user in 800/800 validation samples).

## 2. Fix 1 -- quantization-aware training + fairness (still `train_transformer_K4.py`)
- Added a straight-through estimator (STE) quantizer inside the training loop
  (same idea as `SIM_NN_Phase.py`'s `fake_quant`), so the model learns phases
  robust to the DEPLOYED 2-bit resolution, not just the continuous relaxation.
- Replaced the sum-rate objective with the max-min fairness surrogate
  (soft-min via log-sum-exp, same as `SIM_NN_Phase.py`'s `custom_loss`).
- Result: per-user rate 1.00 / 0.99 / 1.02 / 1.06 bit/s/Hz (near-equal despite
  the x43 power imbalance), sum rate 4.06 bit/s/Hz. Verified not collapsed:
  800/800 unique phase configurations, low sample-to-sample correlation
  (0.13-0.44), real per-sample rate variance for every user.

## 3. Fix 2 -- joint phase + power control (`train_transformer_K4_powerctrl.py`)
User asked whether water-filling could improve the power split further.
Classical water-filling does NOT apply here: it assumes parallel,
non-interfering channels, while here all users share the same SIM beamforming
matrix and interfere with each other -- water-filling under interference
would reproduce the monopolization problem fairness was built to avoid.

Instead, the model's power allocation (per user, summing to the fixed total
budget) is made a second LEARNED output, jointly optimized with phase under
the same fairness objective and quantization-aware training.

Result: sum rate 4.31 bit/s/Hz (+6% over fixed equal-power split), per-user
rate 1.07 / 1.05 / 1.03 / 1.16 bit/s/Hz (still near-equal), and the worst-case
(soft-min) rate also improved (0.407 vs 0.370). The model learned to give the
closest user slightly more power (30.5% vs ~23% for the others). Verified not
collapsed the same way as stage 2.

## Files
| File | Content |
|---|---|
| `generate_channels_K4.py` | Generates the 8,000-channel K=4 dataset (fixed K_factor=1, fixed positions) |
| `channels_K4_N49.mat` | The dataset |
| `train_transformer_K4.py` | Stage 1 (naive) and stage 2 (QAT + fairness) -- see git history / in-file comments for which version is current |
| `train_transformer_K4_powerctrl.py` | Stage 3: joint phase + power control |
| `diagnose_K4.py` | Quantization sensitivity + output-diversity (collapse) check |
| `diagnose_K4_fairness.py` | Per-sample dominant-user check (fairness diagnosis) |
| `transformer_K4_weights.pt` / `transformer_K4_results.mat` | Stage 2 trained model and validation results |
| `transformer_K4_powerctrl_weights.pt` / `transformer_K4_powerctrl_results.mat` | Stage 3 trained model and validation results |
| `train_log_K4*.log` | Full training logs for each stage |
| `training_curve_K4*.png`, `sumrate_distribution_K4*.png`, `peruser_rate_K4*.png`, `power_allocation_K4.png` | Plots |

## Summary of results

| Stage | Sum rate | Per-user rate | Notes |
|---|---|---|---|
| 1. Naive (continuous-trained, sum-rate) | 2.38 (2-bit) / 15.50 (continuous) | 1 user monopolizes ~100% | Broken: not quantization-robust, not fair |
| 2. QAT + fairness | 4.06 | 1.00 / 0.99 / 1.02 / 1.06 | Fixed both issues |
| 3. QAT + fairness + learned power | **4.31** | 1.07 / 1.05 / 1.03 / 1.16 | Best: +6% sum rate, equally fair |

## How to reproduce
```bash
python generate_channels_K4.py           # ~instant, 8000 channels
python train_transformer_K4.py           # ~15-25 min (early stopping)
python train_transformer_K4_powerctrl.py # ~20-30 min (early stopping)
```
