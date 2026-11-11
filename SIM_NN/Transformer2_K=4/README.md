# K=4 Multi-User Generalization (Method 2, end-to-end)

Extends the validated K=1 end-to-end Transformer to K=4 simultaneous users.
Dataset: fixed K_factor=1, fixed user positions (`channels_K4_N49.mat`, 8,000 Rician
channel realizations, 7,200 train / 800 validation). No classical-optimizer labels
needed (Method 2 only requires H).

## Formulation

The Transformer takes the channel matrices `H = {h_1, ..., h_K}` as input and outputs
the per-layer SIM phase shifts `theta` (which build the cascaded matrix `G`) and the
per-user power split `P = {P_1, ..., P_K}` (summing to the total power budget).

**Per-user rate** (k = 1..K):
```
signal_k = P_k * |h_k^H . G . w_k|^2
interf_k = sum_{k'!=k} P_k' * |h_k^H . G . w_k'|^2 + noise
R_k = log2(1 + signal_k / interf_k)
SumRate = sum_k R_k
```
Interference from user `k'` scales with `P_k'` (that user's own power), not `P_k` --
the correct multi-user SINR formula under non-equal power allocation.

**Training loss** (soft max-min fairness, not raw sum-rate):
```
loss = alpha * logsumexp(-R / alpha),   alpha = 0.3
```
As `alpha -> 0` this approaches `-min_k(R_k)`, so minimizing the loss maximizes the
worst-served user's rate rather than the sum rate directly. This is what keeps
per-user rates nearly equal despite the ~x43 near-far power imbalance between users
-- a pure `-SumRate` loss was tried first and collapsed to one user taking 100% of
the service (see stage 1 below).

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
| `test_channel_K4.py` | Runs a trained model on a given channel and reports phases, power allocation, and achieved rates |
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
pip install torch numpy scipy matplotlib   # Python 3.11, CPU-only PyTorch is enough

python generate_channels_K4.py             # ~instant, 8000 channels, 4 fixed users
python train_transformer_K4.py             # stage 1/2 history (naive -> QAT + fairness), ~15-25 min

# Stage 3 (final): joint phase + power control, one model per bit depth
python train_transformer_K4_powerctrl.py 2 # our focus: 2-bit, ~20-30 min
python train_transformer_K4_powerctrl.py 1 # repeat for bit = 1, 3, 4
python train_transformer_K4_powerctrl.py 3
python train_transformer_K4_powerctrl.py 4

python make_comparison_K4.py               # builds the 1-4 bit comparison table/plots
```
Run these from inside this folder (`SIM_NN/Transformer2_K=4/`) — output files are saved
with relative paths. All trained weights and results already exist in this folder;
re-running will overwrite them.

## Testing a custom channel

`test_channel_K4.py` loads a trained model for a chosen bit depth and evaluates it on
any given channel, without retraining.

```bash
# Quick check against an existing validation channel
python test_channel_K4.py --bits 2 --index 0

# Evaluate a custom channel from a .mat file with a variable H of shape
# (4, 49) or (M, 4, 49), matching the convention used in channels_K4_N49.mat
python test_channel_K4.py --bits 2 --mat your_channel.mat

# Optionally save the result (phases, power, rates) to a .mat file
python test_channel_K4.py --bits 2 --mat your_channel.mat --out results.mat
```

It prints the sum rate and, for each user, the achieved rate and allocated power.
The channel normalization statistics used during training are not stored in the
weights file, so they are recomputed deterministically from `channels_K4_N49.mat`
with the training random seed, reproducing the exact same values every time. This
has been verified to reproduce the saved validation results exactly.
