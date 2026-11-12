# K-Agnostic Multi-User Controller (Method 2, end-to-end)

A single Transformer that accepts any number of users K in {1, 2, 3, 4} at
inference time, without retraining. `Transformer2_K=4/` trains one model
hard-wired to exactly 4 users; this chapter instead tokenizes each user
separately and lets self-attention operate over a variable-length sequence,
so the same weights handle K=1 through K=4.

Dataset: fixed K_factor=1, same 4 fixed user positions as `Transformer2_K=4/`
(users 1..K are always the K closest of those 4 positions), 30,000 Rician
channel realizations generated independently per K
(`channels_Kgeneric_N49.mat`).

## Why this chapter matters

`Transformer2_K=4/` only knows K=4: a SIM deployment serving 1, 2, or 3 users
would need a separate model trained from scratch for each case. In practice
the number of active users in a cell changes constantly, so a controller
tied to one fixed K is not something that could actually be deployed as-is.

This chapter removes that constraint: **one trained model** covers K=1
through K=4 at inference time, with no retraining and no need to know K in
advance. That is the practical value of this result -- not a raw
performance number, but operating flexibility a fixed-K model cannot offer.
The cost of that flexibility is quantified honestly below: some sum-rate
loss at the hardest case (K=4) relative to a model specialized for exactly
4 users, and a residual (greatly reduced, not eliminated) per-sample
fairness tail.

## Architecture

Each user's channel `h_k` is projected to a token; a learned aggregator
("CLS") token is prepended; the sequence `[agg, token_1, ..., token_K]` is
passed through a Transformer encoder. The aggregator token's output drives
the per-layer SIM phases `theta` (shared decision for all users), and each
user token's own output drives that user's power share -- both independent
of K in parameter count, so the model generalizes across user counts by
construction rather than by padding a fixed-size input.

## Formulation

Same per-user rate, sum rate, and soft max-min fairness loss as
`Transformer2_K=4/` (see that chapter's README for the exact formulas).
The only addition is K-aware batch sampling during training: each
mini-batch is drawn for one K at a time, sampled with weights favoring
higher K (`[1, 2, 3, 4]` normalized), since higher-K cases are harder and
more prone to being neglected by an unweighted average.

Early stopping and model selection use the **worst-served K**, not the
average across K:
```
vscore = min over K of softmin_objective(rates_K)
```
rather than a simple mean over K's soft-min scores, so that K=1's naturally
high rate cannot mask poor performance at K=3 or K=4.

## 1. Base model -- `train_transformer_Kgeneric.py`

Per-user tokenization + weighted K-sampling + worst-K early stopping (soft
max-min fairness, alpha=0.3), trained from scratch.

| K | Sum rate | Per-user rate (mean) | Power split |
|---|---|---|---|
| 1 | 16.31 | 16.31 | 100% |
| 2 | 6.19 | 3.06 / 3.13 | ~50% / 50% |
| 3 | 3.19 | 0.80 / 1.52 / 0.87 | ~33% each |
| 4 | 2.64 | 0.71 / 0.67 / 0.64 / 0.63 | ~25% each |

No collapse (verified with `diagnose_Kgeneric_fairness.py`): all 3,000
validation samples per K produce distinct phase configurations, with
low-to-moderate consecutive-sample correlation (0.47-0.74).

But the averages above hide a real **per-sample fairness tail**: looking at
the worst/best user rate ratio within each individual sample rather than
the K-wide average, K=4 leaves one user nearly cut off
(< 0.1 bit/s/Hz) on **11.5%** of validation channels, even though the
average per-user rate looks balanced (0.63-0.71). The soft max-min loss
improves the *average* worst-case rate but does not by itself guarantee
fairness on every individual hard channel realization.

## 2. Fairness fine-tune -- same script, `--resume` + sharper alpha

Warm-started from the base model's weights, continued training with a
sharper soft-min (alpha=0.15 vs 0.3, i.e. closer to the true per-sample
minimum rather than a smoothed average) and a lower learning rate
(3e-4), specifically to target the tail problem found above. ~40 minutes
on GPU, no dataset regeneration needed.

| K | Sum rate (base -> fine-tuned) | Starved samples, < 0.1 bit/s/Hz (base -> fine-tuned) | 99th-pct worst/best ratio (base -> fine-tuned) |
|---|---|---|---|
| 2 | 6.19 -> 6.23 | 0.8% -> 0.9% | 26x -> 28x |
| 3 | 3.19 -> 3.24 | 4.2% -> 2.6% | 47x -> 30x |
| 4 | 2.64 -> 2.59 | **11.5% -> 4.8%** | **128x -> 23x** |

The fine-tune more than halves the K=4 starvation rate and cuts the
worst-case (99th percentile) unfairness by ~5.5x, essentially for free
(-2% sum rate). K=3 improves similarly; K=2 was already fine and stays
fine. **`transformer_Kgeneric_bit2_finetune_weights.pt` is this chapter's
final model.**

This is a genuine improvement, not a fix: the fairness tail is reduced,
not eliminated. Some of the remaining tail may reflect a real physical
limit (certain Rician fades may not admit a fair split at all) rather
than a correctable model deficiency -- this is a documented, honest
limitation of the result, not something to infer only from the averages.

K=4's sum rate (2.59-2.64) remains below the K=4-dedicated model's 4.31
(`Transformer2_K=4/`, ~60% of its performance) -- the price of one model
covering every K instead of specializing for exactly 4 users.

## Files

| File | Content |
|---|---|
| `generate_channels_Kgeneric.py` | Generates the 30,000-channel-per-K dataset (K=1..4, same fixed positions as `Transformer2_K=4/`) |
| `channels_Kgeneric_N49.mat` | The dataset |
| `train_transformer_Kgeneric.py` | Training script: per-user tokenization, weighted K-sampling, worst-K early stopping |
| `diagnose_Kgeneric_fairness.py` | Collapse check + per-sample fairness-tail diagnostics |
| `transformer_Kgeneric_bit2_weights.pt` / `transformer_Kgeneric_bit2_results.mat` | Base model and full validation results (all K) |
| `transformer_Kgeneric_bit2_finetune_weights.pt` / `transformer_Kgeneric_bit2_finetune_results.mat` | **Final model**: fairness fine-tune of the base model |
| `train_log_Kgeneric_bit2.log` / `train_log_Kgeneric_bit2_finetune.log` | Full training logs, base and fine-tune |
| `training_curve_Kgeneric_bit2*.png`, `sumrate_by_K_bit2*.png` | Plots, base and fine-tune |

## How to reproduce

```bash
pip install torch numpy scipy matplotlib   # Python 3.11; CUDA build recommended, CPU-only also works

python generate_channels_Kgeneric.py                 # ~a few minutes, 30,000 channels per K
python train_transformer_Kgeneric.py --bits 2        # base model, several hours on GPU

# Fairness fine-tune: warm-start from the base model with a sharper
# per-sample fairness target (lower alpha = closer to the true worst-user
# minimum, rather than a smoothed average)
python train_transformer_Kgeneric.py --bits 2 --resume transformer_Kgeneric_bit2_weights.pt \
    --alpha 0.15 --lr 3e-4 --epochs 2000 --out-suffix _finetune   # ~40 min on GPU

python diagnose_Kgeneric_fairness.py                  # collapse + fairness-tail check
```
Run these from inside this folder (`SIM_NN/Transformer3_Kgeneric/`) -- output
files are saved with relative paths. The trained weights and results for
2-bit quantization already exist in this folder; re-running will overwrite
them. `diagnose_Kgeneric_fairness.py` reads `transformer_Kgeneric_bit2_results.mat`
by default -- point `RESULTS_FILE` at the `_finetune` file to check the final model.
