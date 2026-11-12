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

## Results (2-bit quantization)

| K | Sum rate | Per-user rate (mean) | Power split |
|---|---|---|---|
| 1 | 16.31 | 16.31 | 100% |
| 2 | 6.19 | 3.06 / 3.13 | ~50% / 50% |
| 3 | 3.19 | 0.80 / 1.52 / 0.87 | ~33% each |
| 4 | 2.64 | 0.71 / 0.67 / 0.64 / 0.63 | ~25% each |

K=1 matches the dedicated single-user model. K=2 and K=4 show good average
fairness across users. K=3's middle user receives noticeably more rate than
the other two (~1.9x) on average.

K=4's sum rate (2.64) is below the K=4-dedicated model's 4.31
(`Transformer2_K=4/`, 61% of its performance) -- the price of one model
covering every K instead of specializing for exactly 4 users.

## Verified diagnostics (`diagnose_Kgeneric_fairness.py`)

- **No collapse**: all 3,000 validation samples per K produce distinct phase
  configurations (3000/3000 unique), with low-to-moderate consecutive-sample
  correlation (0.47-0.74) -- the model is responding to each channel, not
  memorizing one fixed answer.
- **Per-sample fairness tail**: the table above shows averages, which hide
  a real tail problem. Looking at the per-sample worst/best user rate ratio
  rather than the K-wide average:

  | K | Median ratio | 99th pct ratio | Samples with worst user < 0.1 bit/s/Hz |
  |---|---|---|---|
  | 2 | 1.9x | 26x | 0.8% |
  | 3 | 3.0x | 47x | 4.2% |
  | 4 | 2.9x | 128x | 11.5% |

  For K=4, about 1 in 9 channel draws leaves one user nearly cut off
  (< 0.1 bit/s/Hz) even though the average per-user rate looks balanced.
  The soft max-min loss improves the *average* worst-case rate but does not
  guarantee fairness on every individual hard channel realization -- some of
  this may be a genuine physical limit of certain Rician fades rather than a
  model deficiency, but it is a real, documented limitation of this chapter's
  result, not something to be inferred only from the training-time averages.

## Files

| File | Content |
|---|---|
| `generate_channels_Kgeneric.py` | Generates the 30,000-channel-per-K dataset (K=1..4, same fixed positions as `Transformer2_K=4/`) |
| `channels_Kgeneric_N49.mat` | The dataset |
| `train_transformer_Kgeneric.py` | Training script: per-user tokenization, weighted K-sampling, worst-K early stopping |
| `diagnose_Kgeneric_fairness.py` | Collapse check + per-sample fairness-tail diagnostics |
| `transformer_Kgeneric_bit2_weights.pt` / `transformer_Kgeneric_bit2_results.mat` | Trained model and full validation results (all K) |
| `train_log_Kgeneric_bit2.log` | Full training log |
| `training_curve_Kgeneric_bit2.png`, `sumrate_by_K_bit2.png` | Plots |

## How to reproduce

```bash
pip install torch numpy scipy matplotlib   # Python 3.11; CUDA build recommended, CPU-only also works

python generate_channels_Kgeneric.py                 # ~a few minutes, 30,000 channels per K
python train_transformer_Kgeneric.py --bits 2         # 2-bit, several hours on GPU
python diagnose_Kgeneric_fairness.py                  # collapse + fairness-tail check
```
Run these from inside this folder (`SIM_NN/Transformer3_Kgeneric/`) -- output
files are saved with relative paths. The trained weights and results for
2-bit quantization already exist in this folder; re-running will overwrite
them.
