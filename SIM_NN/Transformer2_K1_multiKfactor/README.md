# K=1 Multi-K_factor Generalization + Bit-Quantization Comparison

This folder covers two related pieces of work, both using the End-to-End method (Method 2)
validated in the previous K=1 stage:

1. **Generalization across channel types** -- instead of training on a single fixed K_factor=1
   Rician channel, the Transformer is trained on 20,000 channels with K_factor drawn uniformly in
   [0, 10] (pure Rayleigh to near-line-of-sight).
2. **Phase-quantization comparison (1-4 bits)** -- the same trained model, re-evaluated at
   different hardware phase-shifter resolutions, without retraining.

## Key files (this folder)

| File | Content |
|---|---|
| `generate_channels_multiKfactor.py` | Generates the 20,000-channel dataset (random K_factor per sample) |
| `train_transformer_multiKfactor.py` | Trains the Transformer end-to-end on this dataset |
| `eval_multibit.py` | Re-evaluates the trained model at 1/2/3/4-bit quantization (no retraining) |
| `transformer_multiKfactor_weights.pt` | The single trained model, shared across all bit-depth evaluations |
| `training_curve_multiKfactor.png` | Training convergence (continuous rate, pre-quantization) |
| `comparison_bits.png` | Mean rate vs. number of quantization bits |
| `Generalization_Report_multiKfactor.pdf` | Full report on the channel-type generalization result |
| `Comparison_Report_BitQuantization.pdf` | Full report comparing 1-4 bit quantization |

## Subfolders

Each `bitN/` subfolder contains the evaluation results for that specific quantization resolution
(see each subfolder's own `README.md` for details): `results_bitN.mat` (H, K_factor, predicted
theta, rate), plus `rate_vs_kfactor_bitN.png` and `rate_distribution_bitN.png`.

| Bits | Levels | Mean rate | Folder |
|---|---|---|---|
| 1 | 2 | 8.38 bit/s/Hz | `bit1/` |
| 2 | 4 | 16.41 bit/s/Hz | `bit2/` (also contains the full 20,000-channel dataset) |
| 3 | 8 | 18.69 bit/s/Hz | `bit3/` |
| 4 | 16 | 19.16 bit/s/Hz | `bit4/` |

## Key findings

- **Generalization**: the model performs nearly uniformly (within 0.12 bit/s/Hz) across the whole
  K_factor spectrum, confirming it learned a real channel-adaptive rule, not a fit to one
  distribution.
- **Quantization**: classic diminishing returns -- 1->2 bits nearly doubles the rate, each
  additional bit beyond that brings progressively smaller gains, approaching the continuous
  (unquantized) ceiling of ~19.27 bit/s/Hz.

## How to reproduce

```bash
python generate_channels_multiKfactor.py      # ~1 second, 20,000 channels
python train_transformer_multiKfactor.py      # ~30-45 minutes
python eval_multibit.py                       # ~1 minute, reuses the trained model
```
