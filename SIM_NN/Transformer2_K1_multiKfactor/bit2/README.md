# Bit = 2 (4 phase levels)

Evaluation of the trained multi-K_factor Transformer (`transformer_multiKfactor_weights.pt`,
one level up) with the SIM phase shifters quantized to **2 bits** (4 possible phase values:
0, pi/2, pi, 3pi/2) -- this is the configuration used throughout the project (same as the K=1
single-type stage).

## Result
- Mean rate: **16.41 bit/s/Hz**
- Nearly uniform across the whole K_factor spectrum (0-10): only 0.12 bit/s/Hz gap between the
  best and worst channel-type group -- confirms genuine generalization across channel types, not
  just across random realizations of one type.
- Large improvement over 1-bit (+96%), confirming 2 bits already captures most of the achievable
  gain from quantization.

## Files
- `channels_K1_multiKfactor_N49.mat` -- the full 20,000-channel training dataset (train + val).
- `results_bit2.mat` -- H (validation channels), K_factor per sample, theta_predicted (2-bit
  quantized), rate per sample.
- `rate_vs_kfactor_bit2.png` -- achieved rate as a function of the channel's K_factor.
- `rate_distribution_bit2.png` -- histogram of achieved rate across all validation channels.
- `orig_results.mat`, `orig_rate_vs_kfactor.png`, `orig_rate_distribution.png` -- the same
  2-bit evaluation as saved directly by `train_transformer_multiKfactor.py` (NbitQ = 2), under
  their original filenames (renamed here only to keep Windows path lengths short).

See `../Generalization_Report_multiKfactor.pdf` for the full generalization analysis, and
`../Comparison_Report_BitQuantization.pdf` for the comparison across all bit depths.
