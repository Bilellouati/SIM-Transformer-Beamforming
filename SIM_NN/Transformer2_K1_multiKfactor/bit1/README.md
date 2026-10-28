# Bit = 1 (2 phase levels)

Evaluation of the trained multi-K_factor Transformer (`transformer_multiKfactor_weights.pt`,
one level up) with the SIM phase shifters quantized to **1 bit** (2 possible phase values: 0, pi).

No retraining was done -- the model's continuous phase output was re-quantized to this coarser
resolution and the rate was recomputed with the real physics model.

## Result
- Mean rate: **8.38 bit/s/Hz**
- This is the weakest resolution tested -- confirms 1-bit hardware is a poor choice for this system
  (about half the rate of 2-bit hardware).

## Files
- `results_bit1.mat` -- H (validation channels), K_factor per sample, theta_predicted (quantized to
  1 bit), rate per sample.
- `rate_vs_kfactor_bit1.png` -- achieved rate as a function of the channel's K_factor.
- `rate_distribution_bit1.png` -- histogram of achieved rate across all validation channels.

See `../Comparison_Report_BitQuantization.pdf` for the full comparison across all bit depths.
