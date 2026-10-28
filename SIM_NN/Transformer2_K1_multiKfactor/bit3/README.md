# Bit = 3 (8 phase levels)

Evaluation of the trained multi-K_factor Transformer (`transformer_multiKfactor_weights.pt`,
one level up) with the SIM phase shifters quantized to **3 bits** (8 possible phase values).

No retraining was done -- the model's continuous phase output was re-quantized to this finer
resolution and the rate was recomputed with the real physics model.

## Result
- Mean rate: **18.69 bit/s/Hz**
- +14% over 2 bits -- a smaller, diminishing-returns improvement compared to the 1->2 bit jump.

## Files
- `results_bit3.mat` -- H (validation channels), K_factor per sample, theta_predicted (quantized to
  3 bits), rate per sample.
- `rate_vs_kfactor_bit3.png` -- achieved rate as a function of the channel's K_factor.
- `rate_distribution_bit3.png` -- histogram of achieved rate across all validation channels.

See `../Comparison_Report_BitQuantization.pdf` for the full comparison across all bit depths.
