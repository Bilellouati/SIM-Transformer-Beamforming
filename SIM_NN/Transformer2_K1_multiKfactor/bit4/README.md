# Bit = 4 (16 phase levels)

Evaluation of the trained multi-K_factor Transformer (`transformer_multiKfactor_weights.pt`,
one level up) with the SIM phase shifters quantized to **4 bits** (16 possible phase values) --
the finest resolution tested.

No retraining was done -- the model's continuous phase output was re-quantized to this finer
resolution and the rate was recomputed with the real physics model.

## Result
- Mean rate: **19.16 bit/s/Hz**
- +2.5% over 3 bits -- very close to the continuous (unquantized) ceiling of ~19.27 bit/s/Hz seen
  during training. Confirms diminishing returns: beyond 4 bits, further hardware precision would
  bring little additional gain.

## Files
- `results_bit4.mat` -- H (validation channels), K_factor per sample, theta_predicted (quantized to
  4 bits), rate per sample.
- `rate_vs_kfactor_bit4.png` -- achieved rate as a function of the channel's K_factor.
- `rate_distribution_bit4.png` -- histogram of achieved rate across all validation channels.

See `../Comparison_Report_BitQuantization.pdf` for the full comparison across all bit depths.
