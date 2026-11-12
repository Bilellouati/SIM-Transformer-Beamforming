"""
Collapse and per-sample fairness diagnostics for the K-agnostic model.

Checks, for each K in {1,2,3,4}, on the full saved validation set:
  - output diversity (unique phase configurations, consecutive-sample
    correlation) -- rules out the model memorizing one fixed answer;
  - per-sample worst/best user rate ratio -- the training-time average
    per-user rates can look fair while individual hard channel draws are
    still very unequal, which this catches.
"""
import numpy as np
from scipy.io import loadmat

RESULTS_FILE = "transformer_Kgeneric_bit2_results.mat"

d = loadmat(RESULTS_FILE)

for K in [1, 2, 3, 4]:
    theta = d[f"K{K}_theta"]
    sumrate = d[f"K{K}_sumrate"][0]
    rates = d[f"K{K}_rates"]

    flat = theta.reshape(theta.shape[0], -1)
    n_unique = np.unique(flat, axis=0).shape[0]

    corrs = []
    for i in range(min(500, theta.shape[0] - 1)):
        a, b = flat[i], flat[i + 1]
        if a.std() > 1e-6 and b.std() > 1e-6:
            corrs.append(np.corrcoef(a, b)[0, 1])
    mean_corr = np.mean(corrs) if corrs else float("nan")

    print(f"\n=== K={K} ===")
    print(f"Unique phase configs: {n_unique}/{theta.shape[0]}  "
          f"mean consecutive-sample corr: {mean_corr:.4f}")
    print(f"Sum rate: mean={sumrate.mean():.4f}  std={sumrate.std():.4f}")

    if K > 1:
        ratio = rates.max(axis=1) / (rates.min(axis=1) + 1e-9)
        worst = rates.min(axis=1)
        pct = np.percentile(ratio, [50, 75, 90, 99])
        starved = (worst < 0.1).mean() * 100
        print(f"Max/min per-user rate ratio (per sample) percentiles "
              f"[50,75,90,99]: {pct}")
        print(f"Samples with worst-user rate < 0.1 bit/s/Hz: {starved:.1f}%")
