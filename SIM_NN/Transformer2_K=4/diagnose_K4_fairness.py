"""
Check whether the K=4 model's near-total monopolization of User 3 is
sample-dependent (adaptive) or a constant collapse (ignoring the channel).
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from scipy.io import loadmat

d = loadmat("transformer_K4_results.mat")
rates = d["per_user_rate"]   # (N, 4)
print("Per-user rate stats (mean, std, min, max):")
for k in range(rates.shape[1]):
    r = rates[:, k]
    print(f"  User {k+1}: mean={r.mean():.4f} std={r.std():.4f} min={r.min():.4f} max={r.max():.4f}")

dominant_user = np.argmax(rates, axis=1)
print("\nWhich user gets the most rate, per sample (counts):")
for k in range(rates.shape[1]):
    print(f"  User {k+1} is dominant in {np.sum(dominant_user == k)} / {len(dominant_user)} samples")

print("\nFraction of samples where dominant user gets >90% of sum rate:",
      np.mean(rates.max(axis=1) / rates.sum(axis=1) > 0.9))
