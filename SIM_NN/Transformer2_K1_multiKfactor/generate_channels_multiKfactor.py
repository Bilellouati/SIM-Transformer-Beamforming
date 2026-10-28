"""
Multi-K_factor channel generation for K=1, Method 2 (end-to-end).

Unlike the previous K=1 stage (fixed K_factor=1), this generates channels
with a RANDOM K_factor per sample, drawn uniformly in [0, 10] -- spanning
pure Rayleigh (K_factor=0, no line-of-sight) to near-LoS (K_factor=10).
This is the actual generalization dataset: the goal is for the Transformer
to learn a mapping that works across this whole spectrum of channel
conditions, not just one fixed type.

No classical-optimizer labels needed (Method 2 only needs H).
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.io import savemat
from datetime import datetime

from functions import distance_dnn, correlation_matrix, pathloss, generate_rice_channels

# ----------------------------- Configuration -----------------------------
SEED = 42
K = 1
N = 49
Nx, Ny = 7, 7
M_SAMPLES = 20000
K_FACTOR_MIN = 0.0
K_FACTOR_MAX = 10.0

c = 3e8
f = 28e9
lamda = c / f
H_BS = 10
Tsim = 5 * lamda
d_UE = 10
d0 = 1
nbar = 3.5
C0 = 10 ** (-40 / 10)

OUT_FILE = "channels_K1_multiKfactor_N%d.mat" % N

# ----------------------------- Generation -----------------------------
np.random.seed(SEED)

Dnn = distance_dnn(N, Nx, lamda)
Rnn = correlation_matrix(N, Dnn, lamda)
Betak = pathloss(K, H_BS, Tsim, d_UE, C0, d0, nbar)

K_factors = np.random.uniform(K_FACTOR_MIN, K_FACTOR_MAX, size=M_SAMPLES)

H_all = np.zeros((M_SAMPLES, N), dtype=np.complex64)

t0 = datetime.now()
for i in range(M_SAMPLES):
    H_i, _ = generate_rice_channels(K, N, 1, Betak, Rnn, K_factors[i])
    H_all[i] = H_i[:, :, 0][0]
    if (i + 1) % max(1, M_SAMPLES // 10) == 0:
        print(f"[{i+1}/{M_SAMPLES}] generated")

elapsed = (datetime.now() - t0).total_seconds()
print(f"Done. Generated {M_SAMPLES} samples in {elapsed:.1f}s")
print(f"K_factor range: [{K_factors.min():.2f}, {K_factors.max():.2f}]  mean={K_factors.mean():.2f}")

metadata = {
    "K": K, "N": N, "Nx": Nx, "Ny": Ny, "M_samples": M_SAMPLES,
    "K_factor_min": K_FACTOR_MIN, "K_factor_max": K_FACTOR_MAX,
    "seed": SEED, "generated_on": datetime.now().isoformat(),
    "frequency_Hz": f, "d_UE_m": d_UE, "H_BS_m": H_BS,
    "description": (
        "K=1 Rician channels with RANDOM K_factor per sample, uniform in [0,10] "
        "(pure Rayleigh to near-LoS). Generated for Method 2 (end-to-end) training "
        "to test/achieve generalization across channel types, not just one fixed "
        "K_factor. No classical-optimizer labels included."
    ),
}

savemat(OUT_FILE, {"H": H_all, "K_factor": K_factors, "Betak": Betak, "metadata": metadata})
print(f"Saved to {OUT_FILE}")
