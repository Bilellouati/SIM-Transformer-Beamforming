"""
Multi-user (K=4) channel generation for the SIM Transformer controller, stage 2.

Unlike the K=1 stage, Method 2 (end-to-end) does not need a classical-optimizer
teacher (no theta* labels), so this script only generates the Rician channels
H for K=4 users -- much faster than the K=1 dataset generation, since no
iterative optimization is run per channel.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.io import savemat
from datetime import datetime

from functions import (
    distance_dnn, correlation_matrix, pathloss, generate_rice_channels,
)

# ----------------------------- Configuration -----------------------------
SEED = 42
K = 4          # number of users
N = 49         # meta-atoms per SIM layer
Nx, Ny = 7, 7
K_FACTOR = 1   # Rician K-factor (balanced LoS/NLoS, same choice as K=1 stage)
M_SAMPLES = 8000

c = 3e8
f = 28e9
lamda = c / f
H_BS = 10
Tsim = 5 * lamda   # kept for pathloss distance formula consistency with K=1 stage
d_UE = 10          # spacing between users (1D linear placement, same convention as SIM_NN_Phase*.py)
d0 = 1
nbar = 3.5
C0 = 10 ** (-40 / 10)

OUT_FILE = "channels_K%d_N%d.mat" % (K, N)

# ----------------------------- Generation -----------------------------
np.random.seed(SEED)

Dnn = distance_dnn(N, Nx, lamda)
Rnn = correlation_matrix(N, Dnn, lamda)
Betak = pathloss(K, H_BS, Tsim, d_UE, C0, d0, nbar)  # (K,1) path loss per user, fixed positions

H_all = np.zeros((M_SAMPLES, K, N), dtype=np.complex64)

t0 = datetime.now()
for i in range(M_SAMPLES):
    H_i, _ = generate_rice_channels(K, N, 1, Betak, Rnn, K_FACTOR)  # (K, N, 1)
    H_all[i] = H_i[:, :, 0]
    if (i + 1) % max(1, M_SAMPLES // 10) == 0:
        print(f"[{i+1}/{M_SAMPLES}] generated")

elapsed = (datetime.now() - t0).total_seconds()
print(f"Done. Generated {M_SAMPLES} samples (K={K} users each) in {elapsed:.1f}s")

metadata = {
    "K": K, "N": N, "Nx": Nx, "Ny": Ny, "K_factor": K_FACTOR, "M_samples": M_SAMPLES,
    "seed": SEED, "generated_on": datetime.now().isoformat(),
    "frequency_Hz": f, "d_UE_m": d_UE, "H_BS_m": H_BS,
    "description": (
        "Multi-user (K=4) Rician channels for end-to-end Transformer training (Method 2). "
        "No classical-optimizer labels included -- Method 2 only needs H."
    ),
}

savemat(OUT_FILE, {"H": H_all, "Betak": Betak, "metadata": metadata})
print(f"Saved to {OUT_FILE}")
