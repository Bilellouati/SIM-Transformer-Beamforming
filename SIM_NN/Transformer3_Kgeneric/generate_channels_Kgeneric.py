"""
Channel generation for the K-agnostic SIM Transformer controller.

Unlike Transformer2_K=4 (one model hard-wired to exactly 4 users), this stage
trains a single model that accepts any number of users K in {1, 2, 3, 4} at
inference time. To do that, training itself must see examples at every K, so
this script generates one dataset per K value, all sharing the same fixed
user positions (users 1..K are always the K closest of the same 4 fixed
positions used in Transformer2_K=4 -- i.e. the K=2 dataset's two users are
the same two closest users as in the K=4 dataset, not a different scenario).
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
K_MAX = 4
N = 49
Nx, Ny = 7, 7
K_FACTOR = 1
M_SAMPLES = 30000

c = 3e8
f = 28e9
lamda = c / f
H_BS = 10
Tsim = 5 * lamda
d_UE = 10
d0 = 1
nbar = 3.5
C0 = 10 ** (-40 / 10)

OUT_FILE = "channels_Kgeneric_N%d.mat" % N

# ----------------------------- Generation -----------------------------
np.random.seed(SEED)

Dnn = distance_dnn(N, Nx, lamda)
Rnn = correlation_matrix(N, Dnn, lamda)
Betak_full = pathloss(K_MAX, H_BS, Tsim, d_UE, C0, d0, nbar)  # (K_MAX, 1), fixed positions

data = {}
t0 = datetime.now()
for K in range(1, K_MAX + 1):
    Betak_K = Betak_full[:K]
    H_K = np.zeros((M_SAMPLES, K, N), dtype=np.complex64)
    for i in range(M_SAMPLES):
        H_i, _ = generate_rice_channels(K, N, 1, Betak_K, Rnn, K_FACTOR)
        H_K[i] = H_i[:, :, 0]
    data[f"H_K{K}"] = H_K
    print(f"K={K}: generated {M_SAMPLES} samples")

elapsed = (datetime.now() - t0).total_seconds()
print(f"Done in {elapsed:.1f}s")

metadata = {
    "K_values": list(range(1, K_MAX + 1)), "N": N, "Nx": Nx, "Ny": Ny,
    "K_factor": K_FACTOR, "M_samples": M_SAMPLES, "seed": SEED,
    "generated_on": datetime.now().isoformat(), "frequency_Hz": f,
    "d_UE_m": d_UE, "H_BS_m": H_BS,
    "description": (
        "Channels for K in {1,2,3,4} sharing the same fixed user positions as "
        "Transformer2_K=4 (users 1..K are always the K closest of the same 4 "
        "fixed positions). Used to train a single Transformer that accepts "
        "any number of users K without retraining."
    ),
}

data["Betak_full"] = Betak_full
data["metadata"] = metadata
savemat(OUT_FILE, data)
print(f"Saved to {OUT_FILE}")
