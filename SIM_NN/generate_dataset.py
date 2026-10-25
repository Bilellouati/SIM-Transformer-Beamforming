"""
Dataset generation for Transformer-based SIM phase-control training.

Generates (H, theta*) pairs via the classical gradient-ascent optimizer
(gradient_func / objective_func in functions.py) over many random Rician
channel realizations, at K=1 user. This dataset is meant to be used for
supervised pretraining of a Transformer controller (H -> theta), with the
classical optimizer acting as the "teacher".

Output: a single .mat file containing H, theta, rate, a few saved
convergence curves, and a metadata struct documenting exactly how the
dataset was generated (for reproducibility).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from datetime import datetime

from functions import (
    distance_dnn, distance_layers, transmission_matrix,
    distance_BS_Layer, transmission_vector,
    pathloss, correlation_matrix, generate_rice_channels,
    gradient_func, objective_func,
)

# ----------------------------- Configuration -----------------------------
SEED = 42

# Users / BS (K=1 validation stage, per professor's suggested progression)
K = 1
d_UE = 10
PT = 10  # dBm
PTl = 0.001 * (10 ** (PT / 10))
M = K  # BS antennas (matches convention used in SIM_NN_Phase*.py: M = K)
H_BS = 10
BSantGain = 5  # dBi
BSantGainL = 10 ** (BSantGain / 10)
budget = M * BSantGainL * PTl

# SIM geometry
L = 5  # metasurface layers
Nx, Ny = 7, 7
N = Nx * Ny  # meta-atoms per layer
NbitQ = 2
c = 3e8
f = 28e9  # Hz
lamda = c / f
dx = lamda / 2
dy = lamda / 2
Tsim = 5 * lamda
d_layer = Tsim / L
Delta_theta = (2 * np.pi) / (2 ** NbitQ)

# Noise
sigma2k = -104  # dBm
wp = 0.001 * (10 ** (sigma2k / 10))  # W

# Path loss
d0 = 1
nbar = 3.5
C0 = 10 ** (-40 / 10)

# Rician fading
K_FACTOR = 1

# Dataset size / optimizer settings
M_SAMPLES = 8000
N_ITER_MAX = 200
LR_INIT = 0.5
LR_MIN = 1e-5
BACKTRACK_TRIES = 15
N_CONVERGENCE_CURVES_SAVED = 5  # how many sample trajectories to keep for sanity plots

OUT_FILE = "sim_dataset_K1_L%d_N%d.mat" % (L, N)
PLOT_FILE = "sim_dataset_convergence_K1_L%d_N%d.png" % (L, N)

# ------------------------------------------------------------------------

np.random.seed(SEED)

# Physical geometry (fixed across all channel samples)
Dnn = distance_dnn(N, Nx, lamda)
Dl = distance_layers(L, N, d_layer, Dnn)
cosXl = np.zeros((L, N, N), dtype=float)
for l in range(L):
    for n in range(N):
        for npr in range(N):
            cosXl[l, n, npr] = d_layer / Dl[l, n, npr]
Wl = transmission_matrix(L, N, Dl, dx, dy, cosXl, lamda)
D1 = distance_BS_Layer(N, M, Nx, Ny, lamda, d_layer)
W1 = transmission_vector(N, M, d_layer, dx, dy, D1, lamda)

Betak = pathloss(K, H_BS, Tsim, d_UE, C0, d0, nbar)
Rnn = correlation_matrix(N, Dnn, lamda)

P = np.diag([budget] * K)
B = np.array([i * Delta_theta for i in range(2 ** NbitQ)], dtype=np.float64)


def compute_rate(theta, H):
    """Continuous (non-quantized) sum-rate for a given phase configuration."""
    PHI = np.zeros((L, N, N), dtype=complex)
    for l in range(L):
        PHI[l, :, :] = np.diag(np.exp(1j * theta[l]))
    G = PHI[0, :, :]
    for l in range(1, L):
        G = np.matmul(np.matmul(PHI[l, :, :], Wl[l, :, :]), G)
    Rk = np.zeros((K,), dtype=float)
    for k in range(K):
        Pk = P[k, k]
        sig = Pk * np.abs(np.conjugate(H[k, :, 0]).reshape(1, N) @ G @ W1[:, k].reshape(N, 1)) ** 2
        interf = sum(
            (P[kpr, kpr] * np.abs(np.conjugate(H[k, :, 0]).reshape(1, N) @ G @ W1[:, kpr].reshape(N, 1)) ** 2
             for kpr in range(K) if kpr != k),
            np.zeros((1, 1)),
        ) + wp
        Rk[k] = np.log2(1 + (sig / interf)[0, 0].real)
    return Rk.sum()


def optimize_phases(H, n_iter=N_ITER_MAX, lr_init=LR_INIT):
    """Gradient ascent with backtracking line search: theta^(i+1) = theta^i + mu * dR/dtheta."""
    theta = np.random.uniform(0, 2 * np.pi, size=(L, N))
    lr = lr_init
    curve = [compute_rate(theta, H)]
    for _ in range(n_iter):
        grad_ascent_dir = -gradient_func(P, H, W1, K, L, N, wp, theta, Wl)  # gradient_func returns -dR
        rate_cur = curve[-1]
        accepted = False
        step = lr
        for _ in range(BACKTRACK_TRIES):
            theta_new = np.mod(theta + step * grad_ascent_dir, 2 * np.pi)
            rate_new = compute_rate(theta_new, H)
            if rate_new > rate_cur:
                theta = theta_new
                curve.append(rate_new)
                lr = step  # keep the successful step size for next iteration
                accepted = True
                break
            step /= 2
        if not accepted:
            break  # converged: no improving step found even at the smallest scale
        if lr < LR_MIN:
            break
    return theta, np.array(curve)


# ----------------------------- Main generation loop -----------------------------
H_all = np.zeros((M_SAMPLES, N), dtype=np.complex64)
theta_all = np.zeros((M_SAMPLES, L * N), dtype=np.float32)
rate_all = np.zeros((M_SAMPLES,), dtype=np.float32)
saved_curves = []

t0 = datetime.now()
for i in range(M_SAMPLES):
    H_i, _ = generate_rice_channels(K, N, 1, Betak, Rnn, K_FACTOR)
    theta_opt, curve = optimize_phases(H_i)

    # Quantize to the hardware's NbitQ codebook for the final stored target
    theta_q = theta_opt.copy()
    rate_q = objective_func(P, H_i, theta_q, W1, N, wp, Wl, B)  # mutates theta_q in place (quantizes it)

    H_all[i, :] = H_i[0, :, 0]
    theta_all[i, :] = theta_q.reshape(-1)
    rate_all[i] = rate_q.sum()

    if i < N_CONVERGENCE_CURVES_SAVED:
        saved_curves.append(curve)

    if (i + 1) % max(1, M_SAMPLES // 20) == 0:
        print(f"[{i+1}/{M_SAMPLES}] mean rate so far: {rate_all[:i+1].mean():.4f} bit/s/Hz")

elapsed = (datetime.now() - t0).total_seconds()
print(f"Done. Generated {M_SAMPLES} samples in {elapsed:.1f}s "
      f"(mean rate {rate_all.mean():.4f}, std {rate_all.std():.4f} bit/s/Hz)")

# ----------------------------- Save dataset -----------------------------
metadata = {
    "K": K, "L": L, "N": N, "Nx": Nx, "Ny": Ny, "NbitQ": NbitQ,
    "K_factor": K_FACTOR, "M_samples": M_SAMPLES,
    "n_iter_max": N_ITER_MAX, "lr_init": LR_INIT, "backtrack_tries": BACKTRACK_TRIES,
    "seed": SEED, "generated_on": datetime.now().isoformat(),
    "frequency_Hz": f, "Tsim_m": Tsim, "d_UE_m": d_UE,
    "description": (
        "Supervised dataset for Transformer-based SIM phase control. "
        "H: channel per user per meta-atom (M_samples x N, complex). "
        "theta: quantized optimal phase shifts from classical gradient-ascent "
        "optimizer (M_samples x L*N, radians). rate: achieved sum-rate per "
        "sample (bit/s/Hz). Generated with K=1 user as the validation stage "
        "before extending to K>1."
    ),
}

savemat(OUT_FILE, {
    "H": H_all,
    "theta": theta_all,
    "rate": rate_all,
    "convergence_curves": np.array(saved_curves, dtype=object),
    "metadata": metadata,
})
print(f"Saved dataset to {OUT_FILE}")

# ----------------------------- Convergence sanity plot -----------------------------
plt.figure(figsize=(6, 4))
for idx, curve in enumerate(saved_curves):
    plt.plot(curve, label=f"sample {idx}")
plt.xlabel("Gradient-ascent iteration")
plt.ylabel("Rate [bit/s/Hz]")
plt.title("Classical optimizer convergence (K=1 sanity check)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
print(f"Saved convergence plot to {PLOT_FILE}")

plt.figure(figsize=(6, 4))
plt.hist(rate_all, bins=40)
plt.xlabel("Rate [bit/s/Hz]")
plt.ylabel("Count")
plt.title("Distribution of achieved rates across the dataset")
plt.tight_layout()
plt.savefig(PLOT_FILE.replace(".png", "_hist.png"), dpi=150)
