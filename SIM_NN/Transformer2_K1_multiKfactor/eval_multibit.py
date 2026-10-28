"""
Evaluate the SAME trained end-to-end Transformer (multi-K_factor model) at
different phase-shifter quantization resolutions (1, 2, 3, 4 bits).

No retraining needed: the model was trained on the continuous phase
representation (cos/sin), independent of any quantization level. Only the
final quantization step (snapping the continuous phase to the nearest
hardware-allowed codeword) changes per bit depth, so we can test several
bit depths almost instantly by re-quantizing the same model output.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from functions import (
    distance_dnn, distance_layers, transmission_matrix,
    distance_BS_Layer, transmission_vector,
)

# ----------------------------- Setup (same physics/config as training) -----------------------------
DATASET_FILE = "channels_K1_multiKfactor_N49.mat"
WEIGHTS_FILE = "transformer_multiKfactor_weights.pt"
SEED = 0
VAL_FRACTION = 0.1
D_MODEL, NUM_LAYERS, DROPOUT = 128, 3, 0.1
BIT_DEPTHS = [1, 2, 3, 4]

d = loadmat(DATASET_FILE)
H = d["H"].astype(np.complex64)
K_factor = d["K_factor"][0].astype(np.float32)
meta = d["metadata"][0, 0]
K = int(meta["K"][0, 0]); N = int(meta["N"][0, 0])
Nx = int(meta["Nx"][0, 0]); Ny = int(meta["Ny"][0, 0])
M_total = H.shape[0]
L = 5

d_UE = 10; PT = 10; PTl = 0.001 * (10 ** (PT / 10)); M_ant = K; H_BS = 10
BSantGain = 5; BSantGainL = 10 ** (BSantGain / 10); budget = M_ant * BSantGainL * PTl
c = 3e8; f = 28e9; lamda = c / f; dx = lamda / 2; dy = lamda / 2
Tsim = 5 * lamda; d_layer = Tsim / L
sigma2k = -104; wp = 0.001 * (10 ** (sigma2k / 10))

Dnn = distance_dnn(N, Nx, lamda)
Dl = distance_layers(L, N, d_layer, Dnn)
cosXl = np.zeros((L, N, N), dtype=float)
for l in range(L):
    for n in range(N):
        for npr in range(N):
            cosXl[l, n, npr] = d_layer / Dl[l, n, npr]
Wl = transmission_matrix(L, N, Dl, dx, dy, cosXl, lamda)
D1 = distance_BS_Layer(N, M_ant, Nx, Ny, lamda, d_layer)
W1 = transmission_vector(N, M_ant, d_layer, dx, dy, D1, lamda)
Wl_t = torch.tensor(Wl, dtype=torch.complex64)
W1_t = torch.tensor(W1, dtype=torch.complex64)


def achieved_rate_batch(theta_batch, H_batch):
    Bn = theta_batch.shape[0]
    PHIt = torch.exp(1j * theta_batch.reshape(Bn, L, N))
    G = torch.diag_embed(PHIt[:, 0, :])
    for l in range(1, L):
        Dl_ = torch.diag_embed(PHIt[:, l, :])
        Wl_l = Wl_t[l].unsqueeze(0).expand(Bn, -1, -1)
        G = torch.bmm(torch.bmm(Dl_, Wl_l), G)
    Hc = torch.conj(H_batch).reshape(Bn, 1, N)
    GW = torch.bmm(G, W1_t.unsqueeze(0).expand(Bn, -1, -1))
    HGW = torch.bmm(Hc, GW).reshape(Bn)
    signal = budget * torch.abs(HGW) ** 2
    return torch.log2(1 + signal / wp).real


# ----------------------------- Rebuild the same train/val split -----------------------------
np.random.seed(SEED)
idx = np.random.permutation(M_total)
n_val = int(M_total * VAL_FRACTION)
val_idx, train_idx = idx[:n_val], idx[n_val:]

H_ri = np.concatenate([H.real, H.imag], axis=1)
H_mean = H_ri[train_idx].mean(axis=0, keepdims=True)
H_std = H_ri[train_idx].std(axis=0, keepdims=True) + 1e-8
H_norm = (H_ri - H_mean) / H_std

X_val = torch.tensor(H_norm[val_idx], dtype=torch.float32)
Hc_val = torch.tensor(H[val_idx], dtype=torch.complex64)
Kf_val = K_factor[val_idx]


# ----------------------------- Load model (once) -----------------------------
class SIMTransformerK1(nn.Module):
    def __init__(self, N, L, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.N, self.L = N, L
        self.input_proj = nn.Linear(2 * N, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, 2 * L * N),
        )

    def forward(self, x):
        tok = self.input_proj(x).unsqueeze(1)
        tok = self.transformer(tok)
        out = self.output_head(tok.squeeze(1))
        LN = self.L * self.N
        cos_raw, sin_raw = out[:, :LN], out[:, LN:]
        norm = torch.sqrt(cos_raw ** 2 + sin_raw ** 2 + 1e-8)
        return cos_raw / norm, sin_raw / norm


model = SIMTransformerK1(N, L, d_model=D_MODEL, num_layers=NUM_LAYERS, dropout=DROPOUT)
model.load_state_dict(torch.load(WEIGHTS_FILE, weights_only=True))
model.eval()

with torch.no_grad():
    cos_pred, sin_pred = model(X_val)
    phases_pred = (torch.atan2(sin_pred, cos_pred) % (2 * np.pi)).numpy()  # continuous phase, computed ONCE

# ----------------------------- Evaluate each bit depth -----------------------------
summary = []
for nbit in BIT_DEPTHS:
    delta_theta = (2 * np.pi) / (2 ** nbit)
    B = np.array([i * delta_theta for i in range(2 ** nbit)], dtype=np.float32)

    diffs = np.abs(phases_pred[:, :, None] - B[None, None, :])
    thetas_q = B[np.argmin(diffs, axis=2)].astype(np.float32)

    rate = achieved_rate_batch(torch.tensor(thetas_q), Hc_val).numpy()

    outdir = f"bit{nbit}"
    os.makedirs(outdir, exist_ok=True)

    # per-K_factor breakdown
    bins = [0, 2, 4, 6, 8, 10]
    bin_means = []
    for i in range(len(bins) - 1):
        mask = (Kf_val >= bins[i]) & (Kf_val < bins[i + 1])
        bin_means.append(rate[mask].mean() if mask.sum() > 0 else float("nan"))

    print(f"NbitQ={nbit}  ({2**nbit} levels)  mean rate={rate.mean():.4f}  "
          f"median={np.median(rate):.4f} bit/s/Hz")

    plt.figure(figsize=(7, 5))
    plt.scatter(Kf_val, rate, s=6, alpha=0.3, color="seagreen")
    plt.xlabel("K_factor of the channel"); plt.ylabel("Achieved rate [bit/s/Hz]")
    plt.title(f"Rate vs K_factor -- NbitQ={nbit} ({2**nbit} phase levels)")
    plt.grid(True, linestyle="--"); plt.tight_layout()
    plt.savefig(f"{outdir}/rate_vs_kfactor_bit{nbit}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(rate, bins=50, color="steelblue")
    plt.xlabel("Rate [bit/s/Hz]"); plt.ylabel("Count")
    plt.title(f"Rate distribution -- NbitQ={nbit} ({2**nbit} phase levels)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/rate_distribution_bit{nbit}.png", dpi=150)
    plt.close()

    savemat(f"{outdir}/results_bit{nbit}.mat", {
        "H": Hc_val.numpy(), "K_factor": Kf_val,
        "theta_predicted": thetas_q, "rate": rate, "val_idx": val_idx,
        "NbitQ": nbit, "num_levels": 2 ** nbit,
    })

    summary.append({"nbit": nbit, "levels": 2 ** nbit, "mean": rate.mean(),
                     "median": np.median(rate), "bins": bin_means})

# ----------------------------- Comparison plot -----------------------------
plt.figure(figsize=(7, 5))
means = [s["mean"] for s in summary]
plt.plot(BIT_DEPTHS, means, marker="o", linewidth=2, color="darkred")
for nb, m in zip(BIT_DEPTHS, means):
    plt.annotate(f"{m:.2f}", (nb, m), textcoords="offset points", xytext=(0, 8), ha="center")
plt.xlabel("Number of quantization bits (NbitQ)")
plt.ylabel("Mean achieved rate [bit/s/Hz]")
plt.title("Effect of phase-shifter quantization resolution on achieved rate")
plt.xticks(BIT_DEPTHS)
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig("comparison_bits.png", dpi=150)

savemat("comparison_bits_summary.mat", {
    "bit_depths": BIT_DEPTHS,
    "mean_rates": means,
    "median_rates": [s["median"] for s in summary],
})

print("\nSummary:")
for s in summary:
    print(f"  {s['nbit']} bits ({s['levels']} levels): mean={s['mean']:.4f} bit/s/Hz")
print("\nDone.")
