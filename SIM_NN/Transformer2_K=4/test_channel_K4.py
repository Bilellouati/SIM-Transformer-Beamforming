"""
Run the trained K=4 joint phase+power controller on a given channel and report
the resulting phases, power allocation, and achieved rates.

Usage:
    # Use one sample from the project's own validation set (demo / sanity check)
    python test_channel_K4.py --bits 2 --index 0

    # Use your own channel, stored in a .mat file with a variable "H" of shape
    # (K_users, N) or (M, K_users, N) complex (K_users=4, N=49 for this model)
    python test_channel_K4.py --bits 2 --mat my_channel.mat

The model expects the SAME channel normalization statistics (mean/std) used
during training. These are not stored in the weights file, so they are
recomputed here deterministically from channels_K4_N49.mat with the same
random seed used for training -- this reproduces the exact training/validation
split and normalization every time.
"""
import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat, savemat

from functions import (
    distance_dnn, distance_layers, transmission_matrix,
    distance_BS_Layer, transmission_vector,
)

# ----------------------------- Must match training exactly -----------------------------
SEED = 0
DATASET_FILE = os.path.join(os.path.dirname(__file__), "channels_K4_N49.mat")
VAL_FRACTION = 0.1
D_MODEL, NUM_LAYERS, DROPOUT = 128, 3, 0.1
L = 5

torch.manual_seed(SEED)
np.random.seed(SEED)

d = loadmat(DATASET_FILE)
H_all = d["H"].astype(np.complex64)
Betak = d["Betak"].astype(np.float32)
meta = d["metadata"][0, 0]
K_users = int(meta["K"][0, 0]); N = int(meta["N"][0, 0])
Nx = int(meta["Nx"][0, 0]); Ny = int(meta["Ny"][0, 0])
M_total = H_all.shape[0]

idx = np.random.permutation(M_total)
n_val = int(M_total * VAL_FRACTION)
val_idx, train_idx = idx[:n_val], idx[n_val:]

H_flat = H_all.reshape(M_total, -1)
H_ri = np.concatenate([H_flat.real, H_flat.imag], axis=1)
H_mean = H_ri[train_idx].mean(axis=0, keepdims=True)
H_std = H_ri[train_idx].std(axis=0, keepdims=True) + 1e-8

PT = 10; PTl = 0.001 * (10 ** (PT / 10)); M_ant = K_users
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


def sumrate_from_theta_power(theta, H_batch, P):
    """theta: (B, L, N) real. H_batch: (B, K_users, N) complex. P: (B, K_users)."""
    Bn = theta.shape[0]
    phi = torch.complex(torch.cos(theta), torch.sin(theta))
    G = torch.diag_embed(phi[:, 0, :])
    for l in range(1, L):
        Dl_ = torch.diag_embed(phi[:, l, :])
        Wl_l = Wl_t[l].unsqueeze(0).expand(Bn, -1, -1)
        G = torch.bmm(torch.bmm(Dl_, Wl_l), G)
    Ht = torch.conj(H_batch)
    W1_exp = W1_t.unsqueeze(0).expand(Bn, -1, -1)
    GW = torch.bmm(G, W1_exp)
    HGW = torch.bmm(Ht, GW)
    power = P.unsqueeze(1) * torch.abs(HGW) ** 2
    signal = torch.diagonal(power, dim1=-2, dim2=-1)
    total = power.sum(dim=-1)
    interf = total - signal + wp
    rates = torch.log2(1 + signal / interf).real
    return rates.sum(dim=-1), rates


class SIMTransformerK4Power(nn.Module):
    def __init__(self, K_users, N, L, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.N, self.L, self.K_users = N, L, K_users
        self.input_proj = nn.Linear(2 * K_users * N, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.phase_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, 2 * L * N),
        )
        self.power_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, K_users),
        )

    def forward(self, x):
        tok = self.input_proj(x).unsqueeze(1)
        tok = self.transformer(tok)
        feat = tok.squeeze(1)
        out = self.phase_head(feat)
        LN = self.L * self.N
        cos_raw, sin_raw = out[:, :LN], out[:, LN:]
        norm = torch.sqrt(cos_raw ** 2 + sin_raw ** 2 + 1e-8)
        power_frac = torch.softmax(self.power_head(feat), dim=-1)
        return cos_raw / norm, sin_raw / norm, power_frac


def load_channel(mat_path, mat_key="H"):
    H = loadmat(mat_path)[mat_key].astype(np.complex64)
    if H.ndim == 2:
        H = H[None, :, :]
    return H


def run(H_test, bits, weights_dir):
    Delta_theta = (2 * np.pi) / (2 ** bits)
    B_codebook = np.array([i * Delta_theta for i in range(2 ** bits)], dtype=np.float32)

    model = SIMTransformerK4Power(K_users, N, L, d_model=D_MODEL, num_layers=NUM_LAYERS, dropout=DROPOUT)
    weights_path = os.path.join(weights_dir, f"transformer_K4_powerctrl_bit{bits}_weights.pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    H_flat = H_test.reshape(H_test.shape[0], -1)
    H_ri = np.concatenate([H_flat.real, H_flat.imag], axis=1)
    H_norm = (H_ri - H_mean) / H_std
    X = torch.tensor(H_norm, dtype=torch.float32)
    Hc = torch.tensor(H_test, dtype=torch.complex64)

    with torch.no_grad():
        cos_pred, sin_pred, power_frac = model(X)
        phases = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
        phases_np = phases.numpy().reshape(H_test.shape[0], L, N)
        diffs = np.abs(phases_np[:, :, :, None] - B_codebook[None, None, None, :])
        theta_q = B_codebook[np.argmin(diffs, axis=3)].astype(np.float32)
        P = (power_frac * budget).numpy()
        sumrate, rates = sumrate_from_theta_power(torch.tensor(theta_q), Hc, torch.tensor(P))

    return theta_q, P, sumrate.numpy(), rates.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--mat", type=str, default=None, help="Path to a .mat file with variable H")
    parser.add_argument("--mat-key", type=str, default="H")
    parser.add_argument("--index", type=int, default=0, help="If --mat is omitted, index into the validation set")
    parser.add_argument("--out", type=str, default=None, help="Optional .mat path to save the results")
    args = parser.parse_args()

    if args.mat:
        H_test = load_channel(args.mat, args.mat_key)
        print(f"Loaded {H_test.shape[0]} channel(s) from {args.mat}")
    else:
        H_test = H_all[val_idx][args.index:args.index + 1]
        print(f"Using validation-set sample #{args.index} (demo)")

    theta_q, P, sumrate, rates = run(H_test, args.bits, os.path.dirname(__file__) or ".")

    for i in range(H_test.shape[0]):
        print(f"\n--- Channel {i} ---")
        print(f"Sum rate: {sumrate[i]:.4f} bit/s/Hz")
        for k in range(K_users):
            print(f"  User {k+1}: rate = {rates[i, k]:.4f} bit/s/Hz, "
                  f"power = {P[i, k]*1000:.2f} mW ({P[i, k] / budget * 100:.1f}%)")

    if args.out:
        savemat(args.out, {"theta_predicted": theta_q, "power_watts": P,
                            "sum_rate": sumrate, "per_user_rate": rates, "H": H_test})
        print(f"\nSaved results to {args.out}")
