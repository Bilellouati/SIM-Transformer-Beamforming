"""
Diagnostic for the K=4 end-to-end Transformer: check quantization sensitivity
and whether the model collapsed to a near-constant output (same failure mode
seen earlier in the K=1 stage).
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from scipy.io import loadmat

from functions import (
    distance_dnn, distance_layers, transmission_matrix,
    distance_BS_Layer, transmission_vector,
)
import torch.nn as nn

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

d = loadmat("channels_K4_N49.mat")
H = d["H"].astype(np.complex64)
Betak = d["Betak"].astype(np.float32)
meta = d["metadata"][0, 0]
K_users = int(meta["K"][0, 0]); N = int(meta["N"][0, 0])
Nx = int(meta["Nx"][0, 0]); Ny = int(meta["Ny"][0, 0])
M_total = H.shape[0]
L = 5
D_MODEL, NUM_LAYERS, DROPOUT = 128, 3, 0.1

PT = 10; PTl = 0.001 * (10 ** (PT / 10)); M_ant = K_users
BSantGain = 5; BSantGainL = 10 ** (BSantGain / 10); budget = M_ant * BSantGainL * PTl
Pue = budget / K_users
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


def sumrate_from_cossin(cos_t, sin_t, H_batch):
    Bn = cos_t.shape[0]
    phi = torch.complex(cos_t, sin_t).reshape(Bn, L, N)
    G = torch.diag_embed(phi[:, 0, :])
    for l in range(1, L):
        Dl_ = torch.diag_embed(phi[:, l, :])
        Wl_l = Wl_t[l].unsqueeze(0).expand(Bn, -1, -1)
        G = torch.bmm(torch.bmm(Dl_, Wl_l), G)
    Ht = torch.conj(H_batch)
    W1_exp = W1_t.unsqueeze(0).expand(Bn, -1, -1)
    GW = torch.bmm(G, W1_exp)
    HGW = torch.bmm(Ht, GW)
    power = Pue * torch.abs(HGW) ** 2
    signal = torch.diagonal(power, dim1=-2, dim2=-1)
    total = power.sum(dim=-1)
    interf = total - signal + wp
    rates = torch.log2(1 + signal / interf).real
    return rates.sum(dim=-1), rates


class SIMTransformerK4(nn.Module):
    def __init__(self, K_users, N, L, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.N, self.L = N, L
        self.input_proj = nn.Linear(2 * K_users * N, d_model)
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


idx = np.random.permutation(M_total)
n_val = int(M_total * 0.1)
val_idx, train_idx = idx[:n_val], idx[n_val:]
H_flat = H.reshape(M_total, -1)
H_ri = np.concatenate([H_flat.real, H_flat.imag], axis=1)
H_mean = H_ri[train_idx].mean(axis=0, keepdims=True)
H_std = H_ri[train_idx].std(axis=0, keepdims=True) + 1e-8
H_norm = (H_ri - H_mean) / H_std
X = torch.tensor(H_norm, dtype=torch.float32)
H_complex_t = torch.tensor(H, dtype=torch.complex64)
X_val = X[val_idx]
Hc_val = H_complex_t[val_idx]

model = SIMTransformerK4(K_users, N, L, d_model=D_MODEL, num_layers=NUM_LAYERS, dropout=DROPOUT)
model.load_state_dict(torch.load("transformer_K4_weights.pt"))
model.eval()

with torch.no_grad():
    cos_pred, sin_pred = model(X_val)
    phases = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)

phases_np = phases.numpy()
print("=== Diversity check (is the model collapsed?) ===")
print("Std of predicted phase across samples (mean over L*N outputs):", phases_np.std(axis=0).mean())
print("Correlation between sample 0 and sample 1 phase vectors:",
      np.corrcoef(phases_np[0], phases_np[1])[0, 1])
print("Correlation between sample 0 and sample 50 phase vectors:",
      np.corrcoef(phases_np[0], phases_np[50])[0, 1])

print("\n=== Quantization sensitivity (continuous vs 1/2/3/4 bits) ===")
for NbitQ in [None, 1, 2, 3, 4]:
    with torch.no_grad():
        if NbitQ is None:
            sumrate, rates = sumrate_from_cossin(cos_pred, sin_pred, Hc_val)
            label = "continuous"
        else:
            Delta_theta = (2 * np.pi) / (2 ** NbitQ)
            B_codebook = np.array([i * Delta_theta for i in range(2 ** NbitQ)], dtype=np.float32)
            diffs = np.abs(phases_np[:, :, None] - B_codebook[None, None, :])
            thetas_q = B_codebook[np.argmin(diffs, axis=2)].astype(np.float32)
            theta_t = torch.tensor(thetas_q)
            sumrate, rates = sumrate_from_cossin(torch.cos(theta_t), torch.sin(theta_t), Hc_val)
            label = f"{NbitQ}-bit"
    print(f"  {label:12s}: mean sum rate = {sumrate.mean().item():.4f}  "
          f"per-user mean = {rates.mean(dim=0).numpy()}")
