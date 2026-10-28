"""
End-to-end Transformer training, K=1, generalized across channel types
(random K_factor per sample, uniform in [0,10]).

Same Method 2 (direct rate maximization) validated for the single-K_factor
K=1 stage, now trained on a dataset spanning pure Rayleigh to near-LoS
channels, so the model learns a mapping that works across varying channel
conditions rather than one fixed statistical type.

Saves a results .mat file containing: H (all channels used), K_factor per
sample, the predicted theta (SIM phase shifts) from this method, and the
achieved rate -- for full traceability.
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
from torch.utils.data import TensorDataset, DataLoader

from functions import (
    distance_dnn, distance_layers, transmission_matrix,
    distance_BS_Layer, transmission_vector,
)

# ----------------------------- Configuration -----------------------------
SEED = 0
DATASET_FILE = "channels_K1_multiKfactor_N49.mat"
VAL_FRACTION = 0.1
BATCH_SIZE = 256
EPOCHS = 300
PATIENCE = 40
LR = 1e-3
D_MODEL = 128
NUM_LAYERS = 3
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------- Load channels -----------------------------
d = loadmat(DATASET_FILE)
H = d["H"].astype(np.complex64)
K_factor = d["K_factor"][0].astype(np.float32)
meta = d["metadata"][0, 0]
K = int(meta["K"][0, 0]); N = int(meta["N"][0, 0])
Nx = int(meta["Nx"][0, 0]); Ny = int(meta["Ny"][0, 0])
assert K == 1
M_total = H.shape[0]

L = 5          # SIM layers (same physical config as the K=1 single-type stage)
NbitQ = 2
Delta_theta = (2 * np.pi) / (2 ** NbitQ)

# ----------------------------- Physics (differentiable) -----------------------------
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
B_codebook = np.array([i * Delta_theta for i in range(2 ** NbitQ)], dtype=np.float32)


def rate_from_cossin(cos_t, sin_t, H_batch):
    Bn = cos_t.shape[0]
    phi = torch.complex(cos_t, sin_t).reshape(Bn, L, N)
    G = torch.diag_embed(phi[:, 0, :])
    for l in range(1, L):
        Dl_ = torch.diag_embed(phi[:, l, :])
        Wl_l = Wl_t[l].unsqueeze(0).expand(Bn, -1, -1)
        G = torch.bmm(torch.bmm(Dl_, Wl_l), G)
    Hc = torch.conj(H_batch).reshape(Bn, 1, N)
    GW = torch.bmm(G, W1_t.unsqueeze(0).expand(Bn, -1, -1))
    HGW = torch.bmm(Hc, GW).reshape(Bn)
    signal = budget * torch.abs(HGW) ** 2
    return torch.log2(1 + signal / wp).real


def achieved_rate_batch(theta_batch, H_batch):
    return rate_from_cossin(torch.cos(theta_batch), torch.sin(theta_batch), H_batch)


# ----------------------------- Data -----------------------------
idx = np.random.permutation(M_total)
n_val = int(M_total * VAL_FRACTION)
val_idx, train_idx = idx[:n_val], idx[n_val:]

H_ri = np.concatenate([H.real, H.imag], axis=1)
H_mean = H_ri[train_idx].mean(axis=0, keepdims=True)
H_std = H_ri[train_idx].std(axis=0, keepdims=True) + 1e-8
H_norm = (H_ri - H_mean) / H_std

X = torch.tensor(H_norm, dtype=torch.float32)
H_complex_t = torch.tensor(H, dtype=torch.complex64)

X_train, X_val = X[train_idx], X[val_idx]
Hc_train, Hc_val = H_complex_t[train_idx], H_complex_t[val_idx]
Kf_val = K_factor[val_idx]

train_loader = DataLoader(TensorDataset(X_train, Hc_train), batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------- Model (same architecture as K=1 single-type) -----------------------------
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
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 2 * L * N),
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
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters")
print(f"Training set: {len(train_idx)} samples, K_factor range "
      f"[{K_factor[train_idx].min():.2f}, {K_factor[train_idx].max():.2f}]")
print(f"Validation set: {len(val_idx)} samples, K_factor range "
      f"[{Kf_val.min():.2f}, {Kf_val.max():.2f}]")

# ----------------------------- Training -----------------------------
train_losses, val_losses = [], []
best_val, best_state, patience_cnt = float("inf"), None, 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss, nb = 0.0, 0
    for xb, hb in train_loader:
        optimizer.zero_grad()
        cos_pred, sin_pred = model(xb)
        rate = rate_from_cossin(cos_pred, sin_pred, hb)
        loss = -rate.mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        nb += 1
    scheduler.step()
    epoch_loss /= nb

    model.eval()
    with torch.no_grad():
        cos_pred, sin_pred = model(X_val)
        vloss = -rate_from_cossin(cos_pred, sin_pred, Hc_val).mean().item()

    train_losses.append(epoch_loss)
    val_losses.append(vloss)

    if vloss < best_val:
        best_val = vloss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  train_rate={-epoch_loss:.4f}  "
              f"val_rate={-vloss:.4f}  best_val_rate={-best_val:.4f}")

    if patience_cnt >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
print(f"\nBest val rate (continuous, unquantized): {-best_val:.4f} bit/s/Hz")

# ----------------------------- Evaluate with quantization, ALL validation samples -----------------------------
model.eval()
with torch.no_grad():
    cos_pred, sin_pred = model(X_val)
    phases_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
phases_np = phases_pred.numpy()

# vectorized quantization to nearest codeword
diffs = np.abs(phases_np[:, :, None] - B_codebook[None, None, :])
thetas_q = B_codebook[np.argmin(diffs, axis=2)].astype(np.float32)

rate_tf = achieved_rate_batch(torch.tensor(thetas_q), Hc_val).numpy()

print(f"\nValidation set ({len(rate_tf)} samples, K_factor in "
      f"[{Kf_val.min():.2f},{Kf_val.max():.2f}]):")
print(f"  Mean rate: {rate_tf.mean():.4f}  Median: {np.median(rate_tf):.4f} bit/s/Hz")

# Rate broken down by K_factor range (to see if generalization is uniform across types)
bins = [0, 2, 4, 6, 8, 10]
for i in range(len(bins) - 1):
    mask = (Kf_val >= bins[i]) & (Kf_val < bins[i + 1])
    if mask.sum() > 0:
        print(f"  K_factor in [{bins[i]},{bins[i+1]}): n={mask.sum():4d}  "
              f"mean rate = {rate_tf[mask].mean():.4f} bit/s/Hz")

# ----------------------------- Plots -----------------------------
plt.figure(figsize=(7, 4))
plt.plot([-l for l in train_losses], label="Train rate")
plt.plot([-l for l in val_losses], label="Validation rate")
plt.xlabel("Epoch"); plt.ylabel("Rate [bit/s/Hz]")
plt.title("Transformer training -- multi-K_factor generalization (K=1)")
plt.legend(); plt.grid(True, linestyle="--"); plt.tight_layout()
plt.savefig("training_curve_multiKfactor.png", dpi=150)

plt.figure(figsize=(7, 5))
plt.scatter(Kf_val, rate_tf, s=6, alpha=0.3, color="seagreen")
plt.xlabel("K_factor of the channel")
plt.ylabel("Achieved rate [bit/s/Hz]")
plt.title("Rate vs K_factor -- does performance hold across channel types?")
plt.grid(True, linestyle="--"); plt.tight_layout()
plt.savefig("rate_vs_kfactor.png", dpi=150)

plt.figure(figsize=(7, 4))
plt.hist(rate_tf, bins=50, color="steelblue")
plt.xlabel("Rate [bit/s/Hz]"); plt.ylabel("Count")
plt.title("Distribution of achieved rate across all channel types (validation)")
plt.tight_layout()
plt.savefig("rate_distribution_multiKfactor.png", dpi=150)

# ----------------------------- Save everything -----------------------------
torch.save(model.state_dict(), "transformer_multiKfactor_weights.pt")
savemat("transformer_multiKfactor_results.mat", {
    "H": Hc_val.numpy(),               # all validation channels used
    "K_factor": Kf_val,                # their K_factor (channel type)
    "theta_predicted": thetas_q,       # SIM phases predicted by Method 2
    "rate": rate_tf,                   # achieved rate per channel
    "val_idx": val_idx,
})
print("\nSaved: transformer_multiKfactor_weights.pt, transformer_multiKfactor_results.mat, "
      "training_curve_multiKfactor.png, rate_vs_kfactor.png, rate_distribution_multiKfactor.png")
