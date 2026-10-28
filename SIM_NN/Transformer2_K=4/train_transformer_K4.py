"""
End-to-end Transformer training, K=4 (multi-user).

Extends the validated K=1 end-to-end (Method 2) approach to the multi-user case:
the SIM now serves 4 users simultaneously, so the achieved metric becomes the
sum rate (sum of per-user Shannon rates), and each user's rate depends on the
others through inter-user interference (same SINR formula used throughout the
project in functions.gradient_func / functions.objective_func, and in
SIM_NN_Phase.py's vectorized custom_loss).

Dataset: fixed K_factor=1, fixed user positions (channels_K4_N49.mat, 8000
samples) -- same single-channel-type starting point as the original K=1 stage,
before the K_factor/distance generalization work. Generalizing the K=4 case
across channel type and distance is a natural next step once this baseline
is validated.
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
    distance_BS_Layer, transmission_vector, correlation_matrix,
)

# ----------------------------- Configuration -----------------------------
SEED = 0
DATASET_FILE = "channels_K4_N49.mat"
VAL_FRACTION = 0.1
BATCH_SIZE = 256
EPOCHS = 300
PATIENCE = 40
LR = 1e-3
D_MODEL, NUM_LAYERS, DROPOUT, WEIGHT_DECAY = 128, 3, 0.1, 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------- Load channels -----------------------------
d = loadmat(DATASET_FILE)
H = d["H"].astype(np.complex64)             # (M, K_users, N)
Betak = d["Betak"].astype(np.float32)       # (K_users, 1)
meta = d["metadata"][0, 0]
K_users = int(meta["K"][0, 0]); N = int(meta["N"][0, 0])
Nx = int(meta["Nx"][0, 0]); Ny = int(meta["Ny"][0, 0])
M_total = H.shape[0]
L = 5
NbitQ = 2
Delta_theta = (2 * np.pi) / (2 ** NbitQ)

# ----------------------------- Physics (differentiable) -----------------------------
PT = 10; PTl = 0.001 * (10 ** (PT / 10)); M_ant = K_users
BSantGain = 5; BSantGainL = 10 ** (BSantGain / 10); budget = M_ant * BSantGainL * PTl
Pue = budget / K_users                        # equal power allocation per user
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
W1_t = torch.tensor(W1, dtype=torch.complex64)   # (N, K_users)
B_codebook = np.array([i * Delta_theta for i in range(2 ** NbitQ)], dtype=np.float32)
B_codebook_t = torch.tensor(B_codebook)


class _STEQuant(torch.autograd.Function):
    """Straight-through quantization to the phase codebook: forward rounds to
    the nearest codebook level, backward passes the gradient through
    unchanged -- same idea as SIM_NN_Phase.py's fake_quant, needed so the
    model learns phases that are robust to the DEPLOYED bit resolution
    instead of only to the continuous relaxation."""
    @staticmethod
    def forward(ctx, theta):
        diffs = torch.abs(theta.unsqueeze(-1) - B_codebook_t.view(1, 1, -1))
        idx = torch.argmin(diffs, dim=-1)
        return B_codebook_t[idx]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_ste(theta):
    return _STEQuant.apply(theta)


SOFTMIN_ALPHA = 0.3


def softmin_objective(rates):
    """Max-min fairness surrogate (same as SIM_NN_Phase.py's custom_loss):
    smooth approximation of min_k R_k via log-sum-exp, so every user gets
    gradient signal instead of the optimizer dedicating the whole power
    budget to a single (strongest) user -- pure sum-rate maximization under
    severe near-far imbalance was found to monopolize one user 100% of the
    time (diagnosed separately), which is not an acceptable multi-user
    solution even though it is not a training bug."""
    return -SOFTMIN_ALPHA * torch.logsumexp(-rates / SOFTMIN_ALPHA, dim=-1)


def sumrate_from_cossin(cos_t, sin_t, H_batch):
    """H_batch: (B, K_users, N) complex. Returns (B,) sum rate and (B, K_users) per-user rate."""
    Bn = cos_t.shape[0]
    phi = torch.complex(cos_t, sin_t).reshape(Bn, L, N)
    G = torch.diag_embed(phi[:, 0, :])
    for l in range(1, L):
        Dl_ = torch.diag_embed(phi[:, l, :])
        Wl_l = Wl_t[l].unsqueeze(0).expand(Bn, -1, -1)
        G = torch.bmm(torch.bmm(Dl_, Wl_l), G)

    Ht = torch.conj(H_batch)                                      # (B, K_users, N)
    W1_exp = W1_t.unsqueeze(0).expand(Bn, -1, -1)                  # (B, N, K_users)
    GW = torch.bmm(G, W1_exp)                                      # (B, N, K_users)
    HGW = torch.bmm(Ht, GW)                                        # (B, K_users, K_users)

    power = Pue * torch.abs(HGW) ** 2                              # (B, K_users, K_users)
    signal = torch.diagonal(power, dim1=-2, dim2=-1)                # (B, K_users)
    total = power.sum(dim=-1)                                       # (B, K_users)
    interf = total - signal + wp
    rates = torch.log2(1 + signal / interf).real                    # (B, K_users)
    return rates.sum(dim=-1), rates


def achieved_sumrate_batch(theta_batch, H_batch):
    sumrate, rates = sumrate_from_cossin(torch.cos(theta_batch), torch.sin(theta_batch), H_batch)
    return sumrate, rates


# ----------------------------- Data -----------------------------
idx = np.random.permutation(M_total)
n_val = int(M_total * VAL_FRACTION)
val_idx, train_idx = idx[:n_val], idx[n_val:]

H_flat = H.reshape(M_total, -1)                       # (M, K_users*N)
H_ri = np.concatenate([H_flat.real, H_flat.imag], axis=1)   # (M, 2*K_users*N)
H_mean = H_ri[train_idx].mean(axis=0, keepdims=True)
H_std = H_ri[train_idx].std(axis=0, keepdims=True) + 1e-8
H_norm = (H_ri - H_mean) / H_std

X = torch.tensor(H_norm, dtype=torch.float32)
H_complex_t = torch.tensor(H, dtype=torch.complex64)   # (M, K_users, N)

X_train, X_val = X[train_idx], X[val_idx]
Hc_train, Hc_val = H_complex_t[train_idx], H_complex_t[val_idx]

train_loader = DataLoader(TensorDataset(X_train, Hc_train), batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------- Model -----------------------------
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


model = SIMTransformerK4(K_users, N, L, d_model=D_MODEL, num_layers=NUM_LAYERS, dropout=DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters")
print(f"Users: K_users={K_users}, meta-atoms per layer: N={N}, layers: L={L}")
print(f"Training set: {len(train_idx)} samples")
print(f"Validation set: {len(val_idx)} samples")

# ----------------------------- Training -----------------------------
train_losses, val_losses = [], []
best_val, best_state, patience_cnt = float("inf"), None, 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss, nb = 0.0, 0
    for xb, hb in train_loader:
        optimizer.zero_grad()
        cos_pred, sin_pred = model(xb)
        theta_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
        theta_q = quantize_ste(theta_pred.reshape(-1, L * N)).reshape(theta_pred.shape)
        _, rates = sumrate_from_cossin(torch.cos(theta_q), torch.sin(theta_q), hb)
        loss = -softmin_objective(rates).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        nb += 1
    scheduler.step()
    epoch_loss /= nb

    model.eval()
    with torch.no_grad():
        cos_pred, sin_pred = model(X_val)
        theta_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
        theta_q = quantize_ste(theta_pred)
        _, rates_val = sumrate_from_cossin(torch.cos(theta_q), torch.sin(theta_q), Hc_val)
        vloss = -softmin_objective(rates_val).mean().item()

    train_losses.append(epoch_loss)
    val_losses.append(vloss)

    if vloss < best_val:
        best_val = vloss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  train_softmin={-epoch_loss:.4f}  "
              f"val_softmin={-vloss:.4f}  best_val_softmin={-best_val:.4f}")

    if patience_cnt >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
print(f"\nBest val soft-min (2-bit, quantization-aware, fairness training): {-best_val:.4f} bit/s/Hz")

# ----------------------------- Evaluate with quantization -----------------------------
model.eval()
with torch.no_grad():
    cos_pred, sin_pred = model(X_val)
    phases_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
phases_np = phases_pred.numpy()
diffs = np.abs(phases_np[:, :, None] - B_codebook[None, None, :])
thetas_q = B_codebook[np.argmin(diffs, axis=2)].astype(np.float32)
sumrate_tf, rates_tf = achieved_sumrate_batch(torch.tensor(thetas_q), Hc_val)
sumrate_tf = sumrate_tf.numpy()
rates_tf = rates_tf.numpy()

print(f"\nValidation set ({len(sumrate_tf)} samples): mean SUM rate = {sumrate_tf.mean():.4f}  "
      f"median = {np.median(sumrate_tf):.4f} bit/s/Hz")
print("\nPer-user mean rate:")
for k in range(K_users):
    print(f"  User {k+1} (Betak={Betak[k,0]:.3e}): mean rate = {rates_tf[:, k].mean():.4f} bit/s/Hz")

# ----------------------------- Plots -----------------------------
plt.figure(figsize=(7, 4))
plt.plot([-l for l in train_losses], label="Train sum rate")
plt.plot([-l for l in val_losses], label="Validation sum rate")
plt.xlabel("Epoch"); plt.ylabel("Sum rate [bit/s/Hz]")
plt.title("Training -- K=4 multi-user end-to-end")
plt.legend(); plt.grid(True, linestyle="--"); plt.tight_layout()
plt.savefig("training_curve_K4.png", dpi=150)

plt.figure(figsize=(7, 4))
plt.hist(sumrate_tf, bins=50, color="steelblue")
plt.xlabel("Sum rate [bit/s/Hz]"); plt.ylabel("Count")
plt.title("Sum rate distribution -- K=4 (validation)")
plt.tight_layout()
plt.savefig("sumrate_distribution_K4.png", dpi=150)

plt.figure(figsize=(7, 4))
plt.boxplot([rates_tf[:, k] for k in range(K_users)], labels=[f"User {k+1}" for k in range(K_users)])
plt.ylabel("Rate [bit/s/Hz]")
plt.title("Per-user rate distribution -- K=4 (validation)")
plt.grid(True, linestyle="--", axis="y"); plt.tight_layout()
plt.savefig("peruser_rate_K4.png", dpi=150)

# ----------------------------- Save -----------------------------
torch.save(model.state_dict(), "transformer_K4_weights.pt")
savemat("transformer_K4_results.mat", {
    "H": Hc_val.numpy(), "theta_predicted": thetas_q,
    "sum_rate": sumrate_tf, "per_user_rate": rates_tf,
    "Betak": Betak, "val_idx": val_idx,
})
print("\nSaved: transformer_K4_weights.pt, transformer_K4_results.mat, "
      "training_curve_K4.png, sumrate_distribution_K4.png, peruser_rate_K4.png")
