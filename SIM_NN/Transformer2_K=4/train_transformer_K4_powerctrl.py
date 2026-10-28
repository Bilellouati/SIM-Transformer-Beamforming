"""
End-to-end Transformer training, K=4 (multi-user), with JOINT phase +
per-user power control.

Extends train_transformer_K4.py (equal power split, Pue = budget/K, fairness
via soft-min, quantization-aware phases) by also letting the model decide
HOW MUCH power to allocate to each user, under the total power budget
constraint (sum_k P_k = budget), instead of a fixed equal split. This lets
the model compensate a user with a weak phase-based SINR by giving it more
power, rather than being stuck competing for a fair outcome purely through
phase configuration.

Note: classical water-filling does NOT apply here -- it assumes parallel,
non-interfering channels (e.g. OFDM subcarriers / MIMO eigenmodes). Here all
users interfere through the SAME shared SIM beamforming matrix G, so power
allocated to one user directly raises the interference seen by the others.
Water-filling under interference tends to reproduce the single-user
monopolization problem (push power to the strongest, zero to the weakest),
which is exactly what the fairness objective (soft-min) was introduced to
avoid. This script therefore keeps the soft-min fairness objective and adds
power as a second, LEARNED degree of freedom -- jointly optimized with phase,
under the fairness objective -- rather than switching to water-filling.

Correct multi-user SINR with non-equal power: interference seen by user k
from the stream meant for user kpr scales with P_kpr (the power of THAT
stream), not P_k -- this generalizes the project's existing equal-power
formula (functions.gradient_func / objective_func), which only coincides
with it because those originally assumed P_k = P_kpr for all users.
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
H = d["H"].astype(np.complex64)
Betak = d["Betak"].astype(np.float32)
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
B_codebook_t = torch.tensor(B_codebook)


class _STEQuant(torch.autograd.Function):
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
    return -SOFTMIN_ALPHA * torch.logsumexp(-rates / SOFTMIN_ALPHA, dim=-1)


def sumrate_from_cossin_power(cos_t, sin_t, H_batch, P):
    """H_batch: (B, K_users, N) complex. P: (B, K_users), sums to `budget`.
    Interference seen by user k from stream kpr scales with P[:, kpr]
    (the power of that stream), not P[:, k] -- correct multi-user SINR
    under non-equal power allocation."""
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
    HGW = torch.bmm(Ht, GW)                                  # (B, K_users, K_users)

    power = P.unsqueeze(1) * torch.abs(HGW) ** 2             # (B, K_users, K_users), weighted by column (stream) power
    signal = torch.diagonal(power, dim1=-2, dim2=-1)          # (B, K_users)
    total = power.sum(dim=-1)                                 # (B, K_users)
    interf = total - signal + wp
    rates = torch.log2(1 + signal / interf).real
    return rates.sum(dim=-1), rates


def achieved_sumrate_batch(theta_batch, H_batch, P):
    return sumrate_from_cossin_power(torch.cos(theta_batch), torch.sin(theta_batch), H_batch, P)


# ----------------------------- Data -----------------------------
idx = np.random.permutation(M_total)
n_val = int(M_total * VAL_FRACTION)
val_idx, train_idx = idx[:n_val], idx[n_val:]

H_flat = H.reshape(M_total, -1)
H_ri = np.concatenate([H_flat.real, H_flat.imag], axis=1)
H_mean = H_ri[train_idx].mean(axis=0, keepdims=True)
H_std = H_ri[train_idx].std(axis=0, keepdims=True) + 1e-8
H_norm = (H_ri - H_mean) / H_std

X = torch.tensor(H_norm, dtype=torch.float32)
H_complex_t = torch.tensor(H, dtype=torch.complex64)

X_train, X_val = X[train_idx], X[val_idx]
Hc_train, Hc_val = H_complex_t[train_idx], H_complex_t[val_idx]

train_loader = DataLoader(TensorDataset(X_train, Hc_train), batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------- Model -----------------------------
class SIMTransformerK4Power(nn.Module):
    """Same backbone as SIMTransformerK4, with an added power head: softmax
    over K_users scaled by the total budget, so sum_k P_k = budget always."""
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
        power_frac = torch.softmax(self.power_head(feat), dim=-1)   # (B, K_users), sums to 1
        return cos_raw / norm, sin_raw / norm, power_frac


model = SIMTransformerK4Power(K_users, N, L, d_model=D_MODEL, num_layers=NUM_LAYERS, dropout=DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters (joint phase + power control)")
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
        cos_pred, sin_pred, power_frac = model(xb)
        theta_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
        theta_q = quantize_ste(theta_pred.reshape(-1, L * N)).reshape(theta_pred.shape)
        P = power_frac * budget
        _, rates = sumrate_from_cossin_power(torch.cos(theta_q), torch.sin(theta_q), hb, P)
        loss = -softmin_objective(rates).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        nb += 1
    scheduler.step()
    epoch_loss /= nb

    model.eval()
    with torch.no_grad():
        cos_pred, sin_pred, power_frac = model(X_val)
        theta_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
        theta_q = quantize_ste(theta_pred)
        P_val = power_frac * budget
        _, rates_val = sumrate_from_cossin_power(torch.cos(theta_q), torch.sin(theta_q), Hc_val, P_val)
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
print(f"\nBest val soft-min (2-bit, joint phase+power, fairness training): {-best_val:.4f} bit/s/Hz")

# ----------------------------- Evaluate with quantization -----------------------------
model.eval()
with torch.no_grad():
    cos_pred, sin_pred, power_frac = model(X_val)
    phases_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
phases_np = phases_pred.numpy()
diffs = np.abs(phases_np[:, :, None] - B_codebook[None, None, :])
thetas_q = B_codebook[np.argmin(diffs, axis=2)].astype(np.float32)
P_val_np = (power_frac * budget).numpy()
sumrate_tf, rates_tf = achieved_sumrate_batch(torch.tensor(thetas_q), Hc_val, torch.tensor(P_val_np))
sumrate_tf = sumrate_tf.numpy()
rates_tf = rates_tf.numpy()

print(f"\nValidation set ({len(sumrate_tf)} samples): mean SUM rate = {sumrate_tf.mean():.4f}  "
      f"median = {np.median(sumrate_tf):.4f} bit/s/Hz")
print("\nPer-user mean rate and mean power share:")
for k in range(K_users):
    print(f"  User {k+1} (Betak={Betak[k,0]:.3e}): mean rate = {rates_tf[:, k].mean():.4f} bit/s/Hz, "
          f"mean power share = {P_val_np[:, k].mean() / budget * 100:.1f}%")

# ----------------------------- Plots -----------------------------
plt.figure(figsize=(7, 4))
plt.plot([-l for l in train_losses], label="Train soft-min")
plt.plot([-l for l in val_losses], label="Validation soft-min")
plt.xlabel("Epoch"); plt.ylabel("Soft-min rate [bit/s/Hz]")
plt.title("Training -- K=4, joint phase + power control (fairness)")
plt.legend(); plt.grid(True, linestyle="--"); plt.tight_layout()
plt.savefig("training_curve_K4_powerctrl.png", dpi=150)

plt.figure(figsize=(7, 4))
plt.hist(sumrate_tf, bins=50, color="steelblue")
plt.xlabel("Sum rate [bit/s/Hz]"); plt.ylabel("Count")
plt.title("Sum rate distribution -- K=4, joint phase + power (validation)")
plt.tight_layout()
plt.savefig("sumrate_distribution_K4_powerctrl.png", dpi=150)

plt.figure(figsize=(7, 4))
plt.boxplot([rates_tf[:, k] for k in range(K_users)], tick_labels=[f"User {k+1}" for k in range(K_users)])
plt.ylabel("Rate [bit/s/Hz]")
plt.title("Per-user rate distribution -- K=4, joint phase + power (validation)")
plt.grid(True, linestyle="--", axis="y"); plt.tight_layout()
plt.savefig("peruser_rate_K4_powerctrl.png", dpi=150)

plt.figure(figsize=(7, 4))
plt.boxplot([P_val_np[:, k] / budget * 100 for k in range(K_users)], tick_labels=[f"User {k+1}" for k in range(K_users)])
plt.ylabel("Power share [%]")
plt.title("Per-user learned power allocation -- K=4 (validation)")
plt.grid(True, linestyle="--", axis="y"); plt.tight_layout()
plt.savefig("power_allocation_K4.png", dpi=150)

# ----------------------------- Save -----------------------------
torch.save(model.state_dict(), "transformer_K4_powerctrl_weights.pt")
savemat("transformer_K4_powerctrl_results.mat", {
    "H": Hc_val.numpy(), "theta_predicted": thetas_q, "power_watts": P_val_np,
    "sum_rate": sumrate_tf, "per_user_rate": rates_tf,
    "Betak": Betak, "val_idx": val_idx,
})
print("\nSaved: transformer_K4_powerctrl_weights.pt, transformer_K4_powerctrl_results.mat, "
      "training_curve_K4_powerctrl.png, sumrate_distribution_K4_powerctrl.png, "
      "peruser_rate_K4_powerctrl.png, power_allocation_K4.png")
