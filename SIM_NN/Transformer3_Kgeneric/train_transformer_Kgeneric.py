"""
K-agnostic end-to-end Transformer, K=4 multi-user.

Unlike Transformer2_K=4's SIMTransformerK4Power (hard-wired to exactly 4
users via fixed-size Linear layers), this model tokenizes each user's
channel separately and lets self-attention operate over a variable-length
sequence of user tokens (plus one aggregator token for the phase decision).
A single trained model therefore accepts any K in {1,2,3,4} at inference,
without retraining per K.

Architecture:
- user_proj: Linear(2*N, d_model), applied per user (shared weights) ->
  K tokens of size d_model.
- A learned aggregator token is prepended to the sequence (same role as a
  CLS token): the Transformer encoder attends over [aggregator, user_1, ...,
  user_K] -- a variable-length sequence, which plain self-attention handles
  natively regardless of K.
- phase_head: applied to the aggregator token's output -> 2*L*N values
  (phases are a property of the SIM, not of individual users, so they must
  come from a pooled/aggregated representation, not a per-user one).
- power_head: applied per user token (shared weights) -> one logit each ->
  softmax over the K present users -> power fractions summing to 1.

Training alternates batches of different K (1, 2, 3, 4) within each epoch,
using the same quantization-aware training (STE) and soft-min fairness
objective validated in Transformer2_K=4.
"""
import sys
import os
import argparse
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

# ----------------------------- Configuration -----------------------------
SEED = 0
DATASET_FILE = "channels_Kgeneric_N49.mat"
VAL_FRACTION = 0.1
BATCH_SIZE = 192
EPOCHS = 5000
PATIENCE = 400
LR = 1e-3
D_MODEL, NUM_LAYERS, DROPOUT, WEIGHT_DECAY = 256, 4, 0.1, 1e-4
K_VALUES = [1, 2, 3, 4]
K_SAMPLE_WEIGHTS = np.array([1.0, 2.0, 3.0, 4.0])  # favor harder (higher-K) cases during training
K_SAMPLE_WEIGHTS = K_SAMPLE_WEIGHTS / K_SAMPLE_WEIGHTS.sum()
L = 5

parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int, default=2, choices=[1, 2, 3, 4])
parser.add_argument("--device", type=str, default=None,
                     help="cuda or cpu (default: cuda if available)")
args = parser.parse_args()
NbitQ = args.bits
DEVICE = torch.device(args.device) if args.device else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Training K-agnostic model, NbitQ={NbitQ} ({2**NbitQ} phase levels), device={DEVICE} ===")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------- Load channels (one dataset per K) -----------------------------
d = loadmat(DATASET_FILE)
N = int(d["metadata"][0, 0]["N"][0, 0])
Betak_full = d["Betak_full"].astype(np.float32)

H_by_K, train_idx_by_K, val_idx_by_K = {}, {}, {}
H_mean_by_K, H_std_by_K = {}, {}
for K in K_VALUES:
    H_K = d[f"H_K{K}"].astype(np.complex64)
    M_total = H_K.shape[0]
    idx = np.random.permutation(M_total)
    n_val = int(M_total * VAL_FRACTION)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    H_flat = H_K.reshape(M_total, -1)
    H_ri = np.concatenate([H_flat.real, H_flat.imag], axis=1)
    H_mean = H_ri[train_idx].mean(axis=0, keepdims=True)
    H_std = H_ri[train_idx].std(axis=0, keepdims=True) + 1e-8
    H_by_K[K] = H_K
    train_idx_by_K[K] = train_idx
    val_idx_by_K[K] = val_idx
    H_mean_by_K[K] = H_mean
    H_std_by_K[K] = H_std

# ----------------------------- Physics (differentiable, K-independent) -----------------------------
PT = 10; PTl = 0.001 * (10 ** (PT / 10))
BSantGain = 5; BSantGainL = 10 ** (BSantGain / 10)
c = 3e8; f = 28e9; lamda = c / f; dx = lamda / 2; dy = lamda / 2
Tsim = 5 * lamda; d_layer = Tsim / L
sigma2k = -104; wp = 0.001 * (10 ** (sigma2k / 10))

Dnn = distance_dnn(N, 7, lamda)
Dl = distance_layers(L, N, d_layer, Dnn)
cosXl = np.zeros((L, N, N), dtype=float)
for l in range(L):
    for n in range(N):
        for npr in range(N):
            cosXl[l, n, npr] = d_layer / Dl[l, n, npr]
Wl = transmission_matrix(L, N, Dl, dx, dy, cosXl, lamda)

Delta_theta = (2 * np.pi) / (2 ** NbitQ)
B_codebook = np.array([i * Delta_theta for i in range(2 ** NbitQ)], dtype=np.float32)
B_codebook_t = torch.tensor(B_codebook, device=DEVICE)
Wl_t = torch.tensor(Wl, dtype=torch.complex64, device=DEVICE)

# W1 (BS -> layer-1 transmission vector) depends only on K (number of antennas),
# not on the batch data -- precompute once per K instead of every forward pass.
W1_t_by_K = {}
for K in range(1, 5):
    D1_K = distance_BS_Layer(N, K, 7, 7, lamda, d_layer)
    W1_K = transmission_vector(N, K, d_layer, dx, dy, D1_K, lamda)
    W1_t_by_K[K] = torch.tensor(W1_K, dtype=torch.complex64, device=DEVICE)

SOFTMIN_ALPHA = 0.3


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


def softmin_objective(rates):
    return -SOFTMIN_ALPHA * torch.logsumexp(-rates / SOFTMIN_ALPHA, dim=-1)


def budget_for_K(K):
    """Total power budget scales with the number of active antennas/users,
    same convention as Transformer2_K=4 (budget = M_ant * BSantGainL * PTl,
    M_ant = K_users)."""
    return K * BSantGainL * PTl


def sumrate_from_theta_power(theta, H_batch, P, K):
    """theta: (B, L, N) real. H_batch: (B, K, N) complex. P: (B, K)."""
    Bn = theta.shape[0]
    W1_t = W1_t_by_K[K]

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


# ----------------------------- Model -----------------------------
class SIMTransformerKGeneric(nn.Module):
    """Per-user tokens + one aggregator token, variable-length attention."""
    def __init__(self, N, L, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.N, self.L = N, L
        self.user_proj = nn.Linear(2 * N, d_model)
        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
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
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )

    def forward(self, x_users):
        """x_users: (B, K, 2*N) real (per-user normalized channel, re/im concat)."""
        Bn, K, _ = x_users.shape
        user_tok = self.user_proj(x_users)                       # (B, K, d_model)
        agg = self.agg_token.expand(Bn, -1, -1)                   # (B, 1, d_model)
        seq = torch.cat([agg, user_tok], dim=1)                   # (B, K+1, d_model)
        out = self.transformer(seq)
        agg_out, user_out = out[:, 0, :], out[:, 1:, :]

        phase_raw = self.phase_head(agg_out)
        LN = self.L * self.N
        cos_raw, sin_raw = phase_raw[:, :LN], phase_raw[:, LN:]
        norm = torch.sqrt(cos_raw ** 2 + sin_raw ** 2 + 1e-8)

        power_logits = self.power_head(user_out).squeeze(-1)     # (B, K)
        power_frac = torch.softmax(power_logits, dim=-1)
        return cos_raw / norm, sin_raw / norm, power_frac


def make_batch_X(H_K, idx_batch, K):
    H_flat = H_K[idx_batch].reshape(len(idx_batch), -1)
    H_ri = np.concatenate([H_flat.real, H_flat.imag], axis=1)
    H_norm = (H_ri - H_mean_by_K[K]) / H_std_by_K[K]
    H_norm = H_norm.reshape(len(idx_batch), K, 2 * N)
    return torch.tensor(H_norm, dtype=torch.float32, device=DEVICE)


model = SIMTransformerKGeneric(N, L, d_model=D_MODEL, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters (K-agnostic, per-user tokens + aggregator)")
print(f"K values trained on: {K_VALUES}, meta-atoms per layer: N={N}, layers: L={L}")
for K in K_VALUES:
    print(f"  K={K}: {len(train_idx_by_K[K])} train / {len(val_idx_by_K[K])} val samples")

# ----------------------------- Training -----------------------------
train_losses, val_losses = [], []
best_val, best_state, patience_cnt = float("-inf"), None, 0
n_batches_per_epoch = min(len(train_idx_by_K[K]) for K in K_VALUES) // BATCH_SIZE

for epoch in range(EPOCHS):
    model.train()
    epoch_loss, nb = 0.0, 0
    for b in range(n_batches_per_epoch):
        K = int(np.random.choice(K_VALUES, p=K_SAMPLE_WEIGHTS))  # harder (higher-K) cases sampled more often
        batch_idx = np.random.choice(train_idx_by_K[K], size=BATCH_SIZE, replace=False)
        X = make_batch_X(H_by_K[K], batch_idx, K)
        Hc = torch.tensor(H_by_K[K][batch_idx], dtype=torch.complex64, device=DEVICE)

        optimizer.zero_grad()
        cos_pred, sin_pred, power_frac = model(X)
        theta_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
        theta_q = quantize_ste(theta_pred.reshape(-1, L * N)).reshape(theta_pred.shape[0], L, N)
        P = power_frac * budget_for_K(K)
        _, rates = sumrate_from_theta_power(theta_q, Hc, P, K)
        loss = -softmin_objective(rates).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        nb += 1
    scheduler.step()
    epoch_loss /= nb

    model.eval()
    val_softmin_by_K = {}
    with torch.no_grad():
        for K in K_VALUES:
            X_val = make_batch_X(H_by_K[K], val_idx_by_K[K], K)
            Hc_val = torch.tensor(H_by_K[K][val_idx_by_K[K]], dtype=torch.complex64, device=DEVICE)
            cos_pred, sin_pred, power_frac = model(X_val)
            theta_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
            theta_q = quantize_ste(theta_pred.reshape(-1, L * N)).reshape(theta_pred.shape[0], L, N)
            P_val = power_frac * budget_for_K(K)
            _, rates_val = sumrate_from_theta_power(theta_q, Hc_val, P_val, K)
            val_softmin_by_K[K] = softmin_objective(rates_val).mean().item()  # actual soft-min rate (higher is better)

    # Selection metric: worst-performing K, not the average -- a high K=1 rate must
    # not be allowed to mask a collapsed/unfair K=3 or K=4 (true "no K left behind").
    vscore = min(val_softmin_by_K.values())
    vloss = -vscore

    train_losses.append(epoch_loss)
    val_losses.append(vloss)

    if vscore > best_val:
        best_val = vscore
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
        torch.save(best_state, f"transformer_Kgeneric_bit{NbitQ}_weights.pt.ckpt")
    else:
        patience_cnt += 1

    if (epoch + 1) % 10 == 0:
        per_k_str = "  ".join(f"K{K}={val_softmin_by_K[K]:.3f}" for K in K_VALUES)
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  train_softmin={-epoch_loss:.4f}  "
              f"val worst-K={vscore:.4f}  best_worst-K={best_val:.4f}  [{per_k_str}]")

    if patience_cnt >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
print(f"\nBest worst-case K soft-min rate ({NbitQ}-bit): {best_val:.4f} bit/s/Hz")

# ----------------------------- Evaluate per K -----------------------------
model.eval()
results_by_K = {}
with torch.no_grad():
    for K in K_VALUES:
        X_val = make_batch_X(H_by_K[K], val_idx_by_K[K], K)
        Hc_val = torch.tensor(H_by_K[K][val_idx_by_K[K]], dtype=torch.complex64, device=DEVICE)
        cos_pred, sin_pred, power_frac = model(X_val)
        phases = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
        phases_np = phases.cpu().numpy().reshape(-1, L, N)
        diffs = np.abs(phases_np[:, :, :, None] - B_codebook[None, None, None, :])
        theta_q = B_codebook[np.argmin(diffs, axis=3)].astype(np.float32)
        P_val = (power_frac * budget_for_K(K)).cpu().numpy()
        sumrate, rates = sumrate_from_theta_power(
            torch.tensor(theta_q, device=DEVICE), Hc_val, torch.tensor(P_val, device=DEVICE), K)
        sumrate_np = sumrate.cpu().numpy()
        rates_np = rates.cpu().numpy()
        results_by_K[K] = dict(theta=theta_q, power=P_val, sumrate=sumrate_np, rates=rates_np)
        print(f"\nK={K}  (validation, {len(sumrate_np)} samples): mean sum rate = {sumrate_np.mean():.4f} bit/s/Hz")
        for k in range(K):
            print(f"  User {k+1}: mean rate = {rates_np[:, k].mean():.4f} bit/s/Hz, "
                  f"mean power share = {P_val[:, k].mean() / budget_for_K(K) * 100:.1f}%")

# ----------------------------- Plots -----------------------------
tag = f"bit{NbitQ}"

plt.figure(figsize=(7, 4))
plt.plot([-l for l in train_losses], label="Train soft-min (random K per batch)")
plt.plot([-l for l in val_losses], label="Validation worst-K soft-min rate")
plt.xlabel("Epoch"); plt.ylabel("Soft-min rate [bit/s/Hz]")
plt.title(f"Training -- K-agnostic model, {NbitQ}-bit")
plt.legend(); plt.grid(True, linestyle="--"); plt.tight_layout()
plt.savefig(f"training_curve_Kgeneric_{tag}.png", dpi=150)

plt.figure(figsize=(7, 4))
plt.boxplot([results_by_K[K]["sumrate"] for K in K_VALUES], tick_labels=[f"K={K}" for K in K_VALUES])
plt.ylabel("Sum rate [bit/s/Hz]")
plt.title(f"Sum rate per K -- K-agnostic model, {NbitQ}-bit (validation)")
plt.grid(True, linestyle="--", axis="y"); plt.tight_layout()
plt.savefig(f"sumrate_by_K_{tag}.png", dpi=150)

# ----------------------------- Save -----------------------------
torch.save(model.state_dict(), f"transformer_Kgeneric_{tag}_weights.pt")
save_dict = {"Betak_full": Betak_full, "NbitQ": NbitQ}
for K in K_VALUES:
    save_dict[f"K{K}_theta"] = results_by_K[K]["theta"]
    save_dict[f"K{K}_power"] = results_by_K[K]["power"]
    save_dict[f"K{K}_sumrate"] = results_by_K[K]["sumrate"]
    save_dict[f"K{K}_rates"] = results_by_K[K]["rates"]
    save_dict[f"K{K}_val_idx"] = val_idx_by_K[K]
savemat(f"transformer_Kgeneric_{tag}_results.mat", save_dict)
print(f"\nSaved: transformer_Kgeneric_{tag}_weights.pt, transformer_Kgeneric_{tag}_results.mat, "
      f"training_curve_Kgeneric_{tag}.png, sumrate_by_K_{tag}.png")
