"""
End-to-end Transformer training, K=1 (single user).

The previous supervised approach (imitate theta* from the classical
optimizer) turned out to be limited by an unreliable teacher: the same
channel, optimized twice with different random restarts, gives very
different theta and very different achieved rates (verified: rates of
7.87 vs 10.32 bit/s/Hz on the same channel). Imitating a single noisy
run caps what the student can learn.

Fix: train the Transformer directly on the real objective (maximize the
achieved rate), differentiably, as the professor's "second method"
suggested. The classical optimizer's dataset is only used as a source of
channel samples (H) and as a reference for the final comparison -- not as
a training target.
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
    distance_BS_Layer, transmission_vector, pathloss,
)

# ----------------------------- Configuration -----------------------------
SEED = 0
DATASET_FILE = "sim_dataset_K1_L5_N49.mat"
VAL_FRACTION = 0.1
BATCH_SIZE = 128
EPOCHS = 300
PATIENCE = 40
LR = 1e-3
D_MODEL = 128
NUM_LAYERS = 3
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------- Load channels only -----------------------------
d = loadmat(DATASET_FILE)
H = d["H"].astype(np.complex64)
rate_classical = d["rate"][0].astype(np.float32)
meta = d["metadata"][0, 0]

K = int(meta["K"][0, 0]); L = int(meta["L"][0, 0]); N = int(meta["N"][0, 0])
Nx = int(meta["Nx"][0, 0]); Ny = int(meta["Ny"][0, 0]); NbitQ = int(meta["NbitQ"][0, 0])
assert K == 1
M_total = H.shape[0]
Delta_theta = (2 * np.pi) / (2 ** NbitQ)

# ----------------------------- Physics (differentiable, pure torch) -----------------------------
d_UE = 10; PT = 10; PTl = 0.001 * (10 ** (PT / 10)); M_ant = K; H_BS = 10
BSantGain = 5; BSantGainL = 10 ** (BSantGain / 10); budget = M_ant * BSantGainL * PTl
c = 3e8; f = 28e9; lamda = c / f; dx = lamda / 2; dy = lamda / 2
Tsim = 5 * lamda; d_layer = Tsim / L
sigma2k = -104; wp = 0.001 * (10 ** (sigma2k / 10))
d0 = 1; nbar = 3.5; C0 = 10 ** (-40 / 10)

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
    """Differentiable rate computation directly from (cos, sin) phase representation."""
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
    rate = torch.log2(1 + signal / wp)
    return rate.real


def achieved_rate_batch(theta_batch, H_batch):
    """Rate from quantized angles (radians) -- used only for final evaluation."""
    return rate_from_cossin(torch.cos(theta_batch).reshape(theta_batch.shape[0], -1),
                             torch.sin(theta_batch).reshape(theta_batch.shape[0], -1), H_batch)


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
rate_classical_t = torch.tensor(rate_classical, dtype=torch.float32)

X_train, X_val = X[train_idx], X[val_idx]
Hc_train, Hc_val = H_complex_t[train_idx], H_complex_t[val_idx]
rate_classical_val = rate_classical_t[val_idx]

train_loader = DataLoader(TensorDataset(X_train, Hc_train), batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------- Model -----------------------------
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

# ----------------------------- Training (maximize rate directly) -----------------------------
train_losses, val_losses = [], []
best_val, best_state, patience_cnt = float("inf"), None, 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss, nb = 0.0, 0
    for xb, hb in train_loader:
        optimizer.zero_grad()
        cos_pred, sin_pred = model(xb)
        rate = rate_from_cossin(cos_pred, sin_pred, hb)
        loss = -rate.mean()  # maximize rate == minimize negative rate
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

# ----------------------------- Evaluate with quantization -----------------------------
model.eval()
with torch.no_grad():
    cos_pred, sin_pred = model(X_val)
    phases_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
    phases_np = phases_pred.numpy()
    thetas_q = np.array(
        [[B_codebook[np.argmin(np.abs(phases_np[s, p] - B_codebook))] for p in range(L * N)]
         for s in range(phases_np.shape[0])],
        dtype=np.float32,
    )
    rate_tf = achieved_rate_batch(torch.tensor(thetas_q), Hc_val).numpy()

rate_cl = rate_classical_val.numpy()
corr = np.corrcoef(rate_cl, rate_tf)[0, 1]
print(f"\nValidation set ({len(rate_tf)} samples):")
print(f"  Classical optimizer  mean={rate_cl.mean():.4f}  median={np.median(rate_cl):.4f} bit/s/Hz")
print(f"  Transformer (end-to-end) mean={rate_tf.mean():.4f}  median={np.median(rate_tf):.4f} bit/s/Hz")
print(f"  Per-channel correlation (classical vs transformer rate): {corr:.4f}")

# anti-collapse check
pairwise_corr = np.corrcoef(phases_np[0], phases_np[1])[0, 1]
print(f"\nAnti-collapse check: correlation between sample 0 and 1 predictions: {pairwise_corr:.4f}")
if pairwise_corr > 0.95:
    print("  WARNING: predictions nearly identical across samples.")
else:
    print("  OK: predictions vary across channels.")

# ----------------------------- Plots -----------------------------
plt.figure(figsize=(7, 4))
plt.plot([-l for l in train_losses], label="Train rate")
plt.plot([-l for l in val_losses], label="Validation rate")
plt.xlabel("Epoch"); plt.ylabel("Rate [bit/s/Hz]")
plt.title("End-to-end Transformer training (K=1, direct rate maximization)")
plt.legend(); plt.grid(True, linestyle="--"); plt.tight_layout()
plt.savefig("transformer_K1_e2e_training_curve.png", dpi=150)

se_cl, se_tf = sorted(rate_cl), sorted(rate_tf)
cdf = np.arange(1, len(se_cl) + 1) / len(se_cl)
plt.figure(figsize=(7, 5))
plt.plot(se_cl, cdf, label="Classical optimizer", linewidth=2)
plt.plot(se_tf, cdf, label="Transformer (end-to-end)", linewidth=2, linestyle="--")
plt.xlabel("Rate [bit/s/Hz]"); plt.ylabel("CDF")
plt.title("Transformer (end-to-end) vs Classical Optimizer (K=1)")
plt.legend(); plt.grid(True, linestyle="--"); plt.tight_layout()
plt.savefig("transformer_K1_e2e_cdf.png", dpi=150)

plt.figure(figsize=(6, 6))
plt.scatter(rate_cl, rate_tf, s=10, alpha=0.4, color="seagreen")
lims = [0, max(rate_cl.max(), rate_tf.max()) + 0.5]
plt.plot(lims, lims, "r--", linewidth=1.5, label="y = x")
plt.xlabel("Classical optimizer rate [bit/s/Hz]")
plt.ylabel("Transformer (end-to-end) rate [bit/s/Hz]")
plt.title(f"Per-channel comparison (end-to-end) -- correlation = {corr:.3f}")
plt.legend(); plt.grid(True, linestyle="--"); plt.axis("equal"); plt.tight_layout()
plt.savefig("transformer_K1_e2e_scatter.png", dpi=150)

torch.save(model.state_dict(), "transformer_K1_e2e_weights.pt")
savemat("transformer_K1_e2e_eval.mat", {
    "rate_classical_val": rate_cl, "rate_transformer_val": rate_tf,
    "val_idx": val_idx, "correlation": corr,
})
print("\nSaved: transformer_K1_e2e_training_curve.png, transformer_K1_e2e_cdf.png, "
      "transformer_K1_e2e_scatter.png, transformer_K1_e2e_weights.pt, transformer_K1_e2e_eval.mat")
