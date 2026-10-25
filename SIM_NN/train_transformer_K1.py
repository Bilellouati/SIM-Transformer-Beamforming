"""
Supervised Transformer training, step K=1 (single user).

Trains a Transformer (H -> theta) using the classical-optimizer dataset
(sim_dataset_K1_L5_N49.mat) as the teacher: loss = distance between
predicted and optimal phases (theta*), not the physics rate loss directly.

After training, the predicted phases are evaluated with the actual SINR/rate
formula and compared against the classical optimizer's achieved rate on a
held-out validation set, to measure how close the student gets to the
teacher.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from math import pi
from torch.utils.data import Dataset, DataLoader

from functions import (
    distance_dnn, distance_layers, transmission_matrix,
    distance_BS_Layer, transmission_vector, pathloss,
)

# ----------------------------- Configuration -----------------------------
SEED = 0
DATASET_FILE = "sim_dataset_K1_L5_N49.mat"
VAL_FRACTION = 0.1
BATCH_SIZE = 128
EPOCHS = 500
PATIENCE = 60
LR = 1e-3
D_MODEL = 128
NUM_LAYERS = 3
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------- Load dataset -----------------------------
d = loadmat(DATASET_FILE)
H = d["H"].astype(np.complex64)          # (M, N)
theta_star = d["theta"].astype(np.float32)  # (M, L*N), quantized
rate_classical = d["rate"][0].astype(np.float32)  # (M,)
meta = d["metadata"][0, 0]

K = int(meta["K"][0, 0])
L = int(meta["L"][0, 0])
N = int(meta["N"][0, 0])
Nx = int(meta["Nx"][0, 0])
Ny = int(meta["Ny"][0, 0])
NbitQ = int(meta["NbitQ"][0, 0])
assert K == 1, "This script is for the K=1 stage."

M_total = H.shape[0]
Delta_theta = (2 * np.pi) / (2 ** NbitQ)

# ----------------------------- Rebuild physics (for evaluation only) -----------------------------
d_UE = 10
PT = 10
PTl = 0.001 * (10 ** (PT / 10))
M_ant = K
H_BS = 10
BSantGain = 5
BSantGainL = 10 ** (BSantGain / 10)
budget = M_ant * BSantGainL * PTl
c = 3e8
f = 28e9
lamda = c / f
dx = lamda / 2
dy = lamda / 2
Tsim = 5 * lamda
d_layer = Tsim / L
sigma2k = -104
wp = 0.001 * (10 ** (sigma2k / 10))
d0 = 1
nbar = 3.5
C0 = 10 ** (-40 / 10)

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


def achieved_rate_batch(theta_batch, H_batch):
    """theta_batch: (B, L*N) real radians. H_batch: (B, N) complex. Returns (B,) rate."""
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
    rate = torch.log2(1 + signal / wp)
    return rate.real


# ----------------------------- Prepare train/val tensors -----------------------------
idx = np.random.permutation(M_total)
n_val = int(M_total * VAL_FRACTION)
val_idx, train_idx = idx[:n_val], idx[n_val:]

H_ri = np.concatenate([H.real, H.imag], axis=1)  # (M, 2N)
H_mean = H_ri[train_idx].mean(axis=0, keepdims=True)
H_std = H_ri[train_idx].std(axis=0, keepdims=True) + 1e-8
H_norm = (H_ri - H_mean) / H_std

cos_true = np.cos(theta_star)
sin_true = np.sin(theta_star)

X = torch.tensor(H_norm, dtype=torch.float32)
Y_cos = torch.tensor(cos_true, dtype=torch.float32)
Y_sin = torch.tensor(sin_true, dtype=torch.float32)
H_complex_t = torch.tensor(H, dtype=torch.complex64)
rate_classical_t = torch.tensor(rate_classical, dtype=torch.float32)

X_train, X_val = X[train_idx], X[val_idx]
Ycos_train, Ycos_val = Y_cos[train_idx], Y_cos[val_idx]
Ysin_train, Ysin_val = Y_sin[train_idx], Y_sin[val_idx]
Hc_train = H_complex_t[train_idx]
Hc_val = H_complex_t[val_idx]
rate_classical_val = rate_classical_t[val_idx]

H_mean_t = torch.tensor(H_mean[0], dtype=torch.float32)
H_std_t = torch.tensor(H_std[0], dtype=torch.float32)

# NOTE: a per-sample random global phase-rotation augmentation was tried here
# and removed -- it corrupted the fixed input normalization statistics every
# epoch (mean/std computed on unrotated data, applied to rotated data), which
# prevented the model from learning any real H -> theta mapping and caused it
# to collapse to a channel-independent output. With 8000 real, diverse samples
# available, augmentation isn't needed anyway.
from torch.utils.data import TensorDataset
train_dataset = TensorDataset(X_train, Ycos_train, Ysin_train)


# ----------------------------- Model (K=1 -> single token Transformer) -----------------------------
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
        Bn = x.shape[0]
        tok = self.input_proj(x).unsqueeze(1)  # (B, 1, d_model) -- single token (K=1)
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ----------------------------- Training loop -----------------------------
train_losses, val_losses = [], []
best_val, best_state, patience_cnt = float("inf"), None, 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss, nb = 0.0, 0
    for xb, cb, sb in train_loader:
        optimizer.zero_grad()
        cos_pred, sin_pred = model(xb)
        loss = ((cos_pred - cb) ** 2 + (sin_pred - sb) ** 2).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        nb += 1
    scheduler.step()
    epoch_loss /= nb

    model.eval()
    with torch.no_grad():
        cos_pred, sin_pred = model(X_val)
        vloss = ((cos_pred - Ycos_val) ** 2 + (sin_pred - Ysin_val) ** 2).mean().item()

    train_losses.append(epoch_loss)
    val_losses.append(vloss)

    if vloss < best_val:
        best_val = vloss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  train={epoch_loss:.5f}  val={vloss:.5f}  best={best_val:.5f}")

    if patience_cnt >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
print(f"\nBest val loss (cos/sin MSE): {best_val:.5f}")

# ----------------------------- Evaluate achieved rate on validation set -----------------------------
model.eval()
with torch.no_grad():
    cos_pred, sin_pred = model(X_val)
    phases_pred = torch.atan2(sin_pred, cos_pred) % (2 * np.pi)
    # quantize to the hardware codebook
    phases_np = phases_pred.numpy()
    B_q = B_codebook
    thetas_q = np.array(
        [[B_q[np.argmin(np.abs(phases_np[s, p] - B_q))] for p in range(L * N)]
         for s in range(phases_np.shape[0])],
        dtype=np.float32,
    )
    rate_tf = achieved_rate_batch(torch.tensor(thetas_q), Hc_val).numpy()

rate_cl = rate_classical_val.numpy()
print(f"\nValidation set ({len(rate_tf)} samples):")
print(f"  Classical optimizer  mean={rate_cl.mean():.4f}  median={np.median(rate_cl):.4f} bit/s/Hz")
print(f"  Transformer (student) mean={rate_tf.mean():.4f}  median={np.median(rate_tf):.4f} bit/s/Hz")
print(f"  Mean gap (classical - transformer): {(rate_cl - rate_tf).mean():.4f} bit/s/Hz")

# ----------------------------- Anti-collapse check -----------------------------
# A model that ignores the channel and always predicts the same phases would
# falsely look "good" if the channel dataset lacks real diversity. Verify the
# model's predictions actually vary meaningfully across different channels.
per_sample_phase_std = phases_np.std(axis=0).mean()
pairwise_corr = np.corrcoef(phases_np[0], phases_np[1])[0, 1]
print(f"\nAnti-collapse check:")
print(f"  Std of predicted phase across samples (per element, avg): {per_sample_phase_std:.4f} rad")
print(f"  Correlation between sample 0 and sample 1 predictions: {pairwise_corr:.4f}")
if pairwise_corr > 0.95:
    print("  WARNING: predictions are nearly identical across samples -- model may have "
          "collapsed to a channel-independent output. Results above should NOT be trusted "
          "as evidence of real channel adaptation.")
else:
    print("  OK: predictions vary meaningfully across different channels.")

# ----------------------------- Plots -----------------------------
plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("cos/sin MSE loss")
plt.title("Transformer training (K=1, supervised on classical optimizer)")
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig("transformer_K1_training_loss.png", dpi=150)

se_cl = sorted(rate_cl)
se_tf = sorted(rate_tf)
cdf = np.arange(1, len(se_cl) + 1) / len(se_cl)
plt.figure(figsize=(7, 5))
plt.plot(se_cl, cdf, label="Classical optimizer (teacher)", linewidth=2)
plt.plot(se_tf, cdf, label="Transformer (student)", linewidth=2, linestyle="--")
plt.xlabel("Rate [bit/s/Hz]")
plt.ylabel("CDF")
plt.title("Transformer vs Classical Optimizer (K=1, validation set)")
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig("transformer_K1_vs_classical_cdf.png", dpi=150)

torch.save(model.state_dict(), "transformer_K1_weights.pt")
savemat("transformer_K1_eval.mat", {
    "rate_classical_val": rate_cl, "rate_transformer_val": rate_tf,
    "val_idx": val_idx, "best_val_loss": best_val,
})
print("\nSaved: transformer_K1_training_loss.png, transformer_K1_vs_classical_cdf.png, "
      "transformer_K1_weights.pt, transformer_K1_eval.mat")
