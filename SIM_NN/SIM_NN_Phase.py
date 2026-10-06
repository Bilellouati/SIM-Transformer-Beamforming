###### Importing modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from math import pi
from functions import *


# ============================================================
# Phase quantization — Straight-Through Estimator
# Equivalent to TF fake_quant_with_min_max_args
# Forward : round to nearest discrete level
# Backward: gradient passes through unchanged
# ============================================================
class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_val, max_val, num_bits):
        scale = (max_val - min_val) / (2 ** num_bits - 1)
        x_clamped = torch.clamp(x, min_val, max_val)
        x_q = torch.round((x_clamped - min_val) / scale) * scale + min_val
        return x_q

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None


def fake_quant(x, min_val, max_val, num_bits):
    return _STE.apply(x, min_val, max_val, num_bits)


# ============================================================
# CNN Model — same architecture as professor's TF code
# TF: Conv1D(3 filters, kernel 32) → Flatten → Dense(L*N)
#     with sigmoid activation scaled to [0, 2*pi]
# ============================================================
class SIMNet(nn.Module):
    def __init__(self, K, N, L):
        super().__init__()
        # Conv1d(in_channels=1, out_channels=3, kernel_size=32)
        # Input  : (batch, 1, 2*K*N)
        # Output : (batch, 3, 2*K*N - 31)
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=32)
        flat_size = 3 * (2 * K * N - 32 + 1)
        self.fc   = nn.Linear(flat_size, L * N)

    def forward(self, x):
        # x : (batch, 2*K*N)
        x = x.unsqueeze(1)                              # (batch, 1, 2*K*N)
        x = F.relu(self.conv(x))                        # (batch, 3, 2*K*N-31)
        x = x.flatten(1)                                # (batch, flat_size)
        x = 2 * pi * torch.sigmoid(self.fc(x))          # (batch, L*N) in [0, 2*pi]
        return x


# ============================================================
# Physics-based loss — fully vectorized (no Python loops)
# Maximises worst-user rate (min_k R_k) over the batch.
# ============================================================
def custom_loss(y_true, y_pred, K, N, L,
                Wl_t, W1_t, budget, wp, Delta_theta, NbitQ):

    B = y_pred.shape[0]   # actual batch size

    # --- Quantise predicted phases (STE) ---
    ypredQ = fake_quant(y_pred,
                        0.0,
                        float((2 ** NbitQ - 1) * Delta_theta),
                        NbitQ)                           # (B, L*N)

    # --- Complex phase shifts e^{j*theta} : (B, L, N) ---
    phi  = torch.complex(torch.cos(ypredQ), torch.sin(ypredQ))
    PHIt = phi.reshape(B, L, N)

    # --- Build beamforming matrix G for every sample in parallel ---
    # G[i] = diag(phi[i,L-1]) Wl[L-1] ... diag(phi[i,0])
    # Shape: (B, N, N)
    G = torch.diag_embed(PHIt[:, 0, :])                 # (B, N, N)
    for l in range(1, L):
        Dl    = torch.diag_embed(PHIt[:, l, :])         # (B, N, N)
        Wl_l  = Wl_t[l].unsqueeze(0).expand(B, -1, -1) # (B, N, N)
        G     = torch.bmm(torch.bmm(Dl, Wl_l), G)       # (B, N, N)

    # --- Reconstruct complex channel H : (B, K, N) ---
    Hr = y_true[:, :K * N].reshape(B, K, N)
    Hi = y_true[:, K * N:].reshape(B, K, N)
    Ht = torch.complex(Hr, Hi)                          # (B, K, N)

    # --- Compute GW = G @ W1 : (B, N, K) ---
    W1_exp = W1_t.unsqueeze(0).expand(B, -1, -1)        # (B, N, K)
    GW     = torch.bmm(G, W1_exp)                       # (B, N, K)

    # --- h_k^H G w_k for all k : (B, K, K) ---
    # Ht: (B, K, N)  GW: (B, N, K)
    HGW = torch.bmm(Ht, GW)                             # (B, K, K)

    Pue    = budget / K
    power  = Pue * torch.abs(HGW) ** 2                  # (B, K, K)

    # signal[b,k] = power[b,k,k]
    signal = torch.diagonal(power, dim1=-2, dim2=-1)     # (B, K)

    # interf[b,k] = sum_{k'≠k} power[b,k,k'] + noise
    total  = power.sum(dim=-1)                           # (B, K)
    interf = total - signal + wp                         # (B, K)

    # Rate per user, worst-user rate per sample
    rates     = torch.log2(1 + signal / interf)          # (B, K)
    min_rates = rates.min(dim=-1).values                 # (B,)

    return -min_rates.mean()


# ============================================================
# Parameters — identical to professor's code
# ============================================================
K          = 4          # users
d_UE       = 10
PT         = 10         # dBm
PTl        = 0.001 * (10 ** (PT / 10))
M          = K          # BS antennas
H_BS       = 10
BSantGain  = 5          # dBi
BSantGainL = 10 ** (BSantGain / 10)
budget     = M * BSantGainL * PTl
L          = 5          # metasurfaces
Nx         = 7
Ny         = 7
N          = Nx * Ny    # meta-atoms per layer  (49)
NbitQ      = 2
c          = 3e8
f          = 28e9       # Hz
lamda      = c / f
dx         = lamda / 2
dy         = lamda / 2
Tsim       = 5 * lamda
d_layer    = Tsim / L
Delta_theta = (2 * np.pi) / (2 ** NbitQ)
sigma2k    = -104       # dBm
wp         = 0.001 * (10 ** (sigma2k / 10))   # noise power (mW)
d0         = 1
nbar       = 3.5
C0         = 10 ** (-40 / 10)
btch       = 32         # mini-batch size
Nch        = 1000       # number of channel realisations

# ============================================================
# Physical model — vectorized (same formula, no Python loops)
# ============================================================
Dnn   = distance_dnn(N, Nx, lamda)
Dl    = distance_layers(L, N, d_layer, Dnn)
cosXl = d_layer / Dl                                    # (L, N, N)

# transmission_matrix — vectorized
Wl = (dx * dy * cosXl / Dl) * (
    (1 / (2 * np.pi * Dl)) - (1j / lamda)
) * np.exp(1j * 2 * np.pi * Dl / lamda)
Wl = np.complex64(Wl)                                   # (L, N, N)

# transmission_vector — vectorized
D1       = distance_BS_Layer(N, M, Nx, Ny, lamda, d_layer)
cosXlnm  = d_layer / D1                                 # (N, M)
W1       = (dx * dy * cosXlnm / D1) * (
    (1 / (2 * np.pi * D1)) - (1j / lamda)
) * np.exp(1j * 2 * np.pi * D1 / lamda)
W1       = np.complex64(W1)                             # (N, M)

Betak = pathloss(K, H_BS, Tsim, d_UE, C0, d0, nbar)
Rnn   = correlation_matrix(N, Dnn, lamda)

# Pre-convert fixed matrices to PyTorch tensors (done once, outside the loop)
Wl_t = torch.tensor(Wl, dtype=torch.complex64)
W1_t = torch.tensor(W1, dtype=torch.complex64)

# ============================================================
# Dataset — 1000 channels (multi-channel protocol)
# ============================================================
Nch_val  = 200   # separate validation set

H1, _ = generate_channels(K, N, Nch,     Betak, Rnn)   # train
H2, _ = generate_channels(K, N, Nch_val, Betak, Rnn)   # validation

def make_input(H, n):
    H0 = np.reshape(H, (K * N, n))
    return np.concatenate((np.real(H0), np.imag(H0)), axis=0).T  # (n, 2*K*N)

Htrain_raw = make_input(H1, Nch)      # raw channel (used in loss)
Hval_raw   = make_input(H2, Nch_val)

# Normalize network INPUT so gradients are well-scaled
H_mean  = Htrain_raw.mean(axis=0, keepdims=True)
H_std   = Htrain_raw.std(axis=0, keepdims=True) + 1e-8
Htrain_norm = (Htrain_raw - H_mean) / H_std
Hval_norm   = (Hval_raw   - H_mean) / H_std

# Tensors: norm for network input, raw for loss
Htrain_in  = torch.tensor(Htrain_norm, dtype=torch.float32)
Hval_in    = torch.tensor(Hval_norm,   dtype=torch.float32)
Htrain_raw_t = torch.tensor(Htrain_raw, dtype=torch.float32)
Hval_raw_t   = torch.tensor(Hval_raw,   dtype=torch.float32)

from torch.utils.data import DataLoader, TensorDataset
train_loader = DataLoader(TensorDataset(Htrain_in, Htrain_raw_t),
                          batch_size=btch, shuffle=True)

# ============================================================
# Model + optimiser
# ============================================================
model     = SIMNet(K, N, L)
optimizer = torch.optim.Adam(model.parameters())

print(model)
print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ============================================================
# Training — 500 epochs, weights saved at every epoch
# Best epoch selected at the end (same logic as professor's
# LambdaCallback + argmin)
# ============================================================
train_losses   = []
val_losses     = []
weights_history = {}

for epoch in range(500):
    model.train()
    epoch_loss = 0.0
    n_batches  = 0
    for (xb_norm, xb_raw) in train_loader:
        optimizer.zero_grad()
        y_pred = model(xb_norm)               # network sees normalized input
        loss   = custom_loss(xb_raw, y_pred,  # loss uses raw channel values
                             K, N, L,
                             Wl_t, W1_t, budget, wp,
                             Delta_theta, NbitQ)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1
    epoch_loss /= n_batches

    # Validation loss (no gradient)
    model.eval()
    with torch.no_grad():
        vp    = model(Hval_in)
        vloss = custom_loss(Hval_raw_t, vp,
                            K, N, L,
                            Wl_t, W1_t, budget, wp,
                            Delta_theta, NbitQ).item()

    train_losses.append(epoch_loss)
    val_losses.append(vloss)
    weights_history[epoch] = {k: v.clone().cpu()
                               for k, v in model.state_dict().items()}

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1:3d}/500   train={epoch_loss:.4f}   val={vloss:.4f}")

# ============================================================
# Restore best epoch weights (same as professor's savemat logic)
# ============================================================
best_epoch = int(np.argmin(val_losses))
model.load_state_dict(weights_history[best_epoch])
print(f"\nBest epoch : {best_epoch + 1}   val_loss = {val_losses[best_epoch]:.4f}")

savemat(f"./channel_model_L{L}_b{NbitQ}_N{N}.mat",
        {"H": H1, "best_loss": val_losses[best_epoch]})

# ============================================================
# Plot training loss
# ============================================================
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses,   label='Validation')
plt.title('Model loss')
plt.grid(True, which="both", linestyle='--')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim([-10, 0])
plt.legend(loc='upper right')
plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Evaluation on validation set
# ============================================================
model.eval()
with torch.no_grad():
    PHItest = model(Hval_in).numpy()                     # (Nch_val, L*N)

# Quantise to discrete phase levels
B      = np.array([i * Delta_theta for i in range(2 ** NbitQ)], dtype=np.float32)
thetas = np.array([[B[np.argmin(np.abs(PHItest[s, p] - B))]
                    for p in range(L * N)]
                   for s in range(Nch_val)], dtype=np.float32)  # (Nch_val, L*N)

# Evaluate rates with quantised phases (on validation channels H2)
Ht   = H2                                                # (K, N, Nch_val)
PHIt = np.exp(1j * thetas).reshape(Nch_val, L, N)
Pue  = budget / K
RR   = []

for i in range(Nch_val):
    G = np.diag(PHIt[i, 0, :])
    for l in range(1, L):
        G = np.matmul(np.diag(PHIt[i, l, :]), np.matmul(Wl[l], G))

    R = []
    for k in range(K):
        hk     = Ht[k, :, i].reshape(1, N)
        wk     = W1[:, k].reshape(N, 1)
        signal = Pue * np.abs(hk @ G @ wk) ** 2
        interf = sum(Pue * np.abs(hk @ G @ W1[:, kp].reshape(N, 1)) ** 2
                     for kp in range(K) if kp != k) + wp
        R.append(float(np.log2(1 + signal.flat[0] / interf.flat[0])))
    RR.append(R)

# CDF of spectral efficiency
SE = sorted([RR[i][k] for i in range(Nch_val) for k in range(K)])
plt.figure()
plt.plot(SE, np.arange(K * Nch_val) / float(K * Nch_val), color='r', label='CNN (Nch=1000)')
plt.title('CDF of Spectral Efficiency')
plt.xlabel('SE [bit/s/Hz]')
plt.ylabel('CDF')
plt.legend()
plt.grid()
plt.savefig('cdf_SE.png', dpi=150, bbox_inches='tight')
plt.show()
