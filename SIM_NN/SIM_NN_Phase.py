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
# Physics-based loss — exact same formula as professor's
# CustomLoss.  Maximises worst-user rate (min_k R_k).
# ============================================================
def custom_loss(y_true, y_pred, K, N, L, btch,
                Wl_t, W1_t, budget, wp, Delta_theta, NbitQ):

    # --- Quantise predicted phases (STE) ---
    ypredQ = fake_quant(y_pred,
                        0.0,
                        float((2 ** NbitQ - 1) * Delta_theta),
                        NbitQ)                           # (btch, L*N)

    # --- Complex phase shifts  e^{j*theta} ---
    phi  = torch.complex(torch.cos(ypredQ),
                         torch.sin(ypredQ))              # (btch, L*N)
    PHIt = phi.reshape(btch, L, N)                      # (btch, L, N)

    # --- Reconstruct complex channel H from y_true ---
    # y_true : (btch, 2*K*N)  =  [real_part | imag_part]
    Hr = y_true[:, :K * N].T.reshape(K, N, btch)        # (K, N, btch)
    Hi = y_true[:, K * N:].T.reshape(K, N, btch)
    Ht = torch.complex(Hr.float(), Hi.float())           # (K, N, btch)

    # Equal power per user
    Pue = budget / K

    min_rates = []

    for i in range(btch):

        # --- Forward propagation G = diag(phi_L) Wl_L ... diag(phi_1) Wl_1 diag(phi_0) ---
        G = torch.diag(PHIt[i, 0, :])                   # (N, N)
        for l in range(1, L):
            G = torch.matmul(
                    torch.diag(PHIt[i, l, :]),
                    torch.matmul(Wl_t[l], G))

        # --- Per-user SINR and rate ---
        RR = []
        for k in range(K):
            hk = Ht[k, :, i].unsqueeze(0)               # (1, N)
            wk = W1_t[:, k].unsqueeze(1)                 # (N, 1)

            # Signal power : P_k * |h_k^T G w_k|^2
            signal = Pue * torch.abs(
                         torch.matmul(hk, torch.matmul(G, wk))
                     ) ** 2

            # Interference from other users + noise
            interf = sum(
                Pue * torch.abs(
                    torch.matmul(hk, torch.matmul(G, W1_t[:, kp].unsqueeze(1)))
                ) ** 2
                for kp in range(K) if kp != k
            ) + wp

            rate = torch.log2(1 + signal / interf)
            RR.append(rate)

        # Worst-user rate for this sample
        min_rates.append(torch.min(torch.stack(RR)))

    # Loss = -mean(min_k R_k) over the batch
    return -torch.mean(torch.stack(min_rates))


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
btch       = 1          # batch size  (single-channel protocol)
Nch        = 1          # number of channel realisations

# ============================================================
# Physical model — same as professor
# ============================================================
Dnn     = distance_dnn(N, Nx, lamda)
Dl      = distance_layers(L, N, d_layer, Dnn)

cosXl   = np.zeros((L, N, N), dtype=float)
for l in range(L):
    for n in range(N):
        for npr in range(N):
            cosXl[l, n, npr] = d_layer / Dl[l, n, npr]

Wl      = transmission_matrix(L, N, Dl, dx, dy, cosXl, lamda)
D1      = distance_BS_Layer(N, M, Nx, Ny, lamda, d_layer)
W1      = transmission_vector(N, M, d_layer, dx, dy, D1, lamda)
Betak   = pathloss(K, H_BS, Tsim, d_UE, C0, d0, nbar)
Rnn     = correlation_matrix(N, Dnn, lamda)

# Pre-convert fixed matrices to PyTorch tensors (done once, outside the loop)
Wl_t = torch.tensor(Wl, dtype=torch.complex64)
W1_t = torch.tensor(W1, dtype=torch.complex64)

# ============================================================
# Dataset — single channel (professor's original protocol)
# ============================================================
H1, _ = generate_channels(K, N, Nch, Betak, Rnn)
H2     = H1

Htrain0  = np.reshape(H1, (K * N, Nch))
Htrain0r = np.real(Htrain0)
Htrain0i = np.imag(Htrain0)
Htrain   = np.concatenate((Htrain0r, Htrain0i), axis=0).T   # (Nch, 2*K*N)

Hval0  = np.reshape(H2, (K * N, Nch))
Hval0r = np.real(Hval0)
Hval0i = np.imag(Hval0)
Hval   = np.concatenate((Hval0r, Hval0i), axis=0).T         # (Nch, 2*K*N)

Htrain_t = torch.tensor(Htrain, dtype=torch.float32)
Hval_t   = torch.tensor(Hval,   dtype=torch.float32)

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
weights_history = {}

for epoch in range(500):
    model.train()
    optimizer.zero_grad()

    y_pred = model(Htrain_t)
    loss   = custom_loss(Htrain_t, y_pred,
                         K, N, L, btch,
                         Wl_t, W1_t, budget, wp,
                         Delta_theta, NbitQ)
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    train_losses.append(loss_val)
    weights_history[epoch] = {k: v.clone().cpu()
                               for k, v in model.state_dict().items()}

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1:3d}/500   loss = {loss_val:.4f}")

# ============================================================
# Restore best epoch weights (same as professor's savemat logic)
# ============================================================
best_epoch = int(np.argmin(train_losses))
model.load_state_dict(weights_history[best_epoch])
print(f"\nBest epoch : {best_epoch + 1}   loss = {train_losses[best_epoch]:.4f}")

savemat(f"./channel_model_L{L}_b{NbitQ}_N{N}.mat",
        {"H": H1, "best_loss": train_losses[best_epoch]})

# ============================================================
# Plot training loss
# ============================================================
plt.figure()
plt.plot(train_losses, label='Train')
plt.title('Model loss')
plt.grid(True, which="both", linestyle='--')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim([-10, 0])
plt.legend(loc='upper right')
plt.show()

# ============================================================
# Evaluation — same as professor's test section
# ============================================================
model.eval()
with torch.no_grad():
    Htest   = np.concatenate((Htrain0r, Htrain0i), axis=0).T
    Htest_t = torch.tensor(Htest, dtype=torch.float32)
    PHItest = model(Htest_t).numpy()                     # (Nch, L*N)

# Quantise to discrete phase levels
B      = np.array([i * Delta_theta for i in range(2 ** NbitQ)], dtype=np.float32)
thetas = np.array([[B[np.argmin(np.abs(PHItest[s, p] - B))]
                    for p in range(L * N)]
                   for s in range(Nch)], dtype=np.float32)  # (Nch, L*N)

# Evaluate rates with quantised phases
Ht   = H1                                                # (K, N, Nch)
PHIt = np.exp(1j * thetas).reshape(Nch, L, N)
Pue  = budget / K
RR   = []

for i in range(Nch):
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
SE = sorted([RR[i][k] for i in range(Nch) for k in range(K)])
plt.figure()
plt.plot(SE, np.arange(K * Nch) / float(K * Nch), color='r', label='Rate')
plt.title('CDF of Spectral Efficiency')
plt.xlabel('SE [bit/s/Hz]')
plt.ylabel('CDF')
plt.legend()
plt.grid()
plt.show()
