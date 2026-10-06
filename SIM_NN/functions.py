###### Importing modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.ops.numpy_ops import np_config
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers
from math import log
from tensorflow.keras.callbacks import EarlyStopping
from scipy.io import savemat, loadmat
from math import *
import random


def distance_dnn(N, Nx, lamda):
    """Distances between npr-th meta-atom of the (l-1)-th surf and n-th meta-atom of the (l)-th surf"""
    Dnn = np.zeros((N, N), dtype=float)
    for n in range(0, N):
        for npr in range(0, N):
            Dnn[n, npr] = np.sqrt(np.floor(np.abs(n - npr) / Nx) ** 2 + (np.abs(n - npr) % Nx) ** 2) * (lamda / 2)
    return Dnn


def distance_layers(L, N, d_layer, Dnn):
    """Transmission distance between the npr-th meta-atom of the (l-1)-th surf and n-th meta-atom of the (l)-th surf"""
    Dl = np.zeros((L, N, N), dtype=float)
    for l in range(0, L):
        for n in range(0, N):
            for npr in range(0, N):
                Dl[l, n, npr] = np.sqrt(d_layer ** 2 + Dnn[n, npr] ** 2)
    return Dl


def transmission_matrix(L, N, Dl, dx, dy, cosXl, lamda):
    """Transmission matrix from layer (l-1) to l-th layer"""
    Wl = np.zeros((L, N, N), dtype=complex)
    for l in range(L):
        for n in range(0, N):
            for npr in range(0, N):
                Wl[l, n, npr] = (dx * dy * cosXl[l, n, npr] / Dl[l, n, npr]) * (
                    (1 / (2 * np.pi * Dl[l, n, npr])) - (1j / lamda)
                ) * np.exp(1j * 2 * np.pi * Dl[l, n, npr] / lamda)
    Wl = np.complex64(Wl)
    return Wl


def distance_BS_Layer(N, M, Nx, Ny, lamda, d_BS_layer1):
    D1 = np.zeros((N, M), dtype=float)
    for n in range(0, N):
        for m in range(0, M):
            D1[n, m] = np.sqrt(
                d_BS_layer1 ** 2
                + ((((n - 1) % Nx) - ((Nx - 1) / 2)) * lamda / 2 - (m - ((M + 1) / 2)) * lamda / 2) ** 2
                + ((np.ceil(n / Nx) - ((Ny + 1) / 2)) * (lamda / 2)) ** 2
            )
    return D1


def transmission_vector(N, M, d_BS_layer1, dx, dy, D1, lamda):
    W1 = np.zeros((N, M), dtype=complex)
    for n in range(0, N):
        for m in range(0, M):
            cosXlnm = d_BS_layer1 / D1[n, m]
            W1[n, m] = (dx * dy * cosXlnm / D1[n, m]) * (
                (1 / (2 * np.pi * D1[n, m])) - (1j / lamda)
            ) * np.exp(1j * 2 * np.pi * D1[n, m] / lamda)
    W1 = np.complex64(W1)
    return W1


def setUp(K, due):
    coord = []
    for x in np.arange(-K / 2, K / 2, due):
        for y in np.arange(-K / 2, K / 2, due):
            coord.extend([(x * due, y * due), (-x * due, y * due), (x * due, -y * due), (-x * due, -y * due)])
    return list(set(coord))


def pathloss_new(K, H_BS, Tsim, d_UE, C0, d0, nbar, due):
    """Path loss with random UE placement"""
    UEs = setUp(d_UE, due)
    coord = random.sample(UEs, K)
    dk = np.zeros((K, 1), dtype=float)
    print(coord)
    for k in range(K):
        dk[k] = np.sqrt((H_BS - Tsim) ** 2 + (coord[k][0] ** 2) + (coord[k][1] ** 2))
    Betak = np.zeros((K, 1), dtype=float)
    for k in range(0, K):
        Betak[k] = C0 * (dk[k] / d0) ** (-nbar)
    return Betak


def pathloss(K, H_BS, Tsim, d_UE, C0, d0, nbar):
    """Path loss with fixed UE placement"""
    dk = np.zeros((K, 1), dtype=float)
    for k in range(K):
        dk[k] = np.sqrt((H_BS - Tsim) ** 2 + (d_UE * (k + 1)) ** 2)
    Betak = np.zeros((K, 1), dtype=float)
    for k in range(0, K):
        Betak[k] = C0 * (dk[k] / d0) ** (-nbar)
    return Betak


def correlation_matrix(N, Dnn, lamda):
    """Spatial correlation matrix between meta-atoms of the last layer"""
    Rnn = np.zeros((N, N), dtype=float)
    for n in range(0, N):
        for npr in range(0, N):
            Rnn[n, npr] = np.sinc(2 * Dnn[n, npr] / lamda)
    return Rnn


def generate_channels(K, N, Nch, Betak, Rnn):
    """Generate spatially correlated Rayleigh channels"""
    H = np.zeros((K, N, Nch), dtype=complex)
    Ht = np.zeros((N, K, Nch), dtype=complex)
    for k in range(0, K):
        H[k, :, :] = (1 / np.sqrt(2)) * (np.random.randn(N, Nch) + 1j * np.random.randn(N, Nch)) * np.sqrt(Betak[k])
        Ht[:, k, :] = (1 / np.sqrt(2)) * (np.random.randn(N, Nch) + 1j * np.random.randn(N, Nch)) * np.sqrt(Betak[k])
    for nc in range(0, Nch):
        H[:, :, nc] = H[:, :, nc].dot(Rnn)
        test = (Ht[:, :, nc].T).dot(Rnn)
        Ht[:, :, nc] = test.T
    H = np.complex64(H)
    Ht = np.complex64(Ht)
    return H, Ht


def generate_rice_channels(K, N, Nch, Betak, Rnn, K_factor):
    """Generate Rician channels"""
    H = np.zeros((K, N, Nch), dtype=complex)
    Ht = np.zeros((N, K, Nch), dtype=complex)
    for k in range(0, K):
        H[k, :, :] = (
            np.sqrt(K_factor / (1 + K_factor))
            + np.sqrt(1 / (1 + K_factor)) * (1 / np.sqrt(2)) * (np.random.randn(N, Nch) + 1j * np.random.randn(N, Nch))
        )
        Ht[:, k, :] = (
            np.sqrt(K_factor / (1 + K_factor))
            + np.sqrt(1 / (1 + K_factor)) * (1 / np.sqrt(2)) * (np.random.randn(N, Nch) + 1j * np.random.randn(N, Nch))
        )
    for nc in range(0, Nch):
        H[:, :, nc] = H[:, :, nc].dot(Rnn)
        Ht[:, :, nc] = Ht[:, :, nc].dot(Rnn)
    H = np.complex64(H)
    Ht = np.complex64(Ht)
    return H, Ht


def pilotGeneration(tau_p):
    """Pilots' Generation"""
    phi = np.array([[(np.random.random() + np.random.random() * 1j) for _ in range(tau_p)] for _ in range(tau_p)])
    phi_or, _ = np.linalg.qr(phi, mode='complete')
    phi_or = phi_or * sqrt(tau_p)
    return phi_or


def gradient_func(P, H, W1, K, L, N, wp, theta, Wl):
    """Analytical gradient of sum-rate with respect to phase shifts"""
    dR = np.zeros((L, N), dtype=float)
    U = np.zeros((L, N, N), dtype=complex)
    V = np.zeros((L, N, N), dtype=complex)
    nu = np.zeros((L, N, K, K), dtype=float)
    PHI = np.zeros((L, N, N), dtype=complex)

    for l in range(L):
        PHI[l, :, :] = np.diag(np.exp(np.zeros((1, N)) + 1j * theta[l])[0])
    # Forward propagation G
    G = PHI[0, :, :]
    for l in range(1, L):
        G = np.matmul(np.matmul(PHI[l, :, :], Wl[l, :, :]), G)

    delta = np.array([
        (1 / (sum([P[k, k] * np.abs(np.matmul(np.reshape(np.conjugate(H[k, :, 0]), (1, N)), np.matmul(G, np.reshape(W1[:, kpr], (N, 1))))) ** 2 for kpr in range(K) if kpr != k]) + wp))[0, 0]
        for k in range(K)
    ])

    for l in range(L):
        if l != 0:
            U[l, :, :] = Wl[1, :, :].dot(PHI[0, :, :])
            for ll in range(1, l):
                U[l, :, :] = Wl[ll + 1, :, :].dot(PHI[ll, :, :]).dot(U[l, :, :])
        else:
            U[l, :, :] = np.eye(N)
        if l != L - 1:
            V[l, :, :] = PHI[L - 1, :, :].dot(Wl[L - 1, :, :])
            for ll in range(L - 2, l, -1):
                V[l, :, :] = V[l, :, :].dot(PHI[ll, :, :].dot(Wl[ll, :, :]))
        else:
            V[l, :, :] = np.eye(N)

        for n in range(N):
            for k in range(K):
                for kpr in range(K):
                    nu[l, n, k, kpr] = np.imag(
                        np.exp(-1j * theta[l, n])
                        * np.conjugate(W1[:, kpr].T).dot(
                            (U[l, n].T).dot(
                                np.conjugate(V[l, :, n].T).dot(
                                    (H[k, :, 0].T).dot(np.conjugate(H[k, :, 0]).dot(G.dot(W1[:, kpr])))
                                )
                            )
                        )
                    )
                s1 = sum([P[kk, kk] * nu[l, n, k, kk] for kk in range(K) if kk != k])
                Pk = P[k, k]
                NU0num0 = (Pk * np.abs(np.matmul(np.reshape(np.conjugate(H[k, :, 0]), (1, N)), np.matmul(G, np.reshape(W1[:, k], (N, 1))))) ** 2)[0, 0]
                NU0den0 = (sum([Pk * np.abs(np.matmul(np.reshape(np.conjugate(H[k, :, 0]), (1, N)), np.matmul(G, np.reshape(W1[:, kpr], (N, 1))))) ** 2 for kpr in range(K) if kpr != k]) + wp)[0, 0]
                gammak = np.log2(1 + (NU0num0 / NU0den0))
                dR[l, n] = dR[l, n] + (1 / log(2) * delta[k] * (Pk * nu[l, n, k, k] - gammak * s1))
    return -dR


def objective_func(P, H, theta, W1, N, wp, Wl, B):
    """Compute per-user rates for a given phase configuration"""
    K, N, _ = H.shape
    L, _ = theta.shape
    Rk = np.zeros((K,), dtype=float)
    PHI = np.zeros((L, N, N), dtype=complex)

    for l in range(L):
        for n in range(N):
            indx = np.argmin(abs(theta[l, n] - B))
            theta[l, n] = B[indx]

    for l in range(L):
        PHI[l, :, :] = np.diag(np.exp(np.zeros((1, N)) + 1j * theta[l])[0])
    G = PHI[0, :, :]
    for l in range(1, L):
        G = np.matmul(np.matmul(PHI[l, :, :], Wl[l, :, :]), G)

    for k in range(K):
        Pk = P[k, k]
        NU0num0 = (Pk * np.abs(np.matmul(np.reshape(np.conjugate(H[k, :, 0]), (1, N)), np.matmul(G, np.reshape(W1[:, k], (N, 1))))) ** 2)[0, 0]
        NU0den0 = (sum([Pk * np.abs(np.matmul(np.reshape(np.conjugate(H[k, :, 0]), (1, N)), np.matmul(G, np.reshape(W1[:, kpr], (N, 1))))) ** 2 for kpr in range(K) if kpr != k]) + wp)[0, 0]
        Rk[k] = np.log2(1 + (NU0num0 / NU0den0))
    return Rk
