# Transformer-based Control of Stacked Intelligent Metasurfaces (SIM)

ICS Master Project #2, Universite de Rennes — Supervision: Prof. Rafik Zayani.

## Overview

A Stacked Intelligent Metasurface (SIM) is a cascade of programmable phase-shifting
layers placed between a base station and its users, used to shape the propagation
channel directly in the analog domain. This project develops a Transformer-based
controller that predicts the per-layer phase-shift configuration directly from the
channel state, using self-attention to capture interactions across SIM layers,
subject to realistic hardware constraints such as finite-resolution phase shifters.

The controller is evaluated end-to-end: from a single user to multiple interfering
users, across varying channel statistics, and across phase-quantization resolutions.

## Repository structure

```
SIM_NN/
├── SIM_NN_Phase.py, SIM_NN_Phase_bis.py   Original SIM physics model and
│                                           classical (non-learned) optimizer
├── Transformer1_K=1/                      Stage 1 — first working single-user (K=1)
│                                           Transformer controller: dataset generation,
│                                           two training approaches, validated weights
│                                           and results
├── Transformer2_K1_multiKfactor/          Stage 2 — single-user generalization across
│                                           Rician K-factor and user distance, plus a
│                                           1-4 bit phase-quantization comparison
│                                           (bit1/ … bit4/ subfolders; bit=2 is the
│                                           project's standard operating resolution)
└── Transformer2_K=4/                      Stage 3 — multi-user (K=4) controller with
                                            inter-user interference, quantization-aware
                                            training, max-min fairness, and jointly
                                            learned per-user power allocation, evaluated
                                            across the same 1-4 bit resolutions

Meeting_Bussgang/                          Reference material on the Bussgang theorem
                                            and PA-nonlinearity modeling (prior work,
                                            relevant to the upcoming hardware-impairment
                                            stage)
```

Each `Transformer*` folder contains its own `README.md` with methodology, results, and
reproduction steps.

## Environment

```bash
pip install torch numpy scipy matplotlib
```
Python 3.11; CPU-only PyTorch is sufficient, no GPU required. Every stage's trained
weights and results are already included in the repository — see each chapter's
`README.md` for the exact commands to retrain or reproduce them.

## Development history

Each stage below was developed on its own branch and merged into `main` once validated,
so `main` always reflects the latest complete state of the project.

| Stage | Branch | Summary | Status |
|---|---|---|---|
| 0 | `existed_work` | Original SIM physics model and classical optimization baseline | complete |
| 1 | `K1-generalization` | Single-user Transformer controller; generalization across channel K-factor and distance; phase-quantization comparison (1-4 bits) | complete |
| 2 | `multiuser-K4-generalization` | Multi-user (K=4) extension with interference-aware training, fairness, and learned power control; full bit-resolution comparison | complete |
| 3 | — | Power-amplifier nonlinearity compensation | planned |
| 4 | — | Benchmark against a classical/conventional SIM optimizer (performance, robustness, computational complexity) | planned |

## Key results

- Single-user controller matches the classical optimizer's achievable rate while
  running in a single forward pass.
- Generalizes across Rician K-factor without retraining; rate degradation with user
  distance follows the expected path-loss trend.
- Multi-user controller maintains near-equal per-user rates despite a ~40x path-loss
  imbalance between users, through a max-min fairness training objective combined
  with jointly learned power allocation.
- Consistent diminishing-returns behavior across phase-shifter resolutions (1-4 bits)
  in both the single-user and multi-user settings.
