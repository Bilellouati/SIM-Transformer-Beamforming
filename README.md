# Transformer-based Control of Stacked Intelligent Metasurfaces (SIM)

ICS Master Project #2, Universite de Rennes — Supervision: Prof. Rafik Zayani.

Learning-based controller for a Stacked Intelligent Metasurface (SIM): a Transformer
predicts the per-layer phase-shift configuration directly from the channel state,
using attention to capture interactions across SIM layers, under realistic hardware
constraints (finite-resolution phase shifters).

## Project timeline (chapters)

Each stage below is a separate git branch, merged into `main` once validated.
`main` always holds the latest, up-to-date state of the whole project.

| # | Branch | What it covers | Status |
|---|---|---|---|
| 0 | `existed_work` | Professor's original SIM code (classical baseline, no learning) | done |
| 1 | `K1-generalization` | Single-user (K=1) Transformer controller: dataset, training, validated against classical baseline; generalization across Rician K-factor and user distance; 1-4 bit phase-quantization comparison (bit=2 = standard resolution) | done |
| 2 | `multiuser-K4-generalization` | Multi-user (K=4) extension: interference modeling, quantization-aware training, max-min fairness objective, jointly learned per-user power control, full 1-4 bit comparison | done |
| 3 | *(next)* | Power-amplifier nonlinearity compensation | not started |
| 4 | *(next)* | Benchmark vs. classical/conventional SIM optimizer (performance, robustness, computational complexity) | not started |

## Repository layout

- `SIM_NN/Transformer2_K1_multiKfactor/` — Chapter 1 code, results, and `README.md`.
- `SIM_NN/Transformer2_K=4/` — Chapter 2 code, results, and `README.md`.
- `SIM_NN/SIM_NN_Phase.py`, `SIM_NN_Phase_bis.py` — original/reference SIM physics and classical-optimization baseline (professor's code).
- `Project_Overview.pdf`, `Project_Roadmap.pdf`, `Dataset_Report.pdf`, `SIM_Code_Problems_Report.pdf` — project documentation.

See each chapter's own `README.md` for detailed results, methodology, and how to reproduce.
