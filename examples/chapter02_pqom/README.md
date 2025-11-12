# Chapter 2 — Post-Quantum Operator Model (Public Demos)

This folder contains small, reproducible scripts referenced in the Chapter 2 text:

- `pqom_cayley_vs_gd_demo.py` — compares per-mode contraction of Cayley (implicit midpoint) vs. best explicit GD on a 2×2 SPD system.
- `pqom_spectral_flattening_example.py` — shows resolvent `(I+A)^{-1}` contracts worst mode more than best explicit GD.
- `pqom_cost_gap_demo.py` — illustrates the training surcharge of a learned surrogate vs. a direct solver on SPD families.

These are **public** didactic examples. The **private** PQSI core (UPQOM engine, nonlinear `A(x)=b`) lives in the monetized repo.
