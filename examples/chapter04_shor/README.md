# Chapter 4 — From Shor’s Algorithm to Post-Quantum Computation

This folder contains small, self-contained Python demos that illustrate the
examples in Chapter 4 of *Post-Quantum Superintelligence* (World Scientific, 2026).

All scripts here are **purely classical** and safe to run on a laptop.  
They do **not** contain any of the private post-quantum core logic from the
`pqsi-core-private` repository.

---

## 1. Structured order-finding (Shor-style)

**File:** `shor_structured_dft_demo.py`

This script shows how the “structured” version of Shor’s factoring algorithm
can be understood using a classical discrete Fourier transform (DFT).

What it does:

- Fixes a small composite integer, e.g. `N = 15`, and a base `a = 2`.
- Builds the modular sequence
  \[
    f(k) = a^k \bmod N,\quad k = 0,\dots,K-1.
  \]
- Embeds the residues in a complex phase signal
  \[
    s_k = \exp\!\bigl(2\pi i\, f(k) / N\bigr).
  \]
- Computes the DFT of `s_k` and finds the dominant non-DC frequency.
- Uses that frequency to estimate the multiplicative order `r` and then
  performs the classical Shor post-processing step (GCD) to recover factors.

This mirrors the *geometry* of Shor’s algorithm:
hidden periodicity → Fourier peak → candidate order → nontrivial factors.

Run:

```bash
python3 examples/chapter04_shor/shor_structured_dft_demo.py
