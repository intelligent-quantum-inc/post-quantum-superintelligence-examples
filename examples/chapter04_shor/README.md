# Chapter 4 — From Shor’s Algorithm to Post-Quantum Computation

This folder contains small, classical companion scripts for Chapter 4 of
*Post-Quantum Superintelligence*.

## `shor_structured_dft_demo.py`

Structured, toy Shor-style example for factoring a small composite integer
(e.g. `N = 15` with base `a = 2`):

- Builds the modular sequence `f(k) = a^k mod N`.
- Embeds it as a complex signal `s_k = exp(2π i f(k)/N)`.
- Uses a discrete Fourier transform (DFT) to identify the dominant frequency.
- Interprets that frequency as an estimate of the order `r`.
- Applies the classical Shor post-processing (GCD test) to recover the factors.

This demo is intentionally small and educational. It mirrors the geometric
effect of the quantum Fourier transform (QFT) in a simple classical setting.
Larger-scale post-quantum operator methods for order finding live in the
private `pqsi-core-private` repository.
