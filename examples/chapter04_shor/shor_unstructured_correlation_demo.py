#!/usr/bin/env python3
"""
Chapter 4 — Unstructured Order-Finding Toy Demo

This script illustrates the "unstructured correlation" example from the
chapter: we have a short, purely classical time series with a hidden
period r, and we use a simple correlation + Fourier analysis to reveal it.

Here we use:
    N = 8, hidden period r = 3,
    f(k) = [0, 1, 2, 0, 1, 2, 0, 1] for k = 0, ..., 7.

We:
  1) Build all cyclic shifts of f,
  2) Embed the values via a bounded function g(y) = cos(2π y / M),
  3) Form a correlation-style matrix A_f,
  4) Extract its leading eigenvector,
  5) Examine the Fourier spectrum of that eigenvector to read off r.

This is a small, transparent *classical* demo. The full post–quantum
operator flows (Cayley / resolvent based PQ–Shor–U) live in the private
pqsi-core-private repository.
"""

import numpy as np


def make_unstructured_signal() -> np.ndarray:
    """
    Return the toy signal f(k) with a hidden period r = 3 on N = 8 points:

        f = [0, 1, 2, 0, 1, 2, 0, 1].

    In the chapter, this is the example where a correlation operator A_f
    has leading modes aligned with frequency 1/3.
    """
    return np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=float)


def cyclic_shift_matrix(f: np.ndarray) -> np.ndarray:
    """
    Build an N x N matrix whose k-th row is the cyclic shift of f by k steps.

    Row k is:
        S[k, j] = f[(j + k) mod N].
    """
    N = len(f)
    S = np.empty((N, N), dtype=float)
    for k in range(N):
        S[k] = np.roll(f, k)
    return S


def embed_signal(S: np.ndarray, M: float = 3.0) -> np.ndarray:
    """
    Apply a bounded embedding g to each entry of S:

        g(y) = cos(2π y / M).

    This plays the role of "encoding" values into phases or bounded
    amplitudes, as described in the chapter.
    """
    return np.cos(2 * np.pi * S / M)


def build_correlation_operator(f: np.ndarray) -> np.ndarray:
    """
    Construct A_f = (1/N) * sum_k s_k s_k^T in matrix form:

        A_f = (1/N) * S_g^T S_g,

    where S_g is the matrix of embedded cyclic shifts.

    This is a small, explicit version of the "unstructured correlation
    operator" discussed in the text.
    """
    N = len(f)
    S = cyclic_shift_matrix(f)
    S_g = embed_signal(S, M=3.0)
    A_f = (1.0 / N) * (S_g.T @ S_g)
    return A_f


def main() -> None:
    f = make_unstructured_signal()
    N = len(f)
    r_true = 3

    print("=== Unstructured Order-Finding Toy Demo (Classical) ===")
    print(f"N = {N}, hidden period r_true = {r_true}")
    print("\nSignal f(k):")
    for k, val in enumerate(f):
        print(f"  k = {k}, f(k) = {val:.0f}")
    print()

    # Build the correlation-style operator A_f
    A_f = build_correlation_operator(f)

    # Symmetric (real) eigen-decomposition
    evals, evecs = np.linalg.eigh(A_f)
    idx_max = np.argmax(evals)
    v_dom = evecs[:, idx_max]

    print("Leading eigenvalue of A_f:", f"{evals[idx_max]:.6f}")
    print("Corresponding (real) eigenvector v_dom (first 8 entries):")
    for j, val in enumerate(v_dom):
        print(f"  j = {j}, v_dom[j] ≈ {val:+.4f}")
    print()

    # Examine the discrete Fourier transform of v_dom
    fft_v = np.fft.fft(v_dom)
    mags = np.abs(fft_v)

    # Ignore the DC component at k = 0
    k_peak = int(np.argmax(mags[1:]) + 1)
    freq = k_peak / N

    print("DFT of v_dom:")
    print("  |fft_v[k]| magnitudes (k = 0,...,N-1):")
    for k in range(N):
        print(f"    k = {k}, |fft_v[k]| ≈ {mags[k]:.4f}")
    print()

    print(f"Dominant non-DC frequency index: k_peak = {k_peak}")
    print(f"Estimated frequency: ω̂ ≈ k_peak / N = {k_peak}/{N}")
    print(f"⇒ Estimated period r_hat ≈ N / k_peak ≈ {N / k_peak:.3f}")
    print()

    print("Interpretation:")
    print("  • The toy signal f(k) has a hidden period r_true = 3.")
    print("  • We build a correlation-style operator A_f from cyclic shifts")
    print("    and a bounded embedding g(f).")
    print("  • The leading eigenvector v_dom of A_f encodes the dominant")
    print("    frequency content of the signal.")
    print("  • The DFT of v_dom exhibits its largest non-DC peak near")
    print("      frequency 1/3, revealing the underlying period.")
    print("  • In the book, larger unstructured problems are handled by")
    print("    post–quantum Cayley / resolvent flows in the private PQSI core.")


if __name__ == "__main__":
    main()
