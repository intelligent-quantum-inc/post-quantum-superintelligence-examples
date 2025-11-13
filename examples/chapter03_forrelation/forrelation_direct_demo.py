#!/usr/bin/env python3
"""
Chapter 3 — Forrelation (Direct Classical Demo)

This script implements the Forrelation value

    Φ_{f,g} = (1 / N^{3/2}) * sum_{x,y} f(x) (-1)^{x·y} g(y),

using a straightforward O(2^{2n}) double loop over x,y.

It is intended as a small, transparent companion to the
book chapter "Forrelation and the Post–Quantum Turing Machine".
This demo is purely classical and does NOT implement the
post–quantum resolvent / PQ–TM algorithm, which lives in
the private PQSI core library.
"""

import argparse
import numpy as np
from typing import Tuple


def parity_bit(x: int) -> int:
    """
    Return 0 if x has even Hamming weight, 1 if odd.

    We use Python's built-in bit_count() for clarity:
    parity(x) = bit_count(x) mod 2.
    """
    return x.bit_count() & 1


def forrelation_bruteforce(f: np.ndarray, g: np.ndarray) -> float:
    """
    Compute Φ_{f,g} by the definition:

        Φ_{f,g} = (1 / N^{3/2}) * sum_{x,y} f[x] * (-1)^{x·y} * g[y],

    where x·y is the bitwise dot product modulo 2.
    Here we represent x,y as integers in {0, …, N-1}.
    """
    N = len(f)
    assert len(g) == N
    total = 0.0
    for x in range(N):
        for y in range(N):
            # bitwise dot product mod 2 is just parity of (x & y)
            sign = -1.0 if parity_bit(x & y) else 1.0
            total += f[x] * sign * g[y]
    return total / (N ** 1.5)


def make_boolean_functions(n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct three Boolean functions f, g_corr, g_rand : {0,1}^n -> {±1}.

    - f      : random ±1 values
    - g_corr : starts as f, then flips a subset of entries (so it is
               correlated with f in the standard basis)
    - g_rand : fully random, independent of f

    Note:
      Forrelation measures correlation in the Hadamard (Fourier) basis,
      so Φ(f, g_corr) need not be larger than Φ(f, g_rand). These are
      simply two contrasting examples with different structures.
    """
    N = 2 ** n
    rng = np.random.default_rng(seed)

    f = rng.choice([-1, 1], size=N)

    # "Correlated" g: start from f, then flip about 25% of entries
    g_corr = f.copy()
    flip_indices = rng.choice(N, size=max(1, N // 4), replace=False)
    g_corr[flip_indices] *= -1

    # Fully random g: independent of f
    g_rand = rng.choice([-1, 1], size=N)

    return f, g_corr, g_rand


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct classical demo of the Forrelation value Φ_{f,g}."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6,
        help="Number of input bits n (N = 2^n). Use small n (≤10) to keep runtime modest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    n = args.n
    N = 2 ** n
    if n > 10:
        print(f"Warning: n={n} ⇒ N={N} ⇒ O(N^2) = {N*N} operations; this may be slow.")

    print("=== Forrelation Direct Demo (Classical) ===")
    print(f"n = {n}, N = 2^n = {N}")
    print("We build Boolean functions f, g: {0,1}^n → {±1} and")
    print("compute the Forrelation value Φ_{f,g} by its defining sum.\n")

    f, g_corr, g_rand = make_boolean_functions(n=n, seed=args.seed)

    # Compute Φ for a 'correlated' pair (f, g_corr)
    phi_corr = forrelation_bruteforce(f, g_corr)

    # Compute Φ for a random pair (f, g_rand)
    phi_rand = forrelation_bruteforce(f, g_rand)

    print("Case 1: 'correlated' g (starts from f, then flips a subset of entries)")
    print(f"  Φ(f, g_corr) ≈ {phi_corr:.4f}")
    print("  This shows how Φ behaves when g is constructed by perturbing f.\n")

    print("Case 2: independent random g")
    print(f"  Φ(f, g_rand) ≈ {phi_rand:.4f}")
    print("  This provides a baseline Φ value for an independent random pair.\n")

    print("Interpretation:")
    print("  • Φ_{f,g} is defined by the same inner-product formula used in the chapter:")
    print("      Φ_{f,g} = (1 / N^{3/2}) * Σ_{x,y} f(x) (-1)^{x·y} g(y).")
    print("  • This script evaluates the sum directly in O(2^{2n}) time.")
    print("  • In the book, the Post–Quantum Turing Machine (PQ–TM) uses a")
    print("    post–quantum resolvent mechanism (kept in the private PQSI core)")
    print("    to obtain Φ_{f,g} in polynomial time.")


if __name__ == "__main__":
    main()
