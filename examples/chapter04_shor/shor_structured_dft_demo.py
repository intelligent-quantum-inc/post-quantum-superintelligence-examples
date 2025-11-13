#!/usr/bin/env python3
"""
Chapter 4 — Structured Shor-Style Order Finding (Classical DFT Demo)

This script implements the *structured* worked example from Chapter 4:
factoring N = 15 by recovering the multiplicative order of a = 2 using a
classical discrete Fourier transform (DFT).

The idea mirrors the quantum Fourier transform (QFT) logic, but everything
here is purely classical and small-scale:

  1. Generate the modular powers f(k) = a^k mod N for k = 0, ..., K-1.
  2. Embed the residues into a complex signal s_k = exp(2π i f(k) / N).
  3. Compute the DFT of s_k and locate the dominant frequency peak.
  4. Interpret that frequency as ≈ m / r and recover the order r.
  5. Use Shor’s classical post-processing (GCD test) to factor N.

This is a *toy* illustration only; it is not a large-scale factoring engine.
In the book, post-quantum operator methods generalize this geometry using
Cayley/resolvent flows on structured operators.
"""

import argparse
import math
from typing import Tuple

import numpy as np


def multiplicative_order(a: int, N: int) -> int:
    """
    Compute the multiplicative order of a mod N by brute force.
    Used here only as a ground-truth check for small examples.
    """
    if math.gcd(a, N) != 1:
        raise ValueError(f"a={a} is not coprime to N={N}.")
    x = 1
    for r in range(1, 10_000):  # safe upper bound for small N
        x = (x * a) % N
        if x == 1:
            return r
    raise RuntimeError("Order not found within search bound.")


def generate_modular_powers(a: int, N: int, K: int) -> np.ndarray:
    """
    Generate the modular sequence f(k) = a^k mod N for k = 0, ..., K-1.
    """
    f = np.empty(K, dtype=int)
    x = 1
    for k in range(K):
        f[k] = x
        x = (x * a) % N
    return f


def estimate_order_from_dft(f: np.ndarray, N: int) -> Tuple[int, int]:
    """
    Given the modular sequence f(k) (length K), embed it as a complex signal

        s_k = exp(2π i f(k) / N)

    and compute its DFT. We then look for the dominant non-DC frequency index k*
    and interpret it as approximately k* / K ≈ m / r. We return:

        r_hat : estimated order (denominator after simplification)
        k_peak: index of the dominant peak in the DFT magnitude
    """
    K = len(f)
    # Complex embedding carrying the phase information directly
    s = np.exp(2j * math.pi * f / N)

    # Compute DFT
    S = np.fft.fft(s)

    # Ignore the DC component (index 0); look up to Nyquist for simplicity
    mags = np.abs(S)
    half = K // 2
    k_peak = 1 + np.argmax(mags[1:half])  # index of peak, 1 <= k_peak < K/2

    # Frequency in cycles per sample: omega_hat = k_peak / K
    # Interpret omega_hat ≈ m / r via continued fractions.
    omega_hat = k_peak / K

    # For small examples we can just rational_approx with a simple bound.
    # We look for a denominator r up to K.
    best_num = k_peak
    best_den = K
    # Quick continued-fraction-style search: just brute-force denominators
    # and pick the one that minimizes |m/r - omega_hat|.
    min_err = float("inf")
    for r in range(1, K + 1):
        m = round(omega_hat * r)
        approx = m / r
        err = abs(approx - omega_hat)
        if err < min_err and m > 0:
            min_err = err
            best_num, best_den = m, r

    # best_den is our candidate order r_hat (up to small integer factors)
    r_hat = best_den
    return r_hat, k_peak


def shor_postprocessing(a: int, N: int, r_hat: int) -> Tuple[int, int]:
    """
    Shor's classical post-processing step:

    If r_hat is even and x = a^(r_hat/2) is not ±1 mod N, then
        gcd(x-1, N) and gcd(x+1, N)
    give nontrivial factors (or at least one of them does).

    Returns (p, q) where p*q = N, or (1, N) if the attempt fails.
    """
    if r_hat % 2 != 0:
        return 1, N  # fail: odd order candidate

    x = pow(a, r_hat // 2, N)
    if x == 1 or x == N - 1:
        return 1, N  # fail: trivial residue

    p = math.gcd(x - 1, N)
    q = math.gcd(x + 1, N)
    if p * q == N and p not in (1, N) and q not in (1, N):
        return p, q
    return 1, N  # fallback if something degenerate happens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structured Shor-style factoring demo using a classical DFT."
    )
    parser.add_argument(
        "--N", type=int, default=15, help="Composite integer N to factor (default: 15)."
    )
    parser.add_argument(
        "--a",
        type=int,
        default=2,
        help="Base a with gcd(a, N) = 1; we seek the order of a mod N.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=16,
        help="Number of samples K for the modular sequence (default: 16).",
    )
    args = parser.parse_args()

    N = args.N
    a = args.a
    K = args.K

    print("=== Structured Shor-Style Order Finding (Classical DFT Demo) ===")
    print(f"N = {N}, a = {a}, K = {K}")
    if math.gcd(a, N) != 1:
        print(f"Error: gcd(a, N) = gcd({a}, {N}) ≠ 1. Please choose a coprime a.")
        return

    # 1) Generate modular powers
    f = generate_modular_powers(a, N, K)
    print("\nModular sequence f(k) = a^k mod N for k = 0, ..., K-1:")
    for k, val in enumerate(f):
        print(f"  k = {k:2d}, f(k) = {val}")

    # Ground-truth order for validation (small N only)
    try:
        r_true = multiplicative_order(a, N)
        print(f"\nTrue multiplicative order r (by brute force): r_true = {r_true}")
    except ValueError as e:
        print(f"\nWarning: {e}")
        r_true = None

    # 2–3) Embed and compute DFT, estimate order
    r_hat, k_peak = estimate_order_from_dft(f, N)
    print("\nDFT analysis:")
    print(f"  Dominant non-DC frequency index k_peak = {k_peak}")
    print(f"  Estimated frequency ω̂ ≈ k_peak / K = {k_peak}/{K}")
    print(f"  Estimated order r_hat ≈ {r_hat}")

    # 4) Shor-style classical post-processing
    p, q = shor_postprocessing(a, N, r_hat)
    print("\nClassical Shor post-processing:")
    print(f"  Using r_hat = {r_hat}, we compute x = a^(r_hat/2) mod N")
    if p == 1 and q == N:
        print("  This choice of r_hat did NOT yield nontrivial factors.")
    else:
        print(f"  Nontrivial factors found: p = {p}, q = {q}")
        if r_true is not None:
            print(f"  (For comparison: true order r_true = {r_true})")

    print("\nInterpretation:")
    print("  • The modular sequence f(k) is periodic with order r (here r = 4 for N = 15, a = 2).")
    print("  • The complex embedding s_k = exp(2π i f(k)/N) carries this periodicity into phase.")
    print("  • The DFT magnitudes peak near frequency ω ≈ m/r.")
    print("  • Shor’s quantum algorithm uses a quantum Fourier transform (QFT) to")
    print("    achieve the same effect coherently with amplitudes.")
    print("  • In this small classical demo, we see the same geometric story:")
    print("      hidden periodicity → Fourier peak → candidate order → GCD factors.")


if __name__ == "__main__":
    main()
