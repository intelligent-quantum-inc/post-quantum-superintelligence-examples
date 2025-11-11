"""
Chapter 2 — Worked Example: Spectral Flattening.

This script reproduces the 2x2 example from Chapter 2:

    A = diag(1, 4),  b = (1, 1)^T.

It compares the contraction factors of:

  • a best-possible stable explicit gradient step, and
  • the resolvent (I + A)^{-1}

along the eigen-directions of A.
"""

import numpy as np


def main():
    # Matrix and right-hand side from the chapter
    A = np.diag([1.0, 4.0])
    b = np.ones(2)

    # Eigenvalues of A
    lambdas = np.diag(A)
    lam_min = float(lambdas.min())
    lam_max = float(lambdas.max())

    print("=== Chapter 2: Spectral Flattening Example ===")
    print(f"A = diag({lam_min}, {lam_max}),  b = (1, 1)^T\n")

    # -------- Gradient descent: best common step on [lam_min, lam_max] --------
    # For explicit GD on f(x) = 0.5 x^T A x - b^T x with a single step size alpha,
    # the optimal (Chebyshev) choice on [lam_min, lam_max] is:
    #     alpha_opt = 2 / (lam_min + lam_max)
    # giving per-mode factors 1 - alpha_opt * lambda.
    alpha_opt = 2.0 / (lam_min + lam_max)

    gd_factors = 1.0 - alpha_opt * lambdas
    gd_contractions = np.abs(gd_factors)

    # -------- Resolvent: (I + A)^{-1} --------
    # Along an eigen-direction with eigenvalue lambda, the resolvent acts as
    # multiplication by 1 / (1 + lambda).
    resolvent_factors = 1.0 / (1.0 + lambdas)

    print("Eigenvalues of A:")
    print("  lambda_1 =", lam_min)
    print("  lambda_2 =", lam_max)
    print()

    print("Explicit gradient descent (single optimal step size alpha):")
    print(f"  alpha_opt = 2 / (lambda_min + lambda_max) = {alpha_opt:.3f}")
    print(f"  factors 1 - alpha*lambda: {gd_factors[0]:.3f}, {gd_factors[1]:.3f}")
    print(f"  contractions |1 - alpha*lambda|: {gd_contractions[0]:.3f}, "
          f"{gd_contractions[1]:.3f}")
    print(f"  worst-case GD contraction: {gd_contractions.max():.3f}")
    print()

    print("Resolvent (I + A)^(-1):")
    print(f"  factors 1 / (1 + lambda): {resolvent_factors[0]:.3f}, "
          f"{resolvent_factors[1]:.3f}")
    print(f"  worst-case resolvent contraction: {resolvent_factors.max():.3f}")
    print()

    print("Interpretation:")
    print("  • Gradient descent with a single stable step size contracts both "
          "modes by ~0.6.")
    print("  • The resolvent contracts them by 0.5 and 0.2, respectively,")
    print("    which is strictly stronger in the worst case and flattens")
    print("    anisotropy between the slow and fast directions.")
    print()
    print("This numerically matches the worked example in Chapter 2.")


if __name__ == '__main__':
    main()
