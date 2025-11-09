# © 2025 Intelligent Quantum Inc.
# MIT License (see LICENSE).
#
# Simplified demo from *Post-Quantum Superintelligence* (World Scientific, 2026).
#
# This script compares vanilla gradient descent with a post-quantum
# implicit step on a 2D ill-conditioned quadratic. The code is intentionally
# light on derivations; the theory and √κ acceleration analysis are given
# in the book.

import numpy as np


# ---------------------------------------------------------------------
# Problem setup: 2D ill-conditioned quadratic
#   L(theta) = 1/2 * theta^T H theta,  H = diag(1, 100)
# ---------------------------------------------------------------------

H = np.diag([1.0, 100.0])
lam_min, lam_max = 1.0, 100.0
kappa = lam_max / lam_min

def L(theta: np.ndarray) -> float:
    """Quadratic loss."""
    return 0.5 * theta.T @ H @ theta

def grad_L(theta: np.ndarray) -> np.ndarray:
    """Gradient of the loss."""
    return H @ theta


# ---------------------------------------------------------------------
# 1. Gradient Descent with a tuned constant step size
# ---------------------------------------------------------------------

eta = 2.0 / (lam_min + lam_max)  # classical tuned step size

def gradient_descent(theta0, tol=1e-8, max_iters=10_000):
    theta = theta0.astype(float).copy()
    for k in range(max_iters):
        theta -= eta * grad_L(theta)
        if np.linalg.norm(theta) < tol:
            return theta, k + 1
    return theta, max_iters


# ---------------------------------------------------------------------
# 2. Post-quantum implicit step
#
# This uses an implicit midpoint-style update specialized to this
# 2D quadratic. The choice of parameters and the √κ acceleration
# analysis are explained in the book.
# ---------------------------------------------------------------------

tau = 2.0 / np.sqrt(lam_min * lam_max)

A = np.eye(2) + 0.5 * tau * H
B = np.eye(2) - 0.5 * tau * H
R = B @ np.linalg.inv(A)  # linear update operator for this quadratic

def pq_step_method(theta0, tol=1e-8, max_iters=10_000):
    theta = theta0.astype(float).copy()
    for k in range(max_iters):
        theta = R @ theta
        if np.linalg.norm(theta) < tol:
            return theta, k + 1
    return theta, max_iters


# ---------------------------------------------------------------------
# 3. Run both methods from the same initial point and compare
# ---------------------------------------------------------------------

if __name__ == "__main__":
    theta0 = np.array([1.0, 1.0])

    gd_theta, gd_iters = gradient_descent(theta0)
    pq_theta, pq_iters = pq_step_method(theta0)

    print("=== sqrt(kappa) acceleration demo ===")
    print(f"Condition number kappa        : {kappa:.1f}")
    print(f"Initial point theta0          : {theta0}")
    print()
    print(f"Gradient Descent iterations   : {gd_iters}")
    print(f"Post-quantum implicit steps   : {pq_iters}")
    if pq_iters > 0:
        print(f"Empirical speedup (GD / PQ)   : {gd_iters / pq_iters:.2f}x")
    print()
    print(f"Final GD   theta, L(theta)    : {gd_theta}, {L(gd_theta):.3e}")
    print(f"Final PQ   theta, L(theta)    : {pq_theta}, {L(pq_theta):.3e}")
    print()
    print("For the full derivation and interpretation of this update,")
    print("see *Post-Quantum Superintelligence* (World Scientific, 2026).")
