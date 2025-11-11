"""
Chapter 2 — PQOM Cayley vs Gradient Descent demo.
"""

import numpy as np

def a_norm(A, x):
    return float(np.sqrt(x.T @ A @ x))

def gradient_descent(A, b, x0, step, tol=1e-10, max_iters=100000):
    x = x0.copy()
    x_star = np.linalg.solve(A, b)
    e0 = a_norm(A, x0 - x_star)
    target = tol * e0
    for k in range(max_iters):
        grad = A @ x - b
        x = x - step * grad
        err = a_norm(A, x - x_star)
        if err <= target:
            return x, k + 1, err
    return x, max_iters, err

def cayley_midpoint(A, b, x0, h, tol=1e-10, max_iters=100000):
    x_star = np.linalg.solve(A, b)
    u = x0 - x_star
    I = np.eye(A.shape[0])
    M_left = I + 0.5 * h * A
    M_right = I - 0.5 * h * A
    M_left_inv = np.linalg.inv(M_left)
    e0 = a_norm(A, u)
    target = tol * e0
    for k in range(max_iters):
        u = M_left_inv @ (M_right @ u)
        err = a_norm(A, u)
        if err <= target:
            x = x_star + u
            return x, k + 1, err
    x = x_star + u
    return x, max_iters, err

def run_single_demo(kappa=100.0):
    A = np.diag([1.0, kappa])
    b = np.ones(2)
    x0 = np.array([1.0, 1.0])
    step = 1.9 / kappa
    h = 1.0 / np.sqrt(kappa)
    print(f"=== κ = {kappa:.1f} ===")
    x_gd, it_gd, err_gd = gradient_descent(A, b, x0, step)
    x_cy, it_cy, err_cy = cayley_midpoint(A, b, x0, h)
    print(f"GD iters: {it_gd}, Cayley iters: {it_cy}, speedup: {it_gd/it_cy:.2f}x")

if __name__ == "__main__":
    for kappa in [10, 100, 1000]:
        run_single_demo(kappa)
