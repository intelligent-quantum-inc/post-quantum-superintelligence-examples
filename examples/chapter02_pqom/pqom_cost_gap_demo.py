"""
Chapter 2 — Cost Gap Demo: Direct Solver vs "Learned" Solver

This toy experiment compares the computational work needed to:

  1. Solve A x = b for many right-hand sides b_j using
     a direct UPQOM-style solver (Conjugate Gradient), versus

  2. "Learn" an approximate inverse W ≈ A^{-1} by gradient descent
     on many training pairs (b_j, x_j^*), where x_j^* = A^{-1} b_j,
     and then use W b to solve new systems.

We measure work in crude "flop units" ~ O(n^2) operations per
matrix–vector product / parameter update.

This is not meant to be a precise energy model; it illustrates
the training overhead of the learned pipeline compared to a
direct spectrally accelerated solver.
"""

import numpy as np


# ------------------------------------------------------------
# Utility: conjugate gradient with crude flop counting
# ------------------------------------------------------------

def cg_solve(A, b, tol=1e-8, max_iters=1000, flop_counter=None):
    """
    Conjugate Gradient for SPD A.

    Returns x, iters_used, flops_estimate.
    flops_estimate is a rough O(n^2) count: each matvec costs ~2 n^2.
    """
    n = A.shape[0]
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rs_old = float(r @ r)
    flops = 0.0

    for k in range(max_iters):
        Ap = A @ p
        # One matvec ~ 2 n^2 flops (multiplies + adds)
        flops += 2.0 * n * n

        alpha = rs_old / float(p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol:
            if flop_counter is not None:
                flop_counter[0] += flops
            return x, k + 1, flops

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    if flop_counter is not None:
        flop_counter[0] += flops
    return x, max_iters, flops


# ------------------------------------------------------------
# Utility: build a random SPD matrix with controlled kappa
# ------------------------------------------------------------

def random_spd(n, kappa_target=100.0, seed=0):
    rng = np.random.default_rng(seed)
    # Random orthogonal-ish matrix
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    # Spectrum between 1 and kappa_target
    lambdas = np.linspace(1.0, kappa_target, n)
    A = Q @ np.diag(lambdas) @ Q.T
    return A


# ------------------------------------------------------------
# Learned solver: W ≈ A^{-1} via SGD on many (b, x*)
# ------------------------------------------------------------

def train_learned_inverse(A, train_bs, lr=1e-2, epochs=20):
    """
    Train a full matrix W so that W b ≈ A^{-1} b on a set of training
    right-hand sides train_bs.

    We measure work in units ~ 2 n^2 per forward + backward update.
    """
    n = A.shape[0]
    # Exact inverse for generating targets (oracle)
    A_inv = np.linalg.inv(A)
    xs = A_inv @ train_bs.T  # shape: (n, m_train)

    # Initialize W near zero
    rng = np.random.default_rng(123)
    W = rng.normal(scale=1e-2, size=(n, n))

    work_units = 0.0
    m_train = train_bs.shape[0]

    for epoch in range(epochs):
        # simple SGD over each training pair
        for j in range(m_train):
            b_j = train_bs[j]           # shape (n,)
            x_j = xs[:, j]              # shape (n,)

            # Forward: y = W b_j
            y = W @ b_j                 # ~ 2 n^2 work
            # Loss L = 0.5 ||y - x_j||^2
            diff = y - x_j

            # Gradient wrt W: dL/dW = diff * b_j^T
            grad_W = np.outer(diff, b_j)  # ~ n^2 work

            # SGD update
            W = W - lr * grad_W

            # Count work units (very coarse)
            work_units += 3.0 * n * n

    return W, work_units


def evaluate_learned_inverse(W, A, test_bs, tol=1e-6):
    """
    Evaluate how well W approximates A^{-1} on test right-hand sides.
    """
    A_inv = np.linalg.inv(A)
    xs_exact = A_inv @ test_bs.T  # (n, m_test)

    errors = []
    for j in range(test_bs.shape[0]):
        b_j = test_bs[j]
        x_exact = xs_exact[:, j]
        x_pred = W @ b_j
        err = np.linalg.norm(x_pred - x_exact) / np.linalg.norm(x_exact)
        errors.append(err)
    return np.array(errors)


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def run_experiment(n=50, kappa=100.0, m_train=200, m_test=50, tol=1e-8):
    rng = np.random.default_rng(42)

    print("=== PQOM Cost Gap Demo: Direct Solver vs Learned Solver ===")
    print(f"Dimension n           : {n}")
    print(f"Condition number target: {kappa}")
    print(f"Train RHS count       : {m_train}")
    print(f"Test RHS count        : {m_test}")
    print()

    # Build SPD matrix
    A = random_spd(n, kappa_target=kappa, seed=0)

    # -----------------------------
    # Direct solver cost (CG)
    # -----------------------------
    total_flops_direct = [0.0]
    test_bs = rng.normal(size=(m_test, n))

    cg_iters = []
    for j in range(m_test):
        b_j = test_bs[j]
        _, iters, flops = cg_solve(A, b_j, tol=tol, flop_counter=total_flops_direct)
        cg_iters.append(iters)

    avg_cg_iters = sum(cg_iters) / len(cg_iters)

    print("Direct solver (Conjugate Gradient):")
    print(f"  Avg iterations per system : {avg_cg_iters:.1f}")
    print(f"  Total work units (flops)  : {total_flops_direct[0]:.3e}")
    print()

    # -----------------------------
    # Learned inverse: training
    # -----------------------------
    train_bs = rng.normal(size=(m_train, n))
    W, train_work = train_learned_inverse(A, train_bs, lr=1e-2, epochs=20)

    # Inference work on test_bs
    n2 = n * n
    infer_work = m_test * 2.0 * n2  # each W @ b is ~ 2 n^2

    total_learned_work = train_work + infer_work

    # Evaluate error
    rel_errors = evaluate_learned_inverse(W, A, test_bs)
    avg_err = float(rel_errors.mean())
    max_err = float(rel_errors.max())

    print("Learned solver (W ≈ A^{-1} via SGD):")
    print(f"  Training work units     : {train_work:.3e}")
    print(f"  Inference work (test)   : {infer_work:.3e}")
    print(f"  Total work units        : {total_learned_work:.3e}")
    print(f"  Avg relative error      : {avg_err:.3e}")
    print(f"  Max relative error      : {max_err:.3e}")
    print()

    # -----------------------------
    # Comparison
    # -----------------------------
    print("Comparison summary:")
    print(f"  Direct solver total work   : {total_flops_direct[0]:.3e}")
    print(f"  Learned pipeline total work: {total_learned_work:.3e}")
    if total_flops_direct[0] > 0:
        ratio = total_learned_work / total_flops_direct[0]
        print(f"  Work(learned) / Work(direct): {ratio:.2f}x")
    print()
    print("Interpretation:")
    print("  • The direct solver pays only per-instance CG cost.")
    print("  • The learned pipeline pays a large, up-front training cost")
    print("    just to approximate the same mapping x = A^{-1} b.")
    print("  • Even when the learned model works reasonably well, its total")
    print("    work (train + inference) typically exceeds the direct solver")
    print("    on this structured SPD family.")
    print("  • This is a concrete numerical illustration of the cost gap")
    print("    described in Chapter 2.")
    print()


if __name__ == "__main__":
    run_experiment()
