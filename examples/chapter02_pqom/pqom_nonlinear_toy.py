"""
pqom_nonlinear_toy.py — Nonlinear Cayley Step (Educational)
------------------------------------------------------------
This example illustrates a simple nonlinear operator equation A(x)x = b
solved by a post-quantum–style Cayley (implicit midpoint) update.

It is **educational only** and does NOT implement proprietary UPQOM internals.
"""

import numpy as np

def A(x):
    """State-dependent operator."""
    return np.diag([1 + x[0]**2, 4 + x[1]**2])

def J(x):
    """Jacobian of A(x)x with respect to x."""
    return np.diag([1 + 3*x[0]**2, 4 + 3*x[1]**2])

def cayley_step(x, b, tau=0.5):
    """One implicit midpoint update."""
    r = b - A(x) @ x
    Jx = J(x)
    return x + tau * np.linalg.solve(np.eye(2) + 0.5*tau*Jx, r)

def main():
    x = np.zeros(2)
    b = np.ones(2)
    for k in range(20):
        x = cayley_step(x, b)
    print("Final x:", x)
    print("Energy:", x @ A(x) @ x)

if __name__ == "__main__":
    main()
