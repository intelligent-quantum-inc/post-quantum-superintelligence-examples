import numpy as np

print("=== PQOM Cayley vs Gradient Descent ===")
A = np.diag([1.0, 4.0])
b = np.ones(2)
I = np.eye(2)

# Optimal single-step GD on [lambda_min, lambda_max]
lam_min, lam_max = np.min(np.diag(A)), np.max(np.diag(A))
alpha_opt = 2.0/(lam_min + lam_max)
gd_factors = np.abs(1 - alpha_opt*np.diag(A))

# Cayley (implicit midpoint) map with a modest step
h = 0.4
R = np.linalg.solve(I + 0.5*h*A, I - 0.5*h*A)
cayley_eigs = np.linalg.eigvals(R)

print(f"GD:    alpha_opt = {alpha_opt:.3f}, contractions = {gd_factors}")
print(f"Cayley eigenvalues (per-mode contraction): {np.round(cayley_eigs, 6)}")
