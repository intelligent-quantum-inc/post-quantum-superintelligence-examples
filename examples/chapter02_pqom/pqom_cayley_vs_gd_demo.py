import numpy as np

# 2x2 SPD with eigenvalues {1, 4}  -> kappa = 4
A = np.diag([1.0, 4.0])
kappa = np.max(np.diag(A)) / np.min(np.diag(A))
sqrt_k = np.sqrt(kappa)

# --- Gradient Descent (single optimal step) ---
lam = np.diag(A)
alpha_opt = 2.0 / (lam.min() + lam.max())
gd_factors = 1.0 - alpha_opt * lam
gd_contractions = np.abs(gd_factors)  # per-mode |1 - alpha*lambda|

# --- Cayley / implicit midpoint with optimal step h* ---
# h* = 2 / sqrt(lmin * lmax)
h_opt = 2.0 / np.sqrt(lam.min() * lam.max())
cayley_factors = (1.0 - 0.5 * h_opt * lam) / (1.0 + 0.5 * h_opt * lam)
cayley_contractions = np.abs(cayley_factors)

# Theoretical worst-case contraction for midpoint/CG:
rho = (sqrt_k - 1.0) / (sqrt_k + 1.0)

print("=== PQOM Cayley vs Gradient Descent ===")
print(f"Eigenvalues(A): {lam}")
print(f"kappa(A)       : {kappa:.1f},  sqrt(kappa): {sqrt_k:.3f}")
print(f"GD:    alpha_opt = {alpha_opt:.3f}, contractions = {gd_contractions}")
print(f"Cayley: h_opt    = {h_opt:.6f}, per-mode |R_h(λ)| = {cayley_contractions}")
print(f"Worst-case (theory): rho = (sqrt(k)-1)/(sqrt(k)+1) = {rho:.6f}")
