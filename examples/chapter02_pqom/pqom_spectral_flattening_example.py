import numpy as np

print("=== Chapter 2: Spectral Flattening Example ===")
A = np.diag([1.0, 4.0])
lam = np.diag(A)

alpha_opt = 2.0/(lam.min() + lam.max())
gd_contract = np.abs(1 - alpha_opt*lam)
res_contract = 1.0/(1.0 + lam)

print(f"Eigenvalues(A): {lam}")
print(f"Explicit GD contractions: {np.round(gd_contract, 3)}")
print(f"Resolvent (I+A)^(-1) contractions: {np.round(res_contract, 3)}")
print("Interpretation: resolvent contracts worst mode more (0.5 vs 0.6) and reduces anisotropy.")
