import numpy as np

print("=== PQOM Cost Gap Demo: Direct Solver vs Learned Solver ===")
n = 50
kappa = 100.0
M_train = 200    # number of training systems for learned surrogate
M_test  = 50     # number of test solves
np.random.seed(0)

# Construct SPD with condition number ~ kappa
vals = np.linspace(1.0, kappa, n)
A = np.diag(vals)
# Simulate CG-like iteration count ~ sqrt(kappa) * log(1/eps)
eps = 1e-6
cg_iters = int(np.sqrt(kappa)*np.log(1/eps))
work_per_solve = n**2 * cg_iters     # dense matvec proxy
work_direct = M_test * work_per_solve

# Learned pipeline: training cost + inference cost
# Lower bounds/proxies: train ~ M_train * n^2, infer ~ M_test * n^2 (per-call)
train_cost   = 1.5 * M_train * n**2
infer_cost   = 0.1 * M_test  * n**2
work_learned = train_cost + infer_cost

print(f"Dimension n             : {n}")
print(f"Target condition number : {kappa}")
print(f"CG-like iters per solve : {cg_iters}")
print(f"Direct solver total work: {work_direct:.3e}")
print(f"Learned pipeline work   : {work_learned:.3e}")
print(f"Work(learned)/Work(direct): {work_learned/work_direct:.2f}x")

print("\\nInterpretation:")
print("  • Direct solver pays per-instance cost only, with spectrally-accelerated scaling.")
print("  • Learned surrogate adds an upfront training surcharge that dominates unless heavily amortized.")
