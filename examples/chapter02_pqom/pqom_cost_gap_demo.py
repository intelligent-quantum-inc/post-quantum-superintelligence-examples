#!/usr/bin/env python3
"""
PQOM Cost Gap Demo: Direct UPQOM-style solver vs a learned surrogate.

- Direct solver cost ~ unit * k_iters per solve, with k_iters ≈ c_iter * sqrt(kappa) * log(1/eps).
- Learned surrogate cost ~ epochs * M_train * unit  +  M_test * unit  (lower-bound style proxy).

Energy ~ flops, so “work units” are proportional to flops.
"""

import argparse
import math

def fmt_e(x):  # compact scientific formatting
    return f"{x: .3e}"

def main():
    ap = argparse.ArgumentParser(description="PQOM cost gap demo")
    ap.add_argument("--n",        type=int,   default=50,     help="dimension")
    ap.add_argument("--kappa",    type=float, default=100.0,  help="target condition number")
    ap.add_argument("--eps",      type=float, default=1e-6,   help="accuracy (residual) target")
    ap.add_argument("--epochs",   type=int,   default=200,    help="training epochs (surrogate)")
    ap.add_argument("--m-train",  type=int,   default=200,    help="# training systems")
    ap.add_argument("--m-test",   type=int,   default=50,     help="# test solves")
    ap.add_argument("--dense",    action="store_true",        help="use dense (n^2) proxy")
    ap.add_argument("--sparse",   action="store_true",        help="use sparse (~nnz_c * n) proxy")
    ap.add_argument("--nnz-c",    type=int,   default=10,     help="nnz coefficient for sparse proxy")
    ap.add_argument("--c-iter",   type=float, default=2.0,    help="iteration constant in k ≈ c_iter·√κ·log(1/ε)")
    args = ap.parse_args()

    # --- validation / defaults ------------------------------------------------
    if args.dense and args.sparse:
        print("Both --dense and --sparse set; defaulting to --dense.")
        args.sparse = False
    if not args.dense and not args.sparse:
        args.dense = True  # default

    if args.n <= 0:
        raise ValueError("n must be positive.")
    if args.kappa < 1.0:
        # clamp to 1 (perfectly conditioned) to avoid negative/zero logs of eps scaling weirdness
        args.kappa = 1.0
    if not (0.0 < args.eps < 1.0):
        raise ValueError("eps must be in (0, 1).")
    if args.epochs < 0 or args.m_train < 0 or args.m_test < 0:
        raise ValueError("epochs, m-train, and m-test must be nonnegative.")
    if args.c_iter <= 0:
        raise ValueError("c-iter must be positive.")
    if args.sparse and args.nnz_c <= 0:
        raise ValueError("nnz-c must be positive for sparse mode.")

    # --- iteration count: k ≈ c_iter * sqrt(kappa) * log(1/eps) --------------
    k_iters_real = args.c_iter * math.sqrt(args.kappa) * abs(math.log(args.eps))
    k_iters = max(1, math.ceil(k_iters_real))

    # --- unit cost model ------------------------------------------------------
    if args.dense:
        unit = args.n ** 2
        cost_label = "dense (n^2) proxy"
    else:
        unit = args.nnz_c * args.n
        cost_label = f"sparse (~{args.nnz_c}·n) proxy"

    # --- costs ----------------------------------------------------------------
    # Direct UPQOM-style solver:
    work_per_solve = unit * k_iters
    work_direct = args.m_test * work_per_solve

    # Learned surrogate (lower-bound proxies):
    train_cost = args.epochs * args.m_train * unit
    infer_cost = args.m_test * unit
    work_learned = train_cost + infer_cost

    # Break-even M_test*:
    # work_direct = M* * unit * k_iters  and  learned = train_cost + M* * unit
    # => M* = train_cost / (unit * (k_iters - 1))  (if k_iters>1)
    if k_iters > 1:
        m_break = train_cost / (unit * (k_iters - 1))
    else:
        m_break = float("inf")  # with k_iters=1, direct is extremely cheap per solve

    # --- report ---------------------------------------------------------------
    print("=== PQOM Cost Gap Demo: Direct Solver vs Learned Surrogate ===")
    print(f"n={args.n}, kappa={args.kappa}, eps={args.eps}, epochs={args.epochs}, "
          f"M_train={args.m_train}, M_test={args.m_test}")
    print(f"iteration constant c_iter={args.c_iter:.2f}  |  cost model: {cost_label}")
    print(f"k_iters (≈ c_iter·√κ·log(1/ε)) = {k_iters}   (raw={k_iters_real:.2f})")
    print(f"Per-solve (direct) work ≈ unit * k_iters = {unit} * {k_iters} = {fmt_e(work_per_solve)}")
    print(f"Direct total work  ≈ {fmt_e(work_direct)}")
    print(f"Learned total work ≈ {fmt_e(work_learned)}  "
          f"(train={fmt_e(train_cost)} + infer={fmt_e(infer_cost)})")
    ratio = (work_learned / work_direct) if work_direct > 0 else float('inf')
    print(f"ratio (learned/direct) = {ratio: .2f}x")
    print(f"Break-even M_test* ≈ {m_break:.1f}")

    # Verdict
    if math.isfinite(m_break) and args.m_test < m_break:
        print("Verdict: BELOW break-even → direct UPQOM-style solver is cheaper.")
    elif math.isfinite(m_break):
        print("Verdict: ABOVE break-even → learned surrogate can amortize training.")
    else:
        print("Verdict: Direct per-solve cost is minimal (k≈1); break-even is effectively infinite.")

if __name__ == "__main__":
    main()
