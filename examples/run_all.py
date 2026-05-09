"""
examples/run_all.py
--------------------
Operator identification on five benchmark PDEs.

For each PDE: simulate → build feature library → recover with OMP + LassoCV.

Usage:  python examples/run_all.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from opid import PDESimulator, FeatureLibrary, OperatorIdentifier

LIB_CFG = dict(poly_degree=3, max_deriv=4, max_cross_degree=4)


def run_one(sim, true_names, true_coefs, n_nonzero, lasso_alpha, title):
    print(f"\n{'#'*60}")
    print(f"#  {title}")
    print(f"{'#'*60}")

    U, U_t = sim.run()
    lib = FeatureLibrary(**LIB_CFG)
    Theta, names = lib.build(U, sim.k)
    y = U_t.ravel()
    print(f"  Library: {len(names)} terms,  data: {Theta.shape[0]} samples")

    # OMP
    r = OperatorIdentifier(method="omp", n_nonzero=n_nonzero).fit(Theta, y, names)
    print(f"  OMP (n_nonzero={n_nonzero}): {r.names}")
    for n, c in zip(r.names, r.active_coef):
        print(f"    {n:>20s} = {c:+.6f}")

    # LassoCV
    r = OperatorIdentifier(method="lasso", alpha=lasso_alpha).fit(Theta, y, names)
    print(f"  LassoCV ({len(r.names)} terms): {r.names[:5]}...")

    print(f"  True model:")
    for n, c in zip(true_names, true_coefs):
        print(f"    {n:>20s} = {c:+.6f}")


# KdV
run_one(
    PDESimulator.kdv(N=256, T=0.05, n_t=100, n_modes=8, seed=42, backend="numpy"),
    ["u u_x", "u_xxx"], [-6.0, -1.0], 2, 0.5,
    "KdV: u_t = -6 u u_x - u_xxx")

# KS
run_one(
    PDESimulator.ks(N=256, T=0.08, n_t=100, n_modes=8, seed=42, backend="numpy"),
    ["u u_x", "u_xx", "u_xxxx"], [-1.0, -1.0, -1.0], 3, 0.2,
    "KS: u_t = -u u_x - u_xx - u_xxxx")

# Burgers
run_one(
    PDESimulator.burgers(nu=0.05, N=256, T=0.5, n_t=100, n_modes=8, seed=42, backend="numpy"),
    ["u u_x", "u_xx"], [-1.0, 0.05], 2, 0.1,
    "Burgers: u_t = -u u_x + 0.05 u_xx")

# Allen-Cahn
run_one(
    PDESimulator.allen_cahn(eps=0.01, N=256, T=0.3, n_t=100, n_modes=6, seed=42, backend="numpy"),
    ["u", "u^3", "u_xx"], [1.0, -1.0, 0.01], 3, 0.1,
    "Allen-Cahn: u_t = 0.01 u_xx + u - u^3")

# Fisher-KPP
run_one(
    PDESimulator.fisher_kpp(D=0.01, r=1.0, N=256, T=0.5, n_t=100, n_modes=6, seed=42, backend="numpy"),
    ["u", "u^2", "u_xx"], [1.0, -1.0, 0.01], 3, 0.1,
    "Fisher-KPP: u_t = 0.01 u_xx + u (1-u)")

