"""
examples/fkpp_identification.py
=================================
Fisher-KPP operator identification:  u_t = D u_xx + r u (1-u)
with D = 0.01, r = 1.

Expands to  u_t = 0.01 u_xx + u - u^2.
Requires SCIP MILP — OMP fails on this PDE.

Usage:  python examples/fkpp_identification.py
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opid import PDESimulator, FeatureLibrary, OperatorIdentifier, jaccard_score

TRUE  = ["u", "u^2", "u_xx"]
COEFS = [1.0, -1.0, 0.01]

print("Simulating Fisher-KPP (N=256, T=0.5, D=0.01, r=1)...")
sim = PDESimulator.fisher_kpp(D=0.01, r=1.0, N=256, T=0.5, n_t=100,
                              n_modes=6, seed=42, backend="numpy")
U, U_t = sim.run()
y = U_t.ravel()

lib = FeatureLibrary(poly_degree=2, max_deriv=3, max_cross_degree=2)
Theta, names = lib.build(U, sim.k)
print(f"  {len(names)} terms")

# OMP (expected to fail — picks correlated alternatives)
t0 = time.time()
r = OperatorIdentifier(method="omp", n_nonzero=3).fit(Theta, y, names)
print(f"\nOMP ({time.time()-t0:.2f}s)  J={jaccard_score(r.names, TRUE):.2f}")
for n, c in zip(r.names, r.active_coef):
    print(f"  {'✓' if n in TRUE else '✗'} {n:>12s} = {c:+.6f}")

# SCIP MILP (recovers exact support)
t0 = time.time()
r = OperatorIdentifier(method="l0_pareto", n_eps=10, max_samples=2000,
                       milp_solver="SCIP", threshold_coef=1e-8,
                       random_state=42).fit(Theta, y, names)
print(f"\nSCIP MILP ({time.time()-t0:.0f}s)  J={jaccard_score(r.names, TRUE):.2f}")
for n, c in zip(r.names, r.active_coef):
    print(f"  {'✓' if n in TRUE else '✗'} {n:>12s} = {c:+.6f}")

print(f"\n{'─'*45}")
print(f"  Ground truth:  u = +1.0,  u² = -1.0,  u_xx = +0.01")
print(f"  OMP fails (J=0) — MILP required for this PDE.")
