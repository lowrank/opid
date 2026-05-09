"""
examples/allen_cahn_identification.py
=======================================
Allen-Cahn operator identification:  u_t = ε u_xx + u - u³  (ε = 0.01)

Identifies the operator from a simulated trajectory using OMP, LassoCV,
and SCIP MILP.  Requires the L-size library (P=40) to include u³.

Usage:  python examples/allen_cahn_identification.py
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opid import PDESimulator, FeatureLibrary, OperatorIdentifier, jaccard_score

TRUE  = ["u", "u^3", "u_xx"]
COEFS = [1.0, -1.0, 0.01]

print("Simulating Allen-Cahn (N=256, T=0.3, ε=0.01)...")
sim = PDESimulator.allen_cahn(eps=0.01, N=256, T=0.3, n_t=100, n_modes=6, seed=42, backend="numpy")
U, U_t = sim.run()
y = U_t.ravel()

# Need L-size library to include u³
lib = FeatureLibrary(poly_degree=3, max_deriv=4, max_cross_degree=4)
Theta, names = lib.build(U, sim.k)
print(f"  {len(names)} terms")

# OMP
t0 = time.time()
r = OperatorIdentifier(method="omp", n_nonzero=3).fit(Theta, y, names)
print(f"\nOMP ({time.time()-t0:.2f}s)  J={jaccard_score(r.names, TRUE):.2f}")
for n, c in zip(r.names, r.active_coef):
    print(f"  {'✓' if n in TRUE else '✗'} {n:>12s} = {c:+.6f}")

# LassoCV
t0 = time.time()
r = OperatorIdentifier(method="lasso", alpha=0.1).fit(Theta, y, names)
print(f"\nLassoCV ({time.time()-t0:.2f}s)  J={jaccard_score(r.names, TRUE):.2f}")
for n, c in zip(r.names, r.active_coef):
    print(f"  {'✓' if n in TRUE else '✗'} {n:>12s} = {c:+.6f}")

# SCIP MILP
t0 = time.time()
r = OperatorIdentifier(method="l0_pareto", n_eps=10, max_samples=2000,
                       milp_solver="SCIP", threshold_coef=1e-8,
                       random_state=42).fit(Theta, y, names)
print(f"\nSCIP MILP ({time.time()-t0:.0f}s)  J={jaccard_score(r.names, TRUE):.2f}")
for n, c in zip(r.names, r.active_coef):
    print(f"  {'✓' if n in TRUE else '✗'} {n:>12s} = {c:+.6f}")

print(f"\n{'─'*45}")
print(f"  Ground truth:  u = +1.0,  u³ = -1.0,  u_xx = +0.01")
