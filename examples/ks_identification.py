"""
examples/ks_identification.py
===============================
Kuramoto-Sivashinsky operator identification:
    u_t = -u u_x - u_xx - u_xxxx

Usage:  python examples/ks_identification.py
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opid import PDESimulator, FeatureLibrary, OperatorIdentifier, jaccard_score

TRUE  = ["u u_x", "u_xx", "u_xxxx"]
COEFS = [-1.0, -1.0, -1.0]

print("Simulating KS (N=256, T=0.08)...")
sim = PDESimulator.ks(N=256, T=0.08, n_t=100, n_modes=8, seed=42, backend="numpy")
U, U_t = sim.run()
y = U_t.ravel()

lib = FeatureLibrary(poly_degree=2, max_deriv=4, max_cross_degree=3)
Theta, names = lib.build(U, sim.k)
print(f"  {len(names)} terms")

# OMP
t0 = time.time()
r = OperatorIdentifier(method="omp", n_nonzero=3).fit(Theta, y, names)
print(f"\nOMP ({time.time()-t0:.2f}s)  J={jaccard_score(r.names, TRUE):.2f}")
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
print(f"  Ground truth:  all coefficients = -1.0")
