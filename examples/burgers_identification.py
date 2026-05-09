"""
examples/burgers_identification.py
=====================================
Viscous Burgers operator identification:  u_t = -u u_x + ν u_xx  (ν = 0.05)

Identifies the operator from a simulated trajectory using OMP, LassoCV,
and SCIP MILP.

Usage:  python examples/burgers_identification.py
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opid import PDESimulator, FeatureLibrary, OperatorIdentifier, jaccard_score

TRUE  = ["u u_x", "u_xx"]
COEFS = [-1.0, 0.05]

print("Simulating viscous Burgers (N=256, T=0.5, ν=0.05)...")
sim = PDESimulator.burgers(nu=0.05, N=256, T=0.5, n_t=100, n_modes=8, seed=42, backend="numpy")
U, U_t = sim.run()
y = U_t.ravel()

lib = FeatureLibrary(poly_degree=2, max_deriv=3, max_cross_degree=2)
Theta, names = lib.build(U, sim.k)
print(f"  {len(names)} terms")

# OMP
t0 = time.time()
r = OperatorIdentifier(method="omp", n_nonzero=2).fit(Theta, y, names)
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
print(f"  Ground truth:  u u_x = -1.0,  u_xx = +0.05")
