"""
examples/kdv_identification.py
================================
KdV operator identification:  u_t = -6 u u_x - u_xxx

Identifies the KdV operator from a single simulated trajectory using
OMP, column-normalised LassoCV with threshold truncation, and SCIP MILP.

Usage:  python examples/kdv_identification.py
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from opid import PDESimulator, FeatureLibrary, OperatorIdentifier, jaccard_score

TRUE  = ["u u_x", "u_xxx"]
COEFS = [-6.0, -1.0]

# ── 1. Simulate ──────────────────────────────────────────────────────────
print("Simulating KdV (N=256, T=0.05, n_t=100, 8 random Fourier modes)...")
sim = PDESimulator.kdv(N=256, T=0.05, n_t=100, n_modes=8, seed=42, backend="numpy")
U, U_t = sim.run()
y = U_t.ravel()

# ── 2. Build library ─────────────────────────────────────────────────────
print("Building feature library (poly_degree=2, max_deriv=3)...")
lib = FeatureLibrary(poly_degree=2, max_deriv=3, max_cross_degree=2)
Theta, names = lib.build(U, sim.k)
print(f"  {len(names)} terms: {names}")

# ── 3. OMP ───────────────────────────────────────────────────────────────
t0 = time.time()
r_omp = OperatorIdentifier(method="omp", n_nonzero=2).fit(Theta, y, names)
J_omp = jaccard_score(r_omp.names, TRUE)
print(f"\nOMP ({time.time()-t0:.2f}s)  Jaccard = {J_omp:.2f}")
for n, c in zip(r_omp.names, r_omp.active_coef):
    flag = "✓" if n in TRUE else "✗"
    print(f"  {flag} {n:>12s} = {c:+.6f}")

# ── 4. LassoCV + threshold + OLS ────────────────────────────────────────
t0 = time.time()
r_las = OperatorIdentifier(method="lasso", alpha=0.5).fit(Theta, y, names)
J_las = jaccard_score(r_las.names, TRUE)
print(f"\nLassoCV ({time.time()-t0:.2f}s)  Jaccard = {J_las:.2f}")
for n, c in zip(r_las.names, r_las.active_coef):
    flag = "✓" if n in TRUE else "✗"
    print(f"  {flag} {n:>12s} = {c:+.6f}")

# ── 5. SCIP MILP (bisection-based Pareto sweep) ──────────────────────────
t0 = time.time()
r_milp = OperatorIdentifier(
    method="l0_pareto", n_eps=10, max_samples=2000,
    milp_solver="SCIP", threshold_coef=1e-8, random_state=42
).fit(Theta, y, names)
J_milp = jaccard_score(r_milp.names, TRUE)
print(f"\nSCIP MILP ({time.time()-t0:.0f}s)  Jaccard = {J_milp:.2f}")
for n, c in zip(r_milp.names, r_milp.active_coef):
    flag = "✓" if n in TRUE else "✗"
    print(f"  {flag} {n:>12s} = {c:+.6f}")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'─'*45}")
print(f"  Ground truth:  u u_x = -6.0,  u_xxx = -1.0")
print(f"  OMP:     J = {J_omp:.2f}")
print(f"  LassoCV: J = {J_las:.2f}")
print(f"  MILP:    J = {J_milp:.2f}")
