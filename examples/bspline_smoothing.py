"""
examples/bspline_smoothing.py
===============================
B-spline smoothing for noisy spatial data.

When measurement noise contaminates the raw spatial data U(x,t),
high-order derivatives (u_xxx, u_xxxx) amplify the noise.
B-spline pre-smoothing projects each noisy snapshot onto a smooth
function space BEFORE computing FFT derivatives, dramatically
improving derivative quality.

Usage:  python examples/bspline_smoothing.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from opid import PDESimulator, FeatureLibrary, OperatorIdentifier, jaccard_score

NOISE  = 0.01   # 1% measurement noise
TRUE   = ["u u_x", "u_xxx"]
COEFS  = [-6.0, -1.0]

print("Simulating clean KdV solution...")
sim = PDESimulator.kdv(N=256, T=0.05, n_t=100, n_modes=8, seed=42, backend="numpy")
U_clean, U_t = sim.run()

# Add spatial noise (except initial condition)
rng = np.random.default_rng(42)
U_noisy = U_clean.copy()
U_noisy[:, 1:] += NOISE * np.std(U_clean) * rng.standard_normal((U_clean.shape[0], U_clean.shape[1] - 1))
snr = np.std(U_clean) / np.std(U_noisy - U_clean)
print(f"Added {NOISE*100:.0f}% noise  →  SNR = {snr:.0f}")

# ── WITHOUT smoothing ──────────────────────────────────────────────────
print("\n── Without B-spline ──")
lib = FeatureLibrary(poly_degree=2, max_deriv=3, max_cross_degree=2)
Theta_no, names = lib.build(U_noisy, sim.k)
r = OperatorIdentifier(method="omp", n_nonzero=2).fit(Theta_no, U_t.ravel(), names)
print(f"OMP: J={jaccard_score(r.names, TRUE):.2f}  {r.names}")

# ── WITH smoothing ─────────────────────────────────────────────────────
print("\n── With B-spline (degree=5, knots=25) ──")
lib_s = FeatureLibrary(poly_degree=2, max_deriv=3, max_cross_degree=2,
                       smooth=True, smooth_degree=5, smooth_n_knots=25)
Theta_s, names = lib_s.build(U_noisy, sim.k)
r = OperatorIdentifier(method="omp", n_nonzero=2).fit(Theta_s, U_t.ravel(), names)
print(f"OMP: J={jaccard_score(r.names, TRUE):.2f}  {r.names}")

# ── Derivative quality comparison ──────────────────────────────────────
Theta_clean, _ = lib.build(U_clean, sim.k)
for order, term in [(1, "u_x"), (2, "u_xx"), (3, "u_xxx")]:
    i = names.index(term)
    e_no = np.linalg.norm(Theta_clean[:, i] - Theta_no[:, i]) / np.linalg.norm(Theta_clean[:, i])
    e_s  = np.linalg.norm(Theta_clean[:, i] - Theta_s[:, i])  / np.linalg.norm(Theta_clean[:, i])
    print(f"  {term}: error {e_no:.3f} → {e_s:.3f}  ({e_no/e_s:.1f}× better)")
