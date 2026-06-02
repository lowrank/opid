"""CLI worker for solving a single MILP instance (subprocess-safe).

Usage: python -m opid.recovery._milp_worker <problem.json>

Reads Theta, y, eps, M from JSON.  Solves min Σz s.t. L1 ≤ eps, Big-M.
Prints {"z": [...], "xi": [...]} on success.  Dies silently on CoinError.
"""

import json, sys
import numpy as np

try:
    import cvxpy as cp
except ImportError:
    print(json.dumps({"z": [], "xi": []}))
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    Theta = np.array(data["Theta"])
    y = np.array(data["y"])
    eps = float(data["eps"])
    M = np.array(data["M"])
    solver = data.get("solver", "CBC")

    P = Theta.shape[1]
    xi = cp.Variable(P)
    z = cp.Variable(P, boolean=True)

    prob = cp.Problem(
        cp.Minimize(cp.sum(z)),
        [
            cp.norm(y - Theta @ xi, 1) <= eps,
            xi <= cp.multiply(M, z),
            xi >= cp.multiply(-M, z),
        ],
    )

    prob.solve(solver=getattr(cp, solver), warm_start=False)

    if prob.status in ("optimal", "optimal_inaccurate"):
        z_vals = [float(z.value[i]) for i in range(P)]
        xi_vals = [float(xi.value[i]) for i in range(P)]
        print(json.dumps({"z": z_vals, "xi": xi_vals}))
    else:
        print(json.dumps({"z": [], "xi": []}))
