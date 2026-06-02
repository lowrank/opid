"""CLI worker for MILP solves (subprocess-safe, CoinError-crash-resistant).

Usage: python -m opid.recovery._milp_worker <problem.json>

The JSON file contains either:
  a) {"Theta": ..., "y": ..., "eps": ..., "M": ..., "solver": "CBC"}
     → single solve, prints {"z": [...], "xi": [...]}
  b) {"Theta": ..., "y": ..., "eps": [e1, e2, ...], "M": ..., "solver": "CBC"}
     → batch solve for each eps, prints [{"eps": e, "z": [...]}, ...]
  c) {"Theta": ..., "y": ..., "M": ..., "noise_floor": ..., "eps_lo_factor": ..., "eps_hi_factor": ..., "n_eps": ..., "solver": "CBC"}
     → full Pareto sweep, prints [{"eps": e, "z": [...]}, ...]
"""

import json, sys
import numpy as np

try:
    import cvxpy as cp
except ImportError:
    print(json.dumps([]))
    sys.exit(0)

def solve_one(Theta, y, M, eps, solver):
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
    prob.solve(solver=getattr(cp, solver, "CBC"), warm_start=False)
    if prob.status in ("optimal", "optimal_inaccurate"):
        return {"z": [float(z.value[i]) for i in range(P)],
                "xi": [float(xi.value[i]) for i in range(P)]}
    return {"z": [], "xi": []}


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        data = json.load(f)

    Theta = np.array(data["Theta"])
    y = np.array(data["y"])
    M = np.array(data["M"])
    solver = data.get("solver", "CBC")

    # Mode c): full Pareto sweep
    if "noise_floor" in data:
        noise = data["noise_floor"]
        eps_lo = noise * data["eps_lo_factor"]
        eps_hi = noise * data["eps_hi_factor"]
        n_eps = data["n_eps"]

        # Solve at eps_lo
        r_lo = solve_one(Theta, y, M, eps_lo, solver)
        k_lo = len([z for z in r_lo["z"] if z > 0.5]) if r_lo["z"] else 0

        # Exponential search: if infeasible, double eps_lo
        while (not r_lo["z"]) and eps_lo < eps_hi * 0.9:
            eps_lo *= 2.0
            r_lo = solve_one(Theta, y, M, eps_lo, solver)
            k_lo = len([z for z in r_lo["z"] if z > 0.5]) if r_lo["z"] else 0

        # Solve at eps_hi
        r_hi = solve_one(Theta, y, M, eps_hi, solver)
        k_hi = len([z for z in r_hi["z"] if z > 0.5]) if r_hi["z"] else 0
        P = Theta.shape[1]
        if k_lo == 0: k_lo = P  # all terms = no sparsity found

        results = [{"eps": eps_lo, **r_lo}] if r_lo["z"] else []
        if r_hi["z"]:
            results.append({"eps": eps_hi, **r_hi})

        # Bisection for intermediate k
        frontier = {k_lo: eps_lo, k_hi: eps_hi}
        for k in range(k_hi + 1, k_lo):
            left = frontier.get(k + 1, eps_lo)
            right = frontier.get(k - 1, eps_hi)
            if left >= right:
                frontier[k] = right
                continue
            for _ in range(n_eps // max(k_lo - k_hi, 1)):
                mid = np.sqrt(left * right)
                rm = solve_one(Theta, y, M, mid, solver)
                if not rm["z"]: break
                km = len([z for z in rm["z"] if z > 0.5])
                if km <= k: right = mid
                else: left = mid
                if right / left < 1.02: break
            frontier[k] = right

        # Collect all frontier eps values and solve for final support
        seen = set()
        for eps in sorted(frontier.values()):
            if eps in seen: continue
            seen.add(eps)
            r = solve_one(Theta, y, M, eps, solver)
            if r["z"]:
                results.append({"eps": eps, **r})

        results.sort(key=lambda r: r["eps"])
        print(json.dumps(results))

    # Mode b): batch of eps values
    elif isinstance(data.get("eps"), list):
        results = []
        for eps in data["eps"]:
            r = solve_one(Theta, y, M, eps, solver)
            if r["z"]:
                results.append({"eps": eps, **r})
        print(json.dumps(results))

    # Mode a): single eps
    else:
        r = solve_one(Theta, y, M, data["eps"], solver)
        print(json.dumps(r))
