#!/usr/bin/env python3
"""Run CBC n_eps=30 on AC/KS/FHN remaining seeds and update results."""

import numpy as np, time, json, subprocess, sys, warnings
warnings.filterwarnings("ignore")

from opid import FeatureLibrary, OperatorIdentifier, jaccard_score

RESULTS_FILE = "benchmark_results/core_methods.json"
with open(RESULTS_FILE) as f:
    results = json.load(f)

configs = [
    ("KS", range(20), {"u u_x", "u_xx", "u_xxxx"}, [(2, 4, 3, "M"), (3, 4, 4, "L")]),
    ("FHN", range(20), {"u_xx", "u^2", "u^3"}, [(3, 4, 4, "L")]),
]

for pde, seeds, true, libs in configs:
    for seed in seeds:
        for pd, md, mcd, lib_name in libs:
            key = f"{pde}_seed{seed}|{lib_name}|L0_CBC"
            if key in results:
                continue

            data = np.load(f"benchmark_data/{pde}_seed{seed}.npz")
            lib = FeatureLibrary(poly_degree=pd, max_deriv=md, max_cross_degree=mcd)
            Th, nm = lib.build(data["U"])
            y = data["Ut"].ravel()
            if len(true & set(nm)) < len(true):
                continue

            t0 = time.time()
            try:
                oid = OperatorIdentifier(
                    method="l0_pareto", milp_solver="CBC",
                    n_eps=30, max_samples=500, random_state=42, verbose=False,
                )
                res = oid.fit(Th, y, nm)
                J = jaccard_score(res.names, true)
                dt = time.time() - t0
                results[key] = {"J": J, "support": list(res.names), "time": dt, "P": len(nm)}
                print(f"{key}: J={J:.2f}, {dt:.0f}s")
            except Exception as e:
                results[key] = {"J": -1, "error": str(e)[:80], "time": time.time() - t0, "P": len(nm)}
                print(f"{key}: FAILED - {e}")

            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

# Summary
for pde in ["KS", "FHN"]:
    for lib_name in ["M", "L"]:
        j1 = sum(1 for s in range(20) if results.get(f"{pde}_seed{s}|{lib_name}|L0_CBC", {}).get("J", -1) == 1.0)
        cr = sum(1 for s in range(20) if results.get(f"{pde}_seed{s}|{lib_name}|L0_CBC", {}).get("J", -1) < 0)
        total = sum(1 for s in range(20) if f"{pde}_seed{s}|{lib_name}|L0_CBC" in results)
        if total > 0:
            print(f"{pde} {lib_name}: {j1}/{total} J=1, {cr} crashes")
