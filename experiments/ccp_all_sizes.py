#!/usr/bin/env python3
"""CCP benchmark across all library sizes for 7 PDEs × 20 seeds."""

import json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from opid import PDESimulator, FeatureLibrary, OperatorIdentifier, jaccard_score

N_SEEDS = 20
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_data")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "..", "benchmark_results", "ccp_all_sizes.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

PDES = [
    ("KdV",     {"u u_x", "u_xxx"}),
    ("Burgers", {"u u_x", "u_xx"}),
    ("AC",      {"u", "u^3", "u_xx"}),
    ("KS",      {"u u_x", "u_xx", "u_xxxx"}),
    ("FKPP",    {"u", "u^2", "u_xx"}),
    ("KdVB",    {"u u_x", "u_xx", "u_xxx"}),
    ("FHN",     {"u_xx", "u^2", "u^3"}),
]

LIBS = [
    (2,3,2, "S"), (2,4,3, "M"), (3,4,4, "L"), (3,5,5, "XL"),
    (4,4,4, "P45"), (4,5,4, "P55"), (4,5,5, "P75"), (5,5,5, "P81"),
    (4,6,5, "P89"), (5,6,5, "P96"), (5,6,6, "P126"), (6,6,6, "P133"),
    (5,7,6, "P146"), (6,7,6, "P154"), (5,7,7, "P174"), (6,7,7, "P196"),
]

if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE) as f:
        results = json.load(f)
else:
    results = {}

print("=" * 60)
print("  CCP × All Sizes — 7 PDEs × 20 seeds")
print("=" * 60)

for pde_name, true_set in PDES:
    for seed in range(N_SEEDS):
        data_file = os.path.join(DATA_DIR, f"{pde_name}_seed{seed}.npz")
        if not os.path.exists(data_file):
            print(f"  {pde_name}_seed{seed}: NO DATA, skip")
            continue
        data = np.load(data_file)
        U, Ut = data["U"], data["Ut"]
        y = Ut.ravel()

        for pd, md, mcd, lib_name in LIBS:
            lib = FeatureLibrary(poly_degree=pd, max_deriv=md, max_cross_degree=mcd)
            Theta, names = lib.build(U)
            P = len(names)
            if len(true_set & set(names)) < len(true_set):
                continue

            rkey = f"{pde_name}_seed{seed}|{lib_name}|CCP"
            if rkey in results:
                continue

            t0 = time.time()
            try:
                oid = OperatorIdentifier(method="ccp", verbose=False)
                res = oid.fit(Theta, y, names)
                J = jaccard_score(res.names, true_set)
                dt = time.time() - t0
                results[rkey] = {"J": J, "support": list(res.names), "time": dt, "P": P}
            except Exception as e:
                dt = time.time() - t0
                results[rkey] = {"J": -1, "error": str(e)[:100], "time": dt, "P": P}

            with open(RESULT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  {rkey}  J={results[rkey].get('J',-1):.2f} {dt:.1f}s", flush=True)

# Summary
print("\n" + "=" * 60)
print("  Summary: CCP across all library sizes")
print("=" * 60)

print(f"\n{'PDE':>8s}", end="")
for _, _, _, name in LIBS:
    print(f" {name:>6s}", end="")
print()

for pde_name, _ in PDES:
    print(f"{pde_name:>8s}", end="")
    for pd, md, mcd, lib_name in LIBS:
        j1 = 0
        for seed in range(N_SEEDS):
            v = results.get(f"{pde_name}_seed{seed}|{lib_name}|CCP", {})
            if v.get("J", -1) == 1.0:
                j1 += 1
        print(f" {j1:>2d}/{N_SEEDS}", end="")
    print()

print(f"\nResults: {RESULT_FILE}")
