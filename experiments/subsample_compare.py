#!/usr/bin/env python3
"""Compare CCP with Full/Random/QR subsampling at S/M/L/XL."""

import json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from opid import FeatureLibrary, OperatorIdentifier, jaccard_score
from opid.recovery.subsample import FullSubsampler, RandomSubsampler, QRSubsampler

N_SEEDS = 20
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_data")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "..", "benchmark_results", "subsample_compare.json")
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
]

SUBSAMPLERS = [
    ("Full",   FullSubsampler()),
    ("Random", RandomSubsampler(seed=42)),
    ("QR",     QRSubsampler()),
]

if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE) as f:
        results = json.load(f)
else:
    results = {}

print("=" * 60)
print("  CCP × Subsamplers × S/M/L/XL — 7 PDEs × 20 seeds")
print("=" * 60)

for pde_name, true_set in PDES:
    for seed in range(N_SEEDS):
        data_file = os.path.join(DATA_DIR, f"{pde_name}_seed{seed}.npz")
        if not os.path.exists(data_file):
            continue
        data = np.load(data_file)
        U, Ut = data["U"], data["Ut"]
        y = Ut.ravel()

        for pd, md, mcd, lib_name in LIBS:
            lib = FeatureLibrary(poly_degree=pd, max_deriv=md, max_cross_degree=mcd)
            Theta, names = lib.build(U)
            if len(true_set & set(names)) < len(true_set):
                continue

            for sub_name, sub in SUBSAMPLERS:
                rkey = f"{pde_name}_seed{seed}|{lib_name}|CCP_{sub_name}"
                if rkey in results:
                    continue

                t0 = time.time()
                try:
                    oid = OperatorIdentifier(method="ccp", subsampler=sub, verbose=False)
                    res = oid.fit(Theta, y, names)
                    J = jaccard_score(res.names, true_set)
                    dt = time.time() - t0
                    results[rkey] = {"J": J, "support": list(res.names), "time": dt, "P": len(names)}
                except Exception as e:
                    dt = time.time() - t0
                    results[rkey] = {"J": -1, "error": str(e)[:100], "time": dt, "P": len(names)}

                with open(RESULT_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  {rkey}  J={results[rkey].get('J',-1):.2f} {dt:.1f}s", flush=True)

# Summary
print("\n" + "=" * 60)
print("  Summary: Full vs Random vs QR subsampling")
print("=" * 60)

for sub_name in ["Full", "Random", "QR"]:
    print(f"\n  CCP_{sub_name}:")
    print(f"  {'PDE':>8s}   S    M    L   XL")
    for pde_name, _ in PDES:
        row = f"  {pde_name:>8s}"
        for _, _, _, lib_name in LIBS:
            j1 = sum(1 for s in range(N_SEEDS)
                     if results.get(f"{pde_name}_seed{s}|{lib_name}|CCP_{sub_name}", {}).get("J",-1) == 1.0)
            row += f"  {j1:>2d}"
        print(row)

# Timing comparison
print(f"\n  Avg time per seed (seconds):")
print(f"  {'PDE':>8s} {'Full':>8s} {'Random':>8s} {'QR':>8s}")
for pde_name, _ in PDES:
    times = {}
    for sub_name in ["Full", "Random", "QR"]:
        ts = [results.get(f"{pde_name}_seed{s}|L|CCP_{sub_name}", {}).get("time", 0)
              for s in range(N_SEEDS)
              if f"{pde_name}_seed{s}|L|CCP_{sub_name}" in results]
        times[sub_name] = np.mean(ts) if ts else 0
    print(f"  {pde_name:>8s} {times['Full']:>7.2f}s {times['Random']:>7.2f}s {times['QR']:>7.2f}s")

print(f"\nResults: {RESULT_FILE}")
