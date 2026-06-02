#!/usr/bin/env python3
"""Compare all methods across all subsamplers at S/M/L — 7 PDEs × 5 seeds."""

import json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from opid import FeatureLibrary, OperatorIdentifier, jaccard_score
from opid.recovery.subsample import FullSubsampler, RandomSubsampler, QRSubsampler, SignalQRSubsampler

N_SEEDS = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_data")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "..", "benchmark_results", "methods_subsample.json")
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

LIBS = [(2,3,2,"S"), (2,4,3,"M"), (3,4,4,"L")]

METHODS = [
    ("CCP",   dict(method="ccp")),
    ("OMP",   dict(method="omp")),
    ("Lasso", dict(method="lasso")),
]

SUBS = {
    "Full":   FullSubsampler(),
    "Random": RandomSubsampler(42),
    "QR":     QRSubsampler(),
    "SigQR":  SignalQRSubsampler(),
}

if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE) as f: results = json.load(f)
else: results = {}

for pde_name, true_set in PDES:
    for seed in range(N_SEEDS):
        df = os.path.join(DATA_DIR, f"{pde_name}_seed{seed}.npz")
        if not os.path.exists(df): continue
        data = np.load(df)
        U, Ut = data["U"], data["Ut"]
        y = Ut.ravel()
        nnz = len(true_set)

        for pd, md, mcd, lib_name in LIBS:
            lib = FeatureLibrary(poly_degree=pd, max_deriv=md, max_cross_degree=mcd)
            Theta, names = lib.build(U)
            if len(true_set & set(names)) < len(true_set): continue

            for mname, mkw in METHODS:
                for sub_name, sub in SUBS.items():
                    rkey = f"{pde_name}_seed{seed}|{lib_name}|{mname}_{sub_name}"
                    if rkey in results: continue

                    t0 = time.time()
                    try:
                        kw = dict(mkw)
                        kw["subsampler"] = sub
                        if kw["method"] == "omp": kw["n_nonzero"] = nnz
                        oid = OperatorIdentifier(**kw, verbose=False)
                        res = oid.fit(Theta, y, names)
                        J = jaccard_score(res.names, true_set)
                        results[rkey] = {"J": J, "support": list(res.names), "time": time.time()-t0, "P": len(names)}
                    except Exception as e:
                        results[rkey] = {"J": -1, "error": str(e)[:80], "time": time.time()-t0, "P": len(names)}

                    with open(RESULT_FILE, "w") as f: json.dump(results, f, indent=2)
                    print(f"  {rkey}  J={results[rkey].get('J',-1):.2f}", flush=True)

# Summary
print("\nSummary:")
for mname in ["CCP", "OMP", "Lasso"]:
    for sub_name in ["Full", "Random", "QR", "SigQR"]:
        print(f"\n  {mname}_{sub_name}:")
        for pde_name, _ in PDES:
            row = f"  {pde_name:>8s}"
            for _,_,_,lib_name in LIBS:
                j1 = sum(1 for s in range(N_SEEDS) if results.get(f"{pde_name}_seed{s}|{lib_name}|{mname}_{sub_name}",{}).get("J",-1)==1.0)
                row += f" {j1:>2d}"
            print(row)

print(f"\nResults: {RESULT_FILE}")
