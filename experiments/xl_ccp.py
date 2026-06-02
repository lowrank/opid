#!/usr/bin/env python3
"""CCP-family benchmark on XL library (P=59) for all 7 PDEs, 20 seeds."""

import json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from opid import PDESimulator, FeatureLibrary, OperatorIdentifier, jaccard_score

N_SEEDS = 20
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_data")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "..", "benchmark_results", "xl_ccp.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

PDES = [
    ("KdV",     lambda s: PDESimulator.kdv(N=128, T=0.05, n_t=100, n_modes=4, seed=s, backend="numpy"),
     {"u u_x", "u_xxx"}),
    ("Burgers", lambda s: PDESimulator.burgers(nu=0.05, N=128, T=0.3, n_t=100, n_modes=4, seed=s, backend="numpy"),
     {"u u_x", "u_xx"}),
    ("AC",      lambda s: PDESimulator.allen_cahn(eps=0.01, N=128, T=0.1, n_t=100, n_modes=4, seed=s, backend="numpy"),
     {"u", "u^3", "u_xx"}),
    ("KS",      lambda s: PDESimulator.ks(N=128, T=0.08, n_t=100, n_modes=4, seed=s, backend="numpy"),
     {"u u_x", "u_xx", "u_xxxx"}),
    ("FKPP",    lambda s: PDESimulator.fisher_kpp(D=0.1, r=1.0, N=128, T=0.2, n_t=100, n_modes=4, seed=s, backend="numpy"),
     {"u", "u^2", "u_xx"}),
    ("KdVB",    lambda s: PDESimulator.kdv_burgers(N=128, T=0.05, n_t=100, n_modes=4, seed=s, backend="numpy"),
     {"u u_x", "u_xx", "u_xxx"}),
    ("FHN",     lambda s: PDESimulator.fitzhugh_nagumo(N=128, T=0.1, n_t=100, n_modes=4, seed=s, backend="numpy"),
     {"u_xx", "u^2", "u^3"}),
]

XL = dict(poly_degree=3, max_deriv=5, max_cross_degree=5)

CCP_METHODS = [
    ("CCP",       dict(method="ccp")),
    ("CCP_milp",  dict(method="ccp", milp_vote=True)),
    ("CCP_c4",    dict(method="ccp", cluster_size=4)),
    ("CCP_c16",   dict(method="ccp", cluster_size=16)),
]

if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE) as f:
        results = json.load(f)
else:
    results = {}

print("=" * 60)
print("  CCP × XL (P=59) — 7 PDEs × 20 seeds")
print("=" * 60)

for pde_name, factory, true_set in PDES:
    for seed in range(N_SEEDS):
        data_file = os.path.join(DATA_DIR, f"{pde_name}_seed{seed}.npz")
        if not os.path.exists(data_file):
            print(f"  {pde_name}_seed{seed}: NO DATA, skip")
            continue
        data = np.load(data_file)
        U, Ut = data["U"], data["Ut"]
        y = Ut.ravel()

        lib = FeatureLibrary(**XL)
        Theta, names = lib.build(U)
        P = len(names)
        if len(true_set & set(names)) < len(true_set):
            continue

        for method_name, method_kw in CCP_METHODS:
            result_key = f"{pde_name}_seed{seed}|XL|{method_name}"
            if result_key in results:
                continue

            t0 = time.time()
            try:
                kw = dict(method_kw)
                oid = OperatorIdentifier(**kw)
                res = oid.fit(Theta, y, names)
                J = jaccard_score(res.names, true_set)
                dt = time.time() - t0
                results[result_key] = {"J": J, "support": list(res.names),
                                        "time": dt, "P": P}
            except Exception as e:
                dt = time.time() - t0
                results[result_key] = {"J": -1, "error": str(e)[:100], "time": dt, "P": P}

            with open(RESULT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  {result_key}  J={results[result_key].get('J',-1):.2f} {dt:.1f}s", flush=True)

# Summary
print("\n" + "=" * 60)
print("  Summary: XL (P=59)")
print("=" * 60)

summary = {}
for rkey, rval in results.items():
    parts = rkey.split("|")
    if len(parts) != 3: continue
    pde_seed, lib, method = parts
    pde = pde_seed.rsplit("_seed", 1)[0]
    tag = f"{pde}|{method}"
    if tag not in summary:
        summary[tag] = {"total": 0, "J1": 0}
    summary[tag]["total"] += 1
    if rval.get("J", -1) == 1.0:
        summary[tag]["J1"] += 1

for tag, counts in sorted(summary.items()):
    pde, method = tag.split("|")
    t = counts["total"]
    print(f"  {pde:>8s} {method:<12s}: {counts['J1']}/{t} J=1 ({counts['J1']/t*100:.0f}%)")

with open(RESULT_FILE.replace(".json", "_summary.json"), "w") as f:
    out = {}
    for tag, counts in sorted(summary.items()):
        pde, method = tag.split("|")
        out.setdefault(pde, {})[method] = f"{counts['J1']}/{counts['total']}"
    json.dump(out, f, indent=2)

print(f"\nResults: {RESULT_FILE}")
