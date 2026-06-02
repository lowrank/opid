#!/usr/bin/env python3
"""Full benchmark: all methods × library sizes × 20 seeds for 7 PDEs.
Saves simulation data for reuse, runs incrementally, outputs JSON.
Usage: python experiments/full_benchmark.py
"""

import json, os, time, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
from opid import PDESimulator, FeatureLibrary, OperatorIdentifier, jaccard_score

N_SEEDS = 20
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_data")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "..", "benchmark_results", "full.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────
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

LIBRARIES = [
    ("S",  dict(poly_degree=2, max_deriv=3, max_cross_degree=2)),   # P≈12
    ("M",  dict(poly_degree=2, max_deriv=4, max_cross_degree=3)),   # P≈23
    ("L",  dict(poly_degree=3, max_deriv=4, max_cross_degree=4)),   # P=40
]

METHODS = [
    ("OMP",       dict(method="omp")),
    ("Lasso",     dict(method="lasso")),
    ("CCP",       dict(method="ccp")),
    ("CCP_c4",    dict(method="ccp", cluster_size=4)),
    ("CCP_c16",   dict(method="ccp", cluster_size=16)),
    ("L0_CBC",    dict(method="l0_pareto", milp_solver="CBC", n_eps=30, max_samples=500)),
    ("L0_SCIP",   dict(method="l0_pareto", milp_solver="SCIP", n_eps=30, max_samples=500)),
]

# Load or create results dict
if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE) as f:
        results = json.load(f)
else:
    results = {}

# ── Task 1: Generate / load simulation data ───────────────────────────
print("=" * 60)
print("  Task 1: Simulation data")
print("=" * 60)

for pde_name, factory, _ in PDES:
    for seed in range(N_SEEDS):
        key = f"{pde_name}_seed{seed}"
        data_file = os.path.join(DATA_DIR, f"{key}.npz")
        if os.path.exists(data_file):
            print(f"  {key} already exists, skipping")
            continue
        t0 = time.time()
        sim = factory(seed)
        U, Ut = sim.run()
        np.savez_compressed(data_file, U=U, Ut=Ut)
        print(f"  {key} saved ({U.shape}) in {time.time()-t0:.1f}s", flush=True)

# ── Task 2: Run all methods ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  Task 2: Methods × Libraries")
print("=" * 60)

for pde_name, factory, true_set in PDES:
    for seed in range(N_SEEDS):
        key = f"{pde_name}_seed{seed}"
        data_file = os.path.join(DATA_DIR, f"{key}.npz")
        if not os.path.exists(data_file):
            continue
        data = np.load(data_file)
        U, Ut = data["U"], data["Ut"]
        y = Ut.ravel()
        nnz = len(true_set)

        for lib_label, lib_cfg in LIBRARIES:
            lib = FeatureLibrary(**lib_cfg)
            Theta, names = lib.build(U)
            P = len(names)
            if len(true_set & set(names)) < len(true_set):
                continue  # skip incomplete libraries

            for method_name, method_kw in METHODS:
                result_key = f"{key}|{lib_label}|{method_name}"
                if result_key in results:
                    continue  # already computed

                t0 = time.time()
                try:
                    kw = dict(method_kw)
                    # Set OMP sparsity
                    if kw.get("method") == "omp":
                        kw["n_nonzero"] = nnz
                    oid = OperatorIdentifier(**kw)
                    res = oid.fit(Theta, y, names)
                    J = jaccard_score(res.names, true_set)
                    dt = time.time() - t0
                    results[result_key] = {"J": J, "support": list(res.names),
                                            "time": dt, "P": P}
                except Exception as e:
                    dt = time.time() - t0
                    results[result_key] = {"J": -1, "error": str(e)[:100], "time": dt, "P": P}

                # Save incrementally
                with open(RESULT_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  {result_key}  J={results[result_key].get('J',-1):.2f} {dt:.1f}s", flush=True)

# ── Task 3: Summarize ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Task 3: Summary")
print("=" * 60)

# Aggregate by (PDE, P, Method) → J=1 rate
summary = {}
for rkey, rval in results.items():
    parts = rkey.split("|")
    if len(parts) != 3:
        continue
    pde_seed, lib_label, method_name = parts
    pde_name = pde_seed.rsplit("_seed", 1)[0]
    J = rval.get("J", -1)
    tag = f"{pde_name}|{lib_label}|{method_name}"
    if tag not in summary:
        summary[tag] = {"total": 0, "J1": 0, "fails": 0}
    summary[tag]["total"] += 1
    if J == 1.0:
        summary[tag]["J1"] += 1
    elif J < 0:
        summary[tag]["fails"] += 1

with open(RESULT_FILE.replace(".json", "_summary.json"), "w") as f:
    out = {}
    for tag, counts in sorted(summary.items()):
        pde, plab, method = tag.split("|", 2)
        if pde not in out:
            out[pde] = {}
        if plab not in out[pde]:
            out[pde][plab] = {}
        t = counts["total"]
        out[pde][plab][method] = f"{counts['J1']}/{t} ({counts['J1']/t*100:.0f}%) fail={counts['fails']}"
    json.dump(out, f, indent=2)

print(f"\nResults: {RESULT_FILE}")
print(f"Summary: {RESULT_FILE.replace('.json','_summary.json')}")

# Print readable summary
for pde in sorted(out):
    print(f"\n{pde}:")
    for plab in ["S", "M", "L"]:
        if plab in out[pde]:
            print(f"  {plab}:")
            for method in sorted(out[pde][plab]):
                print(f"    {method:<12s}: {out[pde][plab][method]}")
