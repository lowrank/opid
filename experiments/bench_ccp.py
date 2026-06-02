#!/usr/bin/env python3
"""Run CCP benchmarks across 100 seeds for all 7 PDEs at P=40."""
import json, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from opid import PDESimulator, FeatureLibrary, OperatorIdentifier

N, n_t, n_modes, T = 256, 200, 8, None
NSEEDS = 100

EXPERIMENTS = [
    ("KdV",     lambda s: PDESimulator.kdv(N=N, T=0.08, n_t=n_t, n_modes=n_modes, seed=s, backend="numpy"),
     {"u u_x", "u_xxx"}),
    ("Burgers", lambda s: PDESimulator.burgers(nu=0.05, N=N, T=0.5, n_t=n_t, n_modes=n_modes, seed=s, backend="numpy"),
     {"u u_x", "u_xx"}),
    ("AC",      lambda s: PDESimulator.allen_cahn(eps=0.01, N=N, T=0.3, n_t=n_t, n_modes=6, seed=s, backend="numpy"),
     {"u", "u^3", "u_xx"}),
    ("KS",      lambda s: PDESimulator.ks(N=N, T=0.08, n_t=n_t, n_modes=n_modes, seed=s, backend="numpy"),
     {"u u_x", "u_xx", "u_xxxx"}),
    ("FKPP",    lambda s: PDESimulator.fisher_kpp(D=0.01, r=1., N=N, T=0.5, n_t=n_t, n_modes=6, seed=s, backend="numpy"),
     {"u", "u^2", "u_xx"}),
    ("GL",      lambda s: PDESimulator.custom("D[D[u]] - u^3 + u", N=N, T=0.05, n_t=n_t, n_modes=n_modes, seed=s, backend="numpy"),
     {"u_xx", "u^3", "u"}),
    ("KdV-Bg",  lambda s: PDESimulator.custom("0.5*D[D[u]] - u*D[u] - (1.0/6.0)*D[D[D[u]]]", N=N, T=0.05, n_t=n_t, n_modes=n_modes, seed=s),
     {"u_xx", "u u_x", "u_xxx"}),
]

results = {}
for name, factory, true_set in EXPERIMENTS:
    counts = {"J=1": 0, "fail": 0, "times_sim": [], "times_ccp": []}
    for s in range(NSEEDS):
        t0 = time.time()
        try:
            sim = factory(s)
            U, Ut = sim.run()
            dt_sim = time.time() - t0
            t0 = time.time()
            Th, nn = FeatureLibrary(poly_degree=3, max_deriv=4, max_cross_degree=4).build(U)
            r = OperatorIdentifier(method="ccp").fit(Th, Ut.ravel(), nn)
            dt_ccp = time.time() - t0
            inter = true_set & set(r.names)
            union = true_set | set(r.names)
            J = len(inter) / len(union) if union else 0
            if J == 1.0:
                counts["J=1"] += 1
            counts["times_sim"].append(dt_sim)
            counts["times_ccp"].append(dt_ccp)
        except Exception:
            counts["fail"] += 1
        if s % 25 == 0:
            print(f"  {name} {s}/{NSEEDS}")
    results[name] = counts

# Print summary
print("\n" + "=" * 60)
print(f"  CCP BENCHMARK  —  P=40, n_t={n_t}, {NSEEDS} seeds")
print("=" * 60)
for name in results:
    c = results[name]
    total = c["J=1"] + c["fail"]
    ok = c["J=1"]
    sim_avg = np.mean(c["times_sim"]) if c["times_sim"] else 0
    ccp_avg = np.mean(c["times_ccp"]) if c["times_ccp"] else 0
    print(f"  {name:>8s}: J=1 {ok:3d}/{total} ({ok/total*100:.0f}%)  "
          f"sim={sim_avg:.1f}s  ccp={ccp_avg:.2f}s  fail={c['fail']}")

# Save JSON
out = os.path.join(os.path.dirname(__file__), "benchmark_results", "ccp_100seeds.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
