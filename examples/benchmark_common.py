#!/usr/bin/env python3
"""Shared utilities for individual PDE benchmarks."""
import numpy as np, time, os, sys
from opid import PDESimulator, FeatureLibrary, OperatorIdentifier

OUT_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results")
os.makedirs(OUT_DIR, exist_ok=True)

DICTS = [
    ("S",  2, 3, 2),   # P=12
    ("M",  2, 4, 3),   # P=23
    ("L",  3, 4, 4),   # P=40
]

def run_benchmark(name, sim, true_set):
    """Run all methods on one PDE, write results to OUT_DIR/<name>.txt."""
    out_path = os.path.join(OUT_DIR, f"{name}.txt")
    U, U_t = sim.run()

    lines = [f"# {name}  True support: {sorted(true_set)}  {time.strftime('%Y-%m-%d %H:%M:%S')}"]

    for plab, pdeg, mdd, mcross in DICTS:
        lib = FeatureLibrary(poly_degree=pdeg, max_deriv=mdd, max_cross_degree=mcross)
        Theta, names = lib.build(U)
        P = len(names)
        y = U_t.ravel()

        if len(true_set & set(names)) < len(true_set):
            lines.append(f"P={P:2d} {plab}  SKIP (true terms missing from library)")
            continue

        row = [name, plab, str(P)]

        for method, kw in [
            ("omp",   {"n_nonzero": len(true_set)}),
            ("lasso", {}),
            ("ccp", {"cluster_size": 8}),
        ]:
            t0 = time.time()
            try:
                oid = OperatorIdentifier(method=method, **kw)
                res = oid.fit(Theta, y, names)
                dt = time.time() - t0
                rec = set(names[i] for i in res.support)
                j = len(rec & true_set) / max(len(rec | true_set), 1)
            except Exception:
                dt = time.time() - t0
                j = None
            row.append(f"{j:.2f}" if j is not None else "FAIL")
            row.append(f"{dt:.1f}")

        line = "\t".join(row)
        lines.append(line)
        n_methods = (len(row) - 3) // 2
        labels = ["OMP", "Lasso", "CCP", "SCIP", "CBC"][:n_methods]
        parts = []
        for m in range(n_methods):
            parts.append(f"    {labels[m]:>5s} J={row[3+2*m]:>5s}  t={row[4+2*m]:>6s}s")
        print(f"  [{name}] P={P:2d} {plab}\n" + "\n".join(parts), flush=True)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] {name} → {out_path}", flush=True)
