#!/usr/bin/env python3
"""Run all individual PDE benchmarks sequentially."""
import subprocess, sys, os

HERE = os.path.dirname(__file__)
scripts = [
    "bench_kdv.py",
    "bench_kdv_burgers.py",
    "bench_burgers.py",
    "bench_allen_cahn.py",
    "bench_ginzburg_landau.py",
    "bench_ks.py",
    "bench_fkpp.py",
]

for s in scripts:
    print(f"\n{'='*60}\n  Running {s}\n{'='*60}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                       stdout=None, stderr=None)  # inherit parent stdout/stderr
    if r.returncode != 0:
        print(f"[FAIL] {s} exited with {r.returncode}", flush=True)

print("\nAll done. Results in examples/benchmark_results/", flush=True)
