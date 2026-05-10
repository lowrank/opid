#!/usr/bin/env python3
"""Benchmark Ginzburg-Landau: u_t = u_xx - u^3 + u."""
from benchmark_common import run_benchmark
from opid import PDESimulator
sim = PDESimulator.custom("D[D[u]] - u^3 + u", name="GL",
                           N=256, T=0.05, n_t=100, n_modes=8, seed=42, backend="numpy")
run_benchmark("Ginzburg-Landau", sim, {"u_xx", "u^3", "u"})
