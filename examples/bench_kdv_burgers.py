#!/usr/bin/env python3
"""Benchmark KdV-Burgers: u_t = 0.5 u_xx - u u_x - (1/6) u_xxx."""
from benchmark_common import run_benchmark
from opid import PDESimulator
sim = PDESimulator.custom("0.5*D[D[u]] - u*D[u] - (1.0/6.0)*D[D[D[u]]]", name="KdVB",
                           N=256, T=0.05, n_t=100, n_modes=8, seed=42, backend="numpy")
run_benchmark("KdV-Burgers", sim, {"u_xx", "u u_x", "u_xxx"})
