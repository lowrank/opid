#!/usr/bin/env python3
"""Benchmark Allen-Cahn: u_t = 0.01 u_xx + u - u^3."""
from benchmark_common import run_benchmark
from opid import PDESimulator
sim = PDESimulator.allen_cahn(eps=0.01, N=256, T=0.05, n_t=100, n_modes=8, seed=42, backend="numpy")
run_benchmark("Allen-Cahn", sim, {"u_xx", "u", "u^3"})
