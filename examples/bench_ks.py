#!/usr/bin/env python3
"""Benchmark KS: u_t = -u u_x - u_xx - u_xxxx."""
from benchmark_common import run_benchmark
from opid import PDESimulator
sim = PDESimulator.ks(N=256, T=0.05, n_t=100, n_modes=8, seed=42, backend="numpy")
run_benchmark("KS", sim, {"u u_x", "u_xx", "u_xxxx"})
