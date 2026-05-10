#!/usr/bin/env python3
"""Benchmark KdV: u_t = -6 u u_x - u_xxx."""
from benchmark_common import run_benchmark
from opid import PDESimulator
sim = PDESimulator.kdv(N=256, T=0.05, n_t=100, n_modes=8, seed=42, backend="numpy")
run_benchmark("KdV", sim, {"u u_x", "u_xxx"})
