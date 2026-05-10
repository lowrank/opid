#!/usr/bin/env python3
"""Benchmark Burgers: u_t = -u u_x + 0.05 u_xx."""
from benchmark_common import run_benchmark
from opid import PDESimulator
sim = PDESimulator.burgers(nu=0.05, N=256, T=0.05, n_t=100, n_modes=8, seed=42, backend="numpy")
run_benchmark("Burgers", sim, {"u u_x", "u_xx"})
