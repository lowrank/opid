#!/usr/bin/env python3
"""Benchmark Fisher-KPP: u_t = 0.01 u_xx + u(1-u)."""
from benchmark_common import run_benchmark
from opid import PDESimulator
sim = PDESimulator.fisher_kpp(D=0.01, r=1.0, N=256, T=0.05, n_t=100, n_modes=8, seed=42, backend="numpy")
run_benchmark("FKPP", sim, {"u_xx", "u", "u^2"})
