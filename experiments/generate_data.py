#!/usr/bin/env python3
"""Generate simulation data for all 7 PDEs × N_SEEDS seeds.
Saves to benchmark_data/ as {PDE}_seed{seed}.npz files.

Usage: python experiments/generate_data.py [--seeds N] [--pde NAME]
"""

import os, argparse, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
from opid import PDESimulator

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmark_data")
os.makedirs(DATA_DIR, exist_ok=True)

PDES = [
    ("KdV",     lambda s: PDESimulator.kdv(N=128, T=0.05, n_t=100, n_modes=4, seed=s, backend="numpy")),
    ("Burgers", lambda s: PDESimulator.burgers(nu=0.05, N=128, T=0.3, n_t=100, n_modes=4, seed=s, backend="numpy")),
    ("AC",      lambda s: PDESimulator.allen_cahn(eps=0.01, N=128, T=0.1, n_t=100, n_modes=4, seed=s, backend="numpy")),
    ("KS",      lambda s: PDESimulator.ks(N=128, T=0.08, n_t=100, n_modes=4, seed=s, backend="numpy")),
    ("FKPP",    lambda s: PDESimulator.fisher_kpp(D=0.1, r=1.0, N=128, T=0.2, n_t=100, n_modes=4, seed=s, backend="numpy")),
    ("KdVB",    lambda s: PDESimulator.kdv_burgers(N=128, T=0.05, n_t=100, n_modes=4, seed=s, backend="numpy")),
    ("FHN",     lambda s: PDESimulator.fitzhugh_nagumo(N=128, T=0.1, n_t=100, n_modes=4, seed=s, backend="numpy")),
]

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark simulation data")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (default: 20)")
    parser.add_argument("--pde", type=str, help="Generate only this PDE (default: all)")
    args = parser.parse_args()

    for pde_name, factory in PDES:
        if args.pde and pde_name != args.pde:
            continue
        for seed in range(args.seeds):
            fname = os.path.join(DATA_DIR, f"{pde_name}_seed{seed}.npz")
            if os.path.exists(fname):
                print(f"  {pde_name}_seed{seed}: exists, skip")
                continue
            t0 = time.time()
            sim = factory(seed)
            U, Ut = sim.run()
            np.savez_compressed(fname, U=U, Ut=Ut)
            print(f"  {pde_name}_seed{seed}: saved ({U.shape}) in {time.time()-t0:.1f}s")

    print(f"\nSaved to {DATA_DIR}/")

if __name__ == "__main__":
    main()
