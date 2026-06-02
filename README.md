# opid — Operator Identification from PDE Trajectory Data

`opid` provides a sklearn-style API for identifying nonlinear PDE operators
from simulated or measured trajectory data.

## Installation

```bash
cd opid/
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from opid import PDESimulator, FeatureLibrary, OperatorIdentifier

# 1. Simulate KdV: u_t = -6 u u_x - u_xxx
sim = PDESimulator.kdv(N=256, T=0.05, n_t=100, n_modes=8, seed=42)
U, U_t = sim.run()

# 2. Build a feature library
lib = FeatureLibrary(poly_degree=3, max_deriv=4, max_cross_degree=4)
Theta, names = lib.build(U)

# 3. Recover the operator (CCP is the recommended default)
result = OperatorIdentifier(method="ccp").fit(Theta, U_t.ravel(), names)
print(result)
# → u_xxx: -1.00,  u u_x: -6.00
```

## Supported PDEs

| Factory method | Equation |
|----------------|----------|
| `PDESimulator.kdv()` | u_t = -6 u u_x - u_xxx |
| `PDESimulator.ks()` | u_t = -u u_x - u_xx - u_xxxx |
| `PDESimulator.burgers(nu=0.05)` | u_t = -u u_x + ν u_xx |
| `PDESimulator.allen_cahn(eps=0.01)` | u_t = ε u_xx + u - u³ |
| `PDESimulator.fisher_kpp(D=0.1, r=1.0)` | u_t = D u_xx + r u(1-u) |
| `PDESimulator.kdv_burgers(nu=0.05, beta=1/6)` | u_t = ν u_xx - u u_x - β u_xxx |
| `PDESimulator.fitzhugh_nagumo()` | u_t = u_xx + u² - u³ |
| `PDESimulator.swift_hohenberg()` | u_t = u - u³ - 2 u_xx - u_xxxx |
| `PDESimulator.nls(sigma=1.0)` | i ψ_t + ψ_xx + σ|ψ|² ψ = 0 |
| `PDESimulator.custom(rhs_str)` | Arbitrary Mathematica-string RHS |

All simulations use periodic boundary conditions on [0, 2π] with
spectral integration via scipy's `odeint` (adaptive stepping).
FFTW acceleration available when g++/fftw3 are installed.

## Recovery Methods

| Method | Key | Description |
|--------|-----|-------------|
| CCP | `'ccp'` | **Recommended default.** Correlation-cut pursuit: spectral clustering + cross-group normalized OLS voting. Handles up to P=196. |
| OMP | `'omp'` | Orthogonal Matching Pursuit — fast, column-normalised. Requires known sparsity. |
| Lasso | `'lasso'` | Column-normalised Lasso with relative threshold (0.1% of max) and OLS debiasing. |
| L0 MILP | `'l0_pareto'` | Bisection-based MILP Pareto sweep with SCIP/CBC solvers. Provably optimal for given ε. |
| L0 SDP2 | `'l0_sdp2'` | Order-2 Lasserre SDP relaxation. Tractable for P ≤ ~9. |

### Subsampling

All methods support a `Subsampler` interface for row selection:

```python
from opid.recovery.subsample import FullSubsampler, RandomSubsampler, QRSubsampler, SignalQRSubsampler

# Default: signal-aware pivoted QR (filter tail, then max-volume)
OperatorIdentifier(method="ccp")

# Explicit:
OperatorIdentifier(method="ccp", subsampler=QRSubsampler())
OperatorIdentifier(method="omp", subsampler=RandomSubsampler(seed=42))
```

| Sampler | Strategy |
|---------|----------|
| `SignalQRSubsampler` (default) | Filter top 2m rows by norm, then pivoted QR — balances signal + rank |
| `QRSubsampler` | Pivoted QR on all rows — maximises submatrix volume |
| `RandomSubsampler` | Uniform random m rows |
| `FullSubsampler` | All rows (no subsampling) |

## Architecture

```
opid/
├── simulator.py       # PDESimulator — spectral PDE integrator (10 factory methods)
├── library.py         # FeatureLibrary — candidate term dictionary
├── utils.py           # jaccard_score, add_noise, relative_error
├── recovery/          # Sparse recovery (modular package)
│   ├── __init__.py      # OperatorIdentifier — dispatcher/facade
│   ├── base.py          # RecoveryResult dataclass, BaseRecovery ABC
│   ├── ccp.py           # CCPRecovery (recommended default)
│   ├── omp.py           # OMPRecovery
│   ├── lasso.py         # LassoRecovery
│   ├── l0_pareto.py     # L0ParetoRecovery (MILP)
│   ├── l0_sdp2.py       # L0SDP2Recovery (SDP)
│   ├── subsample.py     # Subsampler hierarchy
│   └── _utils.py        # Shared helpers
├── _backend/          # SpectralEngine — spectral solver + FFTW compilation
├── _bspline/          # B-spline design matrix (Cython extension)
├── tests/             # 103 tests (1 skipped: l0_sdp2)
├── experiments/       # Benchmark scripts
├── examples/          # PDE gallery and individual benchmarks
└── benchmark_results/ # Pre-computed results
```

## Benchmarks

Full benchmarks across 7 PDEs, 16 library sizes (P=12 to P=196), 20 seeds:
```bash
python experiments/core_methods.py      # CCP, OMP, Lasso at S/M/L/XL
python experiments/ccp_scaling.py       # CCP across all library sizes
python experiments/subsample_compare.py # Compare subsampling strategies
```

Results are saved to `benchmark_results/`.

### Key Results

CCP is the only method that works across all 7 PDEs at all library sizes
(S=12, M=23, L=40, XL=59). MILP (L0 SCIP) matches CCP on KdV/Burgers/KdVB/FKPP
but is 1000× slower (15-30 min/seed vs 0.5 s) and crashes on AC/KS.

| PDE | CCP | OMP | Lasso | L0 CBC |
|-----|-----|-----|-------|--------|
| KdV | 95-100% | 95% | 20% | — |
| Burgers | 95-100% | 75-95% | 0-20% | — |
| AC | 100% | 30% | 0% | 100% |
| KS | 80-90% | 0% | 0% | 85%* |
| FKPP | 70-100% | 0% | 0% | — |
| KdVB | 95-100% | 85% | 0% | — |
| FHN | 90% | 25% | 0% | 10% |

CCP uses SignalQR subsampling (4000 rows). L0 CBC uses per-column Big-M,
subprocess-safe CoinError handling, n_eps=30, and 4000 samples.
KS M 20/20, L 14/20 at 4000 samples (was 6/40 at 500). CCP is 50-2000× faster.

SignalQR subsampling recovers 31 more seeds than full-data CCP across all benchmarks.

## Testing

```bash
python -m pytest tests/ -v           # 103 tests
python -m pytest tests/ -q           # 104 passed, 1 skipped
```

## Classical PDE Gallery

```bash
python examples/gallery.py     # generates docs/example.png
```

![Classical PDE solutions](docs/example.png)

Solves 9 classical evolution equations on [0, 2π] and plots solution snapshots.
