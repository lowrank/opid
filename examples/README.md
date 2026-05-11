# Examples

Demonstration scripts for the `opid` package.

## Benchmarks

| Script | PDE |
|---|---|
| `bench_kdv.py` | KdV: u_t = -6 u u_x - u_xxx |
| `bench_kdv_burgers.py` | KdV-Burgers: u_t = 0.5 u_xx - u u_x - (1/6) u_xxx |
| `bench_burgers.py` | Burgers: u_t = -u u_x + 0.05 u_xx |
| `bench_allen_cahn.py` | Allen-Cahn: u_t = 0.01 u_xx + u - u^3 |
| `bench_ginzburg_landau.py` | Ginzburg-Landau: u_t = u_xx - u^3 + u |
| `bench_ks.py` | KS: u_t = -u u_x - u_xx - u_xxxx |
| `bench_fkpp.py` | Fisher-KPP: u_t = 0.01 u_xx + u(1-u) |
| `benchmark_all.py` | All seven PDEs sequentially |

Each benchmark runs OMP, LassoCV, MILP (SCIP + CBC), and CCP across three dictionary sizes (S=12, M=23, L=40 terms), reporting Jaccard index and wall-clock time.

## Other

| Script | Description |
|---|---|
| `gallery.py` | Classical PDE solution gallery (generates `docs/example.png`) |
| `bspline_smoothing.py` | B-spline pre-smoothing for noisy spatial data |

## Usage

```bash
python examples/bench_kdv.py              # Single PDE
python examples/benchmark_all.py          # All seven PDEs
python examples/gallery.py                # PDE gallery figure
python examples/bspline_smoothing.py      # B-spline smoothing demo
```
