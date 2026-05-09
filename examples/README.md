# Examples

Demonstration scripts for the `opid` package.

## PDE Identification

| Script | PDE | Methods |
|--------|-----|---------|
| `kdv_identification.py` | KdV: u_t = -6 u u_x - u_xxx | OMP, LassoCV, SCIP MILP |
| `burgers_identification.py` | Burgers: u_t = -u u_x + 0.05 u_xx | OMP, LassoCV, SCIP MILP |
| `allen_cahn_identification.py` | Allen-Cahn: u_t = 0.01 u_xx + u - u³ | OMP, LassoCV, SCIP MILP |
| `ks_identification.py` | KS: u_t = -u u_x - u_xx - u_xxxx | OMP, SCIP MILP |
| `fkpp_identification.py` | Fisher-KPP: u_t = 0.01 u_xx + u(1-u) | OMP, SCIP MILP |
| `run_all.py` | All five PDEs | OMP, LassoCV |

## B-spline Smoothing

| Script | Description |
|--------|-------------|
| `bspline_smoothing.py` | B-spline pre-smoothing for noisy spatial data |

## Usage

```bash
# Single PDE with all methods
python examples/kdv_identification.py

# All five PDEs
python examples/run_all.py

# B-spline smoothing demo
python examples/bspline_smoothing.py
```
