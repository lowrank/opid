"""
opid - Operator Identification from PDE trajectory data.

Provides a unified sklearn-style API for PDE operator identification via
sparse regression on a feature library constructed from trajectory data.

Package layout
--------------
opid.simulator   — PDESimulator: FFTW-accelerated spectral solver (fallback to numpy)
opid.library     — FeatureLibrary: builds candidate term matrices
opid.recovery    — OperatorIdentifier + RecoveryResult
opid.utils       — helpers: add_noise, relative_error, print_table

Typical usage
-------------
>>> from opid import PDESimulator, FeatureLibrary, OperatorIdentifier
>>> sim  = PDESimulator.kdv(N=256, T=0.05, n_modes=8, seed=42)
>>> U, U_t = sim.run()
>>> lib  = FeatureLibrary(poly_degree=3, max_deriv=4, max_cross_degree=4)
>>> Theta, names = lib.build(U, sim.k)
>>> oid  = OperatorIdentifier(method="l0_pareto", n_eps=40)
>>> res  = oid.fit(Theta, U_t.ravel(), feature_names=names)
>>> print(res)
"""

from .simulator import PDESimulator
from .library import FeatureLibrary
from .recovery import OperatorIdentifier, RecoveryResult
from .utils import add_noise, relative_error, print_recovery_table, jaccard_score

__version__ = "0.2.0"
__all__ = [
    "PDESimulator",
    "FeatureLibrary",
    "OperatorIdentifier",
    "RecoveryResult",
    "jaccard_score",
    "add_noise",
    "relative_error",
    "print_recovery_table",
]
