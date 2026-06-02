"""Base classes and data structures for sparse recovery methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RecoveryResult:
    """Container for the output of a sparse recovery algorithm.

    Parameters
    ----------
    method : str
        Name of the recovery method (e.g. "omp", "lasso", "ccp").
    coef : ndarray (P,)
        Full coefficient vector on the library (unselected terms are zero).
    support : list of int
        Indices of non-zero coefficients.
    names : list of str
        Human-readable feature names for the active coefficients.
    active_coef : list of float
        Values of the non-zero coefficients.
    residual : float
        L2 norm of the residual  ‖y - Θ ξ‖₂.
    meta : dict, optional
        Additional diagnostic metadata.
    """

    method: str
    coef: np.ndarray
    support: List[int]
    names: List[str]
    active_coef: List[float]
    residual: float
    meta: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"RecoveryResult [{self.method}]  residual={self.residual:.4e}"]
        for n, c in zip(self.names, self.active_coef):
            lines.append(f"  {n:>18s}: {c:+.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


class BaseRecovery:
    """Abstract base for sparse recovery methods.

    Subclasses implement ``_fit(Theta, y, names) -> RecoveryResult``.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, Theta: np.ndarray, y: np.ndarray,
            names: Optional[List[str]] = None) -> RecoveryResult:
        """Fit the recovery model.

        Parameters
        ----------
        Theta : ndarray (n, P)
            Feature library matrix.
        y : ndarray (n,)
            Time-derivative vector.
        names : list of str, optional
            Feature names.

        Returns
        -------
        RecoveryResult
        """
        if names is None:
            names = [f"f{i}" for i in range(Theta.shape[1])]
        return self._fit(Theta, y, names)

    def _fit(self, Theta: np.ndarray, y: np.ndarray,
             names: List[str]) -> RecoveryResult:
        raise NotImplementedError
