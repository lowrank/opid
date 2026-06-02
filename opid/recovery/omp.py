"""Orthogonal Matching Pursuit recovery."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

from .base import BaseRecovery, RecoveryResult
from ._utils import _column_normalise
from .subsample import Subsampler, SignalQRSubsampler


class OMPRecovery(BaseRecovery):
    """Orthogonal Matching Pursuit — greedy, fast, requires sparsity target."""

    def __init__(self, n_nonzero=2, subsampler=None, max_samples=4000,
                 verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.n_nonzero = n_nonzero
        self.subsampler = subsampler
        self.max_samples = max_samples
        self.verbose = verbose

    def _fit(self, Theta, y, names):
        Theta_n, col_norms = _column_normalise(Theta)

        s = self.subsampler or SignalQRSubsampler()
        ridx = s.select(Theta_n, self.max_samples)
        Th_s, ys = Theta_n[ridx], y[ridx]

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero)
        omp.fit(Th_s, ys)
        coef_n = omp.coef_
        coef = coef_n / col_norms
        support = [i for i, c in enumerate(coef) if abs(c) > 1e-10]
        residual = float(np.linalg.norm(y - Theta @ coef))
        return RecoveryResult(
            method="omp",
            coef=coef,
            support=support,
            names=[names[i] for i in support],
            active_coef=[float(coef[i]) for i in support],
            residual=residual,
        )
