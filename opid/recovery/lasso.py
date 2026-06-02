"""Lasso (L1-regularised) recovery."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso

from .base import BaseRecovery, RecoveryResult
from ._utils import _column_normalise
from .subsample import Subsampler, SignalQRSubsampler


class LassoRecovery(BaseRecovery):
    """L1-regularised least squares — convex, globally optimal, shrinkage bias."""

    def __init__(self, alpha=0.1, subsampler=None, max_samples=4000,
                 verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.subsampler = subsampler
        self.max_samples = max_samples
        self.verbose = verbose

    def _fit(self, Theta, y, names):
        Theta_n, col_norms = _column_normalise(Theta)

        s = self.subsampler or SignalQRSubsampler()
        ridx = s.select(Theta_n, self.max_samples)
        Th_s, ys = Theta_n[ridx], y[ridx]

        lasso_cv = Lasso(alpha=self.alpha, max_iter=5000, tol=1e-4, fit_intercept=False)
        lasso_cv.fit(Th_s, ys)
        coef_n = lasso_cv.coef_

        max_abs = float(np.max(np.abs(coef_n)))
        thresh = max_abs * 1e-3 if max_abs > 1e-10 else 1e-10
        support = [i for i, c in enumerate(coef_n) if np.abs(c) > thresh]
        if not support:
            support = [int(np.argmax(np.abs(coef_n)))]

        ols_coef, _, _, _ = np.linalg.lstsq(Theta[:, support], y, rcond=None)
        coef = np.zeros(len(names))
        for j, col in enumerate(support):
            coef[col] = float(ols_coef[j])

        residual = float(np.linalg.norm(y - Theta @ coef))
        return RecoveryResult(
            method="lasso",
            coef=coef,
            support=support,
            names=[names[i] for i in support],
            active_coef=[float(coef[i]) for i in support],
            residual=residual,
        )
