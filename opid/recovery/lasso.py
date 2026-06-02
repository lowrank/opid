"""Lasso (L1-regularised) recovery."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso

from .base import BaseRecovery, RecoveryResult
from ._utils import _column_normalise


class LassoRecovery(BaseRecovery):
    """L1-regularised least squares — convex, globally optimal, shrinkage bias."""

    def __init__(self, alpha=0.1, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.verbose = verbose

    def _fit(self, Theta, y, names):
        Theta_n, col_norms = _column_normalise(Theta)

        lasso_cv = Lasso(alpha=self.alpha, max_iter=5000, tol=1e-4, fit_intercept=False)
        lasso_cv.fit(Theta_n, y)
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
