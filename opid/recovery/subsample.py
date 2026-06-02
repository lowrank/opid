"""Row subsampling strategies for sparse recovery methods."""

from __future__ import annotations

import numpy as np


class Subsampler:
    """Base class for row subsampling strategies."""

    def select(self, Theta: np.ndarray, m: int) -> np.ndarray:
        """Return indices of m selected rows from Theta (n × P).

        Parameters
        ----------
        Theta : ndarray (n, P)
        m : int
            Number of rows to select (clamped to n).

        Returns
        -------
        idx : ndarray (m,) of int
        """
        n = Theta.shape[0]
        m = min(m, n)
        return self._select(Theta, m)

    def _select(self, Theta: np.ndarray, m: int) -> np.ndarray:
        raise NotImplementedError


class FullSubsampler(Subsampler):
    """Use all rows (no subsampling)."""

    def _select(self, Theta: np.ndarray, m: int) -> np.ndarray:
        return np.arange(m)


class RandomSubsampler(Subsampler):
    """Uniform random subsampling."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _select(self, Theta: np.ndarray, m: int) -> np.ndarray:
        n = Theta.shape[0]
        rng = np.random.default_rng(self.seed)
        return rng.choice(n, m, replace=False)


class QRSubsampler(Subsampler):
    """Pivoted-QR max-volume row selection.

    Selects the m most linearly independent rows using column-pivoted QR
    on Theta^T.  Maximises the determinant (volume) of the selected
    submatrix, giving the most information-dense subsample.
    """

    def _select(self, Theta: np.ndarray, m: int) -> np.ndarray:
        from scipy.linalg import qr
        _, _, piv = qr(Theta.T, pivoting=True, mode='economic')
        return np.asarray(piv[:m], dtype=int)
