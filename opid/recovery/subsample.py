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


class SignalQRSubsampler(Subsampler):
    """Signal-aware pivoted-QR row selection.

    First filters to the top 2m rows by L2 norm (removing the
    dissipative tail), then applies pivoted QR among survivors to
    select m maximally independent rows.  Balances signal strength
    with rank maximization.

    Parameters
    ----------
    ratio : float
        Keep top ``ratio * m`` rows by norm before QR (default 2.0).
    """

    def __init__(self, ratio: float = 2.0):
        self.ratio = ratio

    def _select(self, Theta: np.ndarray, m: int) -> np.ndarray:
        from scipy.linalg import qr
        n = Theta.shape[0]
        pool_size = min(int(self.ratio * m), n)
        norms = np.linalg.norm(Theta, axis=1)
        # argpartition needs kth < len; when pool_size == n, use argsort
        if pool_size < n:
            top = np.argpartition(-norms, pool_size - 1)[:pool_size]
        else:
            top = np.arange(n)
        Th_top = Theta[top]
        _, _, piv = qr(Th_top.T, pivoting=True, mode='economic')
        return np.asarray(top[piv[:m]], dtype=int)
