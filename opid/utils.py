"""
utils — helper functions for the opid package.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np


def add_noise(
    Theta: np.ndarray,
    y: np.ndarray,
    noise_level: float = 0.01,
    seed: Optional[int] = None,
) -> tuple:
    """
    Add relative Gaussian noise to library and target.

    Parameters
    ----------
    Theta       : ndarray (n, P)   Feature matrix.
    y           : ndarray (n,)     Target vector.
    noise_level : float            Relative std (0.01 = 1%).
    seed        : int|None         RNG seed.

    Returns
    -------
    Theta_noisy : ndarray (n, P)
    y_noisy     : ndarray (n,)
    """
    rng = np.random.default_rng(seed)
    col_std = np.std(Theta, axis=0)
    col_std[col_std < 1e-14] = 1.0
    Theta_noisy = Theta + noise_level * col_std * rng.standard_normal(Theta.shape)
    y_std = float(np.std(y))
    y_std = y_std if y_std > 1e-14 else 1.0
    y_noisy = y + noise_level * y_std * rng.standard_normal(y.shape)
    return Theta_noisy, y_noisy


def relative_error(coef_true: np.ndarray, coef_pred: np.ndarray) -> float:
    """
    Relative L2 error in recovered coefficients:
      ||ξ_true - ξ_pred||₂ / ||ξ_true||₂
    """
    denom = np.linalg.norm(coef_true)
    if denom < 1e-14:
        return float(np.linalg.norm(coef_pred))
    return float(np.linalg.norm(coef_true - coef_pred) / denom)


def jaccard_score(found: List[str], true: List[str]) -> float:
    """
    Jaccard index for support recovery.

    J = |A ∩ B| / |A ∪ B|,  where A = set(true), B = set(found).

    Returns
    -------
    float in [0, 1].  1.0 = perfect recovery, 0 = empty intersection.
    """
    A = set(true)
    B = set(found)
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union > 0 else 0.0


def print_recovery_table(
    results: list,
    true_names: List[str],
    true_coefs: List[float],
    title: str = "",
) -> None:
    """
    Print a comparison table of multiple RecoveryResult objects.

    Parameters
    ----------
    results    : list of RecoveryResult
    true_names : list of str    True active feature names.
    true_coefs : list of float  True coefficient values.
    title      : str            Optional table title.
    """
    SEP = "─" * 72
    if title:
        print(f"\n{'═'*72}")
        print(f"  {title}")
        print(f"{'═'*72}")
    print(f"  True model:")
    for n, c in zip(true_names, true_coefs):
        print(f"    {n:>20s}:  {c:+.6f}")
    print(SEP)
    for res in results:
        print(f"  [{res.method}]  residual = {res.residual:.4e}")
        for n, c in zip(res.names, res.active_coef):
            flag = "✓" if n in true_names else "✗"
            print(f"    {flag} {n:>20s}:  {c:+.6f}")
        print(SEP)
