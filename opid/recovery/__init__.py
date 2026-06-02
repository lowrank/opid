"""
opid.recovery — OperatorIdentifier dispatcher and public exports.

Provides a unified sklearn-style facade over five sparse recovery algorithms.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np

from .base import RecoveryResult, BaseRecovery
from .omp import OMPRecovery
from .lasso import LassoRecovery
from .l0_pareto import L0ParetoRecovery
from .l0_sdp2 import L0SDP2Recovery
from .ccp import CCPRecovery

warnings.filterwarnings("ignore")


class OperatorIdentifier:
    """Identify the sparse coefficient vector of a nonlinear PDE operator.

    Parameters
    ----------
    method : {'omp', 'lasso', 'l0_pareto', 'l0_sdp2', 'ccp'}
        Recovery algorithm.
    n_nonzero : int or None
        Target sparsity for OMP.  Ignored by other methods.
    alpha : float
        Regularisation strength for Lasso.
    n_eps : int
        Number of epsilon bisection iterations for the L0 Pareto / SDP sweep.
    eps_factor_hi : float
        Upper bound for epsilon sweep as a multiple of the noise floor.
    eps_factor_lo : float
        Lower bound for epsilon sweep as a multiple of the noise floor.
    max_samples : int
        Maximum number of randomly selected rows for MILP / SDP subproblems
        (reduces wall time; full data is used for OLS).
    milp_solver : str
        CVXPY MILP solver (SCIP, CBC, or GLPK_MI).
    cluster_size : int
        Maximum features per cluster for CCP (default 8).
    threshold_coef : float or None
        Coefficient truncation threshold after OLS debiasing.
        Entries with ϟξ_iμ < threshold_coef·max(ϟξμ) are pruned.
    max_rounds : int
        Maximum refinements (L0 Pareto only).
    feature_names : list of str or None
        Optional human-readable feature names.
    random_state : int or None
        Seed for sub-sampling in L0 Pareto / SDP.
    verbose : bool
        Print progress messages (default False).
    """

    def __init__(
        self,
        method: str = "l0_pareto",
        n_nonzero: int = 2,
        alpha: float = 0.1,
        n_eps: int = 30,
        eps_factor_hi: float = 100.0,
        eps_factor_lo: float = 0.01,
        max_samples: int = 5000,
        milp_solver: str = "GLPK_MI",
        threshold_coef: Optional[float] = None,
        feature_names: Optional[List[str]] = None,
        random_state: Optional[int] = 42,
        max_rounds: int = 3,
        cluster_size: int = 8,
        adaptive_groups_lambda: float = 0.5,
        verbose: bool = False,
    ):
        if method not in ("omp", "lasso", "l0_pareto", "l0_sdp2", "ccp"):
            raise ValueError(f"Unknown method '{method}'.")
        self.method = method
        self.n_nonzero = n_nonzero
        self.alpha = alpha
        self.n_eps = n_eps
        self.eps_factor_hi = eps_factor_hi
        self.eps_factor_lo = eps_factor_lo
        self.max_samples = max_samples
        self.milp_solver = milp_solver
        self.cluster_size = cluster_size
        self.adaptive_groups_lambda = adaptive_groups_lambda
        self.feature_names = feature_names
        self.random_state = random_state
        self.threshold_coef = threshold_coef
        self.max_rounds = max_rounds
        self.verbose = verbose

    def fit(
        self,
        Theta: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> RecoveryResult:
        """Identify the operator from feature matrix and target vector.

        Parameters
        ----------
        Theta : ndarray, shape (n_samples, P)
            Feature / library matrix.
        y : ndarray, shape (n_samples,)
            Target vector (e.g., u_t flattened).
        feature_names : list of str or None
            Override instance-level feature names.

        Returns
        -------
        RecoveryResult
        """
        names = feature_names or self.feature_names or [f"f{i}" for i in range(Theta.shape[1])]

        if self.method == "omp":
            rec = OMPRecovery(n_nonzero=self.n_nonzero, verbose=self.verbose)
        elif self.method == "lasso":
            rec = LassoRecovery(alpha=self.alpha, verbose=self.verbose)
        elif self.method == "l0_pareto":
            rec = L0ParetoRecovery(
                n_eps=self.n_eps,
                eps_factor_hi=self.eps_factor_hi,
                eps_factor_lo=self.eps_factor_lo,
                max_samples=self.max_samples,
                milp_solver=self.milp_solver,
                threshold_coef=self.threshold_coef,
                random_state=self.random_state,
                max_rounds=self.max_rounds,
                verbose=self.verbose,
            )
        elif self.method == "l0_sdp2":
            rec = L0SDP2Recovery(
                n_eps=self.n_eps,
                eps_factor_hi=self.eps_factor_hi,
                eps_factor_lo=self.eps_factor_lo,
                max_samples=self.max_samples,
                random_state=self.random_state,
                verbose=self.verbose,
            )
        elif self.method == "ccp":
            rec = CCPRecovery(
                cluster_size=self.cluster_size,
                threshold_coef=self.threshold_coef,
                verbose=self.verbose,
                adaptive_groups_lambda=self.adaptive_groups_lambda,
            )
        else:
            raise ValueError(f"Unknown method '{self.method}'.")

        return rec.fit(Theta, y, names)
