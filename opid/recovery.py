"""
OperatorIdentifier — sparse recovery of PDE coefficients.

Three complementary algorithms are exposed through a single sklearn-style API:

Method 1 — ``'omp'``
    Orthogonal Matching Pursuit.  Greedy, fast, robust to moderate noise.
    Requires an explicit sparsity target ``n_nonzero``.

Method 2 — ``'lasso'``
    L1-regularised least squares.  Convex, globally optimal, but suffers
    shrinkage bias and can activate spurious terms under noise.

Method 3 — ``'l0_pareto'``
    Mixed-Integer L0 minimisation with a Pareto sweep over the error
    tolerance epsilon.  A *stable plateau* in the (sparsity, epsilon) curve
    is taken as the identified support; a final OLS step debiases the
    coefficients.  Most accurate but most expensive.

All methods return a :class:`RecoveryResult` with a uniform interface.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════ #
#  Result container                                                       #
# ═══════════════════════════════════════════════════════════════════════ #


@dataclass
class RecoveryResult:
    """Structured result from operator identification.

    Attributes
    ----------
    method : str
        Name of the recovery method used.
    coef : ndarray, shape (P,)
        Identified coefficient vector (zero for inactive terms).
    support : list of int
        Indices of the active (non-zero) terms.
    names : list of str
        Feature names corresponding to the active support.
    active_coef : list of float
        Coefficient values for the active terms.
    residual : float
        L2 residual ||y - Theta @ coef||_2 on the full data.
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


# ═══════════════════════════════════════════════════════════════════════ #
#  Main class                                                             #
# ═══════════════════════════════════════════════════════════════════════ #


class OperatorIdentifier:
    """
    Identify the sparse coefficient vector of a nonlinear PDE operator.

    Parameters
    ----------
    method : {'omp', 'lasso', 'l0_pareto'}
        Recovery algorithm.
    n_nonzero : int or None
        Target sparsity for OMP.  Ignored by other methods.
    alpha : float
        Regularisation strength for Lasso.
        n_eps : int
            Number of epsilon grid points for the L0 Pareto sweep.
        eps_factor_hi : float
            Upper bound for epsilon sweep as a multiple of the noise floor.
        eps_factor_lo : float
            Lower bound for epsilon sweep as a multiple of the noise floor.
        max_samples : int
            Maximum number of randomly selected rows used in the MILP subproblem
            (reduces wall time for the L0 method; full data is used for OLS).
        milp_solver : str
            CVXPY solver name for the MILP (default ``'GLPK_MI'``).
        threshold_coef : float or None
            Coefficient truncation threshold after OLS debiasing.
            Entries with |ξ_i| < threshold_coef·max(|ξ|) are pruned.
            Default: 1e-10 for clean data, 1e-4 for noisy data.
        max_rounds : int
            Maximum refinements.  Each round uses the previous round's
            OLS L1 residual to narrow (or widen) the epsilon range.
        feature_names : list of str or None
            Optional human-readable feature names.
        random_state : int or None
            Seed for sub-sampling in L0 Pareto.
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
        verbose: bool = False,
    ):
        if method not in ("omp", "lasso", "l0_pareto"):
            raise ValueError(f"Unknown method '{method}'. Choose 'omp', 'lasso', or 'l0_pareto'.")
        self.method = method
        self.n_nonzero = n_nonzero
        self.alpha = alpha
        self.n_eps = n_eps
        self.eps_factor_hi = eps_factor_hi
        self.eps_factor_lo = eps_factor_lo
        self.max_samples = max_samples
        self.milp_solver = milp_solver
        self.feature_names = feature_names
        self.random_state = random_state
        self.threshold_coef = threshold_coef
        self.max_rounds = max_rounds
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        Theta: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> RecoveryResult:
        """
        Identify the operator from feature matrix and target vector.

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
            return self._fit_omp(Theta, y, names)
        elif self.method == "lasso":
            return self._fit_lasso(Theta, y, names)
        else:
            return self._fit_l0_pareto(Theta, y, names)

    # ------------------------------------------------------------------ #
    #  Method implementations                                             #
    # ------------------------------------------------------------------ #

    def _fit_omp(self, Theta, y, names) -> RecoveryResult:
        # Column-normalise for OMP (OMP is sensitive to feature scale)
        col_norms = np.linalg.norm(Theta, axis=0)
        col_norms[col_norms < 1e-14] = 1.0
        Theta_n = Theta / col_norms

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero)
        omp.fit(Theta_n, y)
        coef_n = omp.coef_
        # Rescale back to original space
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

    def _fit_lasso(self, Theta, y, names) -> RecoveryResult:
        """Column-normalised Lasso with cross-validated α and thresholding.

        Steps:
          1. Column-normalise Θ → Θ̃ (unit ℓ² norm) so all features are
             on a comparable ``importance'' scale, preventing small-physical-
             coefficient terms (e.g., ν u_xx with ν=0.05) from being
             drowned by large-coefficient terms.
          2. Fit Lasso on Θ̃ with α chosen by 5-fold cross-validation.
          3. Threshold |coef| < τ · max(|coef|) with τ = 10⁻³ to prune
             spurious terms.
          4. OLS debias on the pruned support (original-scale Θ).
        """
        from sklearn.linear_model import LassoCV

        col_norms = np.linalg.norm(Theta, axis=0)
        col_norms[col_norms < 1e-14] = 1.0
        Theta_n = Theta / col_norms

        lasso_cv = LassoCV(cv=5, max_iter=20000, fit_intercept=False,
                           n_alphas=50, random_state=0)
        lasso_cv.fit(Theta_n, y)
        coef_n = lasso_cv.coef_

        # Relative threshold at 10⁻³ of max |coef|
        max_abs = float(np.max(np.abs(coef_n)))
        thresh = max_abs * 1e-3 if max_abs > 1e-10 else 1e-10
        support = [i for i, c in enumerate(coef_n) if np.abs(c) > thresh]
        if not support:
            support = [int(np.argmax(np.abs(coef_n)))]

        # OLS debiasing on original scale
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

    def _fit_l0_pareto(self, Theta, y, names) -> RecoveryResult:
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError("cvxpy is required for the 'l0_pareto' method.") from e

        rng = np.random.default_rng(self.random_state)
        n = len(y)
        m = min(self.max_samples, n)
        idx = rng.choice(n, m, replace=False)
        Ts = Theta[idx]
        ys = y[idx]

        # Column-normalise for numerical stability
        col_scales = np.linalg.norm(Ts, axis=0, keepdims=True)
        col_scales[col_scales < 1e-14] = 1.0
        Ts_norm = Ts / col_scales

        # Noise floor from OLS on normalized system
        try:
            xi_ls, _, _, _ = np.linalg.lstsq(Ts_norm, ys, rcond=None)
            ols_residual = float(np.linalg.norm(ys - Ts_norm @ xi_ls, 1))
        except np.linalg.LinAlgError:
            # lstsq can fail on ill-conditioned matrices; fall back to
            # a regularized solve and accept a larger noise floor.
            reg = 1e-6 * np.eye(Ts_norm.shape[1])
            xi_ls = np.linalg.solve(Ts_norm.T @ Ts_norm + reg, Ts_norm.T @ ys)
            ols_residual = float(np.linalg.norm(ys - Ts_norm @ xi_ls, 1))

        # Base scale: use the L1 norm of the FULL data (not subsample)
        # so the eps range is consistent across rounds.
        # With eps_factor_hi ~ 200, eps_hi ≈ 20 × ||y||_1, loose enough
        # that even xi=0 is feasible.
        y_scale_full = float(np.linalg.norm(y, 1))
        noise_floor = y_scale_full * 0.1

        # Big-M: use 2× the OLS bound with a floor of 1 to avoid trivial relaxation
        M_tight = max(2.0 * float(np.max(np.abs(xi_ls))), 1.0)

        P = Ts_norm.shape[1]
        xi_var  = cp.Variable(P)
        z_var   = cp.Variable(P, boolean=True)
        eps_param = cp.Parameter(nonneg=True)

        prob = cp.Problem(
            cp.Minimize(cp.sum(z_var)),
            [
                cp.norm(ys - Ts_norm @ xi_var, 1) <= eps_param,
                xi_var <=  M_tight * z_var,
                xi_var >= -M_tight * z_var,
            ],
        )

        # ── Bisection-based Pareto sweep ──────────────────────────────
        eps_hi = noise_floor * self.eps_factor_hi
        eps_lo = noise_floor * self.eps_factor_lo

        def _solve(eps_val):
            eps_param.value = eps_val
            try:
                prob.solve(solver=getattr(cp, self.milp_solver), warm_start=False)
            except Exception:
                return None, None
            if prob.status in ("optimal", "optimal_inaccurate"):
                supp = frozenset(i for i in range(P) if float(z_var.value[i]) > 0.5)
                return supp, xi_var.value / col_scales.flatten()
            return None, None

        # Exponential search: if eps_lo is infeasible, double until feasible
        sl, _ = _solve(eps_lo)
        while sl is None and eps_lo < eps_hi * 0.9:
            eps_lo *= 2.0
            sl, _ = _solve(eps_lo)
            if self.verbose:
                print(f"  eps_lo → {eps_lo:.3e} {'feasible' if sl is not None else 'infeasible'}")

        sh, _ = _solve(eps_hi)
        k_hi = len(sh) if sh is not None else 0
        k_lo = len(sl) if sl is not None else P
        if self.verbose:
            print(f"  eps [{eps_lo:.1e}, {eps_hi:.1e}]  k_lo={k_lo}  k_hi={k_hi}")

        # Collect: for each k, the smallest eps where k is feasible
        frontier = {k_lo: eps_lo, k_hi: eps_hi}

        for k in range(k_hi + 1, k_lo):
            # left  = tight eps where k is NOT feasible (sparsity > k)
            # right = loose eps where k IS  feasible (sparsity ≤ k)
            left  = frontier.get(k + 1, eps_lo)   # need >k terms here
            right = frontier.get(k - 1, eps_hi)   # feasible with ≤k here
            if left >= right:
                frontier[k] = right
                continue
            for _ in range(self.n_eps // max(k_lo - k_hi, 1)):
                mid = np.sqrt(left * right)
                sm, _ = _solve(mid)
                if sm is None: break
                km = len(sm)
                if self.verbose:
                    print(f"  k={k} mid={mid:.3e} → k={km}")
                if km <= k:
                    right = mid  # feasible, try tighter (lower eps)
                else:
                    left  = mid  # too many terms, loosen (higher eps)
                if right / left < 1.02: break
            frontier[k] = right

        # Re-solve at each frontier eps to get the actual support
        pairs = []
        for eps in sorted(frontier.values()):
            sm, _ = _solve(eps)
            if sm is not None:
                pairs.append((eps, sm))

        # Sort by eps ascending, filter out converged points (supp=None)
        pairs = [(e, s) for e, s in pairs if s is not None]
        pairs.sort(key=lambda p: p[0])
        if not pairs:
            return RecoveryResult(
                method="l0_pareto", coef=xi_ls / col_scales.flatten(),
                support=list(range(P)), names=names,
                active_coef=list(xi_ls / col_scales.flatten()),
                residual=float(np.linalg.norm(y - Theta @ (xi_ls / col_scales.flatten()))),
                meta={"warning": "no_feasible_milp"})

        valid_eps    = [p[0] for p in pairs]
        supports_raw = [p[1] for p in pairs]

        if self.verbose:
            print(f"  {len(pairs)} distinct sparsity levels found")

        runs = self._find_plateau_runs(supports_raw, valid_eps)

        if self.verbose and len(runs) > 1:
            for s, e, supp in runs:
                print(f"    k={len(supp)}  eps=[{valid_eps[s]:.3e}, {valid_eps[e]:.3e}]  len={e-s+1}")

        # Select: smallest eps lower bound (tightest tolerance), skip k=0.
        non_trivial = [(s, e, supp) for s, e, supp in runs if len(supp) > 0]
        if not non_trivial:
            non_trivial = runs
        # Prefer the plateau with the smallest eps_lower — this is the
        # sparsest model that survives at the tightest tolerance.
        rec_s, rec_e, rec_supp = min(non_trivial, key=lambda r: valid_eps[r[0]])

        if self.verbose:
            print(f"  selected: k={len(rec_supp)}  "
                  f"eps=[{valid_eps[rec_s]:.3e}, {valid_eps[rec_e]:.3e}]")

        # OLS debiasing on FULL data
        rec_cols = sorted(rec_supp)
        if rec_cols:
            ols_coef, _, _, _ = np.linalg.lstsq(Theta[:, rec_cols], y, rcond=None)
        else:
            ols_coef = np.array([])

        # Threshold truncation
        thresh = self.threshold_coef if self.threshold_coef is not None else 1e-4
        coef = np.zeros(P)
        for j, col in enumerate(rec_cols):
            coef[col] = float(ols_coef[j])

        if thresh > 0 and rec_cols:
            max_abs = float(np.max(np.abs(coef)))
            if max_abs > 0:
                keep = [c for c in rec_cols if abs(coef[c]) > thresh * max_abs]
                if 1 <= len(keep) < len(rec_cols):
                    ols_new, _, _, _ = np.linalg.lstsq(Theta[:, keep], y, rcond=None)
                    coef = np.zeros(P)
                    for j, col in enumerate(keep):
                        coef[col] = float(ols_new[j])
                    rec_cols = keep

        residual = float(np.linalg.norm(y - Theta @ coef))
        return RecoveryResult(
            method="l0_pareto",
            coef=coef,
            support=rec_cols,
            names=[names[i] for i in rec_cols],
            active_coef=[float(coef[i]) for i in rec_cols],
            residual=residual,
            meta={
                "n_feasible": len(valid_eps),
                "plateau_eps_range": (valid_eps[rec_s], valid_eps[rec_e]),
                "all_runs": [(s, e, [names[i] for i in supp])
                             for s, e, supp in self._find_plateau_runs(supports_raw, valid_eps)],
                "M_tight": M_tight,
                "noise_floor": noise_floor,
            },
        )

    @staticmethod
    def _find_plateau_runs(supports, valid_eps):
        runs = []
        cur_supp = supports[0]
        run_start = 0
        for i, supp in enumerate(supports):
            if supp != cur_supp:
                runs.append((run_start, i - 1, cur_supp))
                cur_supp = supp
                run_start = i
        runs.append((run_start, len(supports) - 1, cur_supp))
        return runs
