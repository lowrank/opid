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
    method : {'omp', 'lasso', 'l0_pareto', 'l0_sdp'}
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
        cluster_size: int = 8,
        verbose: bool = False,
    ):
        if method not in ("omp", "lasso", "l0_pareto", "l0_sdp", "l0_sdp2", "l0_sdp2p"):
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
        elif self.method == "l0_sdp":
            return self._fit_l0_sdp(Theta, y, names)
        elif self.method == "l0_sdp2":
            return self._fit_l0_sdp2(Theta, y, names)
        elif self.method == "l0_sdp2p":
            return self._fit_l0_sdp2_pursuit(Theta, y, names)
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

    def _fit_l0_sdp(self, Theta, y, names) -> RecoveryResult:
        """L0 via Lasserre order-1 SDP relaxation with L2 constraint.

        Lifts the L0 problem into a (2P+1)×(2P+1) moment matrix X ≽ 0.
        Binary (z_i² = z_i) and complementarity (ξ_i z_i = 0) constraints
        are linear in X.  A Pareto sweep over ε selects the support via
        plateau detection, followed by OLS debiasing on the full data.
        """
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError("cvxpy is required for the 'l0_sdp' method.") from e

        rng = np.random.default_rng(self.random_state)
        n = len(y)
        m = min(self.max_samples, n)
        idx = rng.choice(n, m, replace=False)
        Ts = Theta[idx]
        ys = y[idx]

        P = Ts.shape[1]
        N = 1 + 2 * P
        a = Ts.T @ ys
        Q = Ts.T @ Ts
        c2 = float(ys @ ys)

        try:
            xi_ols, _, _, _ = np.linalg.lstsq(Ts, ys, rcond=None)
            ols_res = float(np.linalg.norm(ys - Ts @ xi_ols))
        except np.linalg.LinAlgError:
            xi_ols = np.zeros(P)
            ols_res = float(np.linalg.norm(ys))
        y_scale = float(np.linalg.norm(y, 1))
        noise_floor = max(y_scale * 0.1, ols_res * 0.5)

        X = cp.Variable((N, N), PSD=True)
        eps_param = cp.Parameter(nonneg=True)

        constraints = [X[0, 0] == 1.0]
        for i in range(P):
            j = 1 + P + i
            constraints.append(X[j, 0] >= 0)
            constraints.append(X[j, 0] <= 1)
            constraints.append(X[j, 0] == X[j, j])
        for i in range(P):
            constraints.append(X[1 + i, 1 + P + i] == 0)

        lin = 2.0 * cp.sum(cp.multiply(a, X[1:1+P, 0]))
        quad = cp.trace(Q @ X[1:1+P, 1:1+P])
        constraints.append(c2 - lin + quad <= eps_param)

        obj = cp.Minimize(cp.sum(X[1+P:1+2*P, 0]))
        prob = cp.Problem(obj, constraints)

        eps_hi = noise_floor * self.eps_factor_hi
        eps_lo = noise_floor * self.eps_factor_lo

        def _solve(eps_val):
            eps_param.value = eps_val
            try:
                prob.solve(solver=cp.SCS, warm_start=False)
            except Exception:
                return None, None
            if prob.status in ("optimal", "optimal_inaccurate"):
                zv = X[1+P:1+2*P, 0].value
                if zv is None:
                    return None, None
                supp = frozenset(i for i in range(P) if float(zv[i]) < 0.5)
                xv = X[1:1+P, 0].value
                xi = np.array(xv).flatten() if xv is not None else np.zeros(P)
                return supp, xi
            return None, None

        sl, _ = _solve(eps_lo)
        while sl is None and eps_lo < eps_hi * 0.9:
            eps_lo *= 2.0
            sl, _ = _solve(eps_lo)
            if self.verbose:
                print(f"  [sdp] eps_lo → {eps_lo:.3e} {'✓' if sl else '✗'}")

        sh, _ = _solve(eps_hi)
        k_hi = len(sh) if sh is not None else 0
        k_lo = len(sl) if sl is not None else P
        if self.verbose:
            print(f"  [sdp] eps [{eps_lo:.1e}, {eps_hi:.1e}]  k_lo={k_lo}  k_hi={k_hi}")

        frontier = {k_lo: eps_lo, k_hi: eps_hi}
        for k in range(k_hi + 1, k_lo):
            left  = frontier.get(k + 1, eps_lo)
            right = frontier.get(k - 1, eps_hi)
            if left >= right:
                frontier[k] = right; continue
            for _ in range(max(self.n_eps // max(k_lo - k_hi, 1), 3)):
                mid = np.sqrt(left * right)
                sm, _ = _solve(mid)
                if sm is None: break
                if len(sm) <= k: right = mid
                else:             left  = mid
                if right / left < 1.02: break
            frontier[k] = right

        pairs = []
        for eps in sorted(set(frontier.values())):
            sm, _ = _solve(eps)
            if sm is not None:
                pairs.append((eps, sm))
        pairs.sort(key=lambda p: p[0])
        if not pairs:
            return RecoveryResult(method="l0_sdp", coef=np.zeros(P),
                support=[], names=[], active_coef=[],
                residual=float(np.linalg.norm(y)),
                meta={"warning": "no_feasible_sdp"})

        valid_eps = [p[0] for p in pairs]
        supports_raw = [p[1] for p in pairs]
        if self.verbose:
            print(f"  [sdp] {len(pairs)} sparsity levels")

        runs = self._find_plateau_runs(supports_raw, valid_eps)
        non_trivial = [(s, e, supp) for s, e, supp in runs if len(supp) > 0] or runs
        rec_s, rec_e, rec_supp = min(non_trivial, key=lambda r: valid_eps[r[0]])

        if self.verbose:
            print(f"  [sdp] selected: k={len(rec_supp)}  "
                  f"eps=[{valid_eps[rec_s]:.3e}, {valid_eps[rec_e]:.3e}]")

        rec_cols = sorted(rec_supp)
        if rec_cols:
            ols_coef, _, _, _ = np.linalg.lstsq(Theta[:, rec_cols], y, rcond=None)
        else:
            ols_coef = np.array([])
        coef = np.zeros(P)
        for j, col in enumerate(rec_cols):
            coef[col] = float(ols_coef[j])

        return RecoveryResult(method="l0_sdp", coef=coef,
            support=rec_cols, names=[names[i] for i in rec_cols],
            active_coef=[float(coef[i]) for i in rec_cols],
            residual=float(np.linalg.norm(y - Theta @ coef)),
            meta={"plateau_eps_range": (valid_eps[rec_s], valid_eps[rec_e])},
        )

    def _fit_l0_sdp2(self, Theta, y, names) -> RecoveryResult:
        """L0 via Lasserre order-2 SDP relaxation with L2 constraint.

        Extends the order-1 moment matrix (degree-1 monomial basis) to a
        degree-2 monomial basis with ~C(2P+2,2) entries, providing a
        tighter relaxation of the binary and complementarity constraints.
        Localising matrices of order-1 enforce z_i²=z_i, ξ_i z_i=0, and
        the L2 bound.  Pareto sweep + OLS as in ``l0_sdp``.
        """
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError("cvxpy is required for the 'l0_sdp2' method.") from e

        rng = np.random.default_rng(self.random_state)
        n = len(y)
        m = min(self.max_samples, n)
        idx = rng.choice(n, m, replace=False)
        Ts = Theta[idx]
        ys = y[idx]

        P = Ts.shape[1]
        nv = 2 * P
        a = Ts.T @ ys
        Q = Ts.T @ Ts
        c2 = float(ys @ ys)

        # ── monomial helpers ─────────────────────────────────────────
        def _mons(dim, deg):
            """All multi-indices of length *dim* with sum ≤ *deg*."""
            out = []
            for d in range(deg + 1):
                stack = [([], d, dim - 1)]
                while stack:
                    pref, rem, k = stack.pop()
                    if k == 0:
                        out.append(tuple(pref + [rem]))
                    else:
                        for e in range(rem + 1):
                            stack.append((pref + [e], rem - e, k - 1))
            return out

        d1 = _mons(nv, 1)                     # degree ≤ 1  (N1 = nv+1)
        d2 = _mons(nv, 2)                     # degree ≤ 2  (N2)
        d4_map = {t: i for i, t in enumerate(_mons(nv, 4))}  # degree ≤ 4 → idx

        N2 = len(d2)
        N1 = len(d1)
        N4 = len(d4_map)

        _xi = lambda i: i
        _zi = lambda i: P + i
        _add = lambda t1, t2: tuple(t1[k] + t2[k] for k in range(nv))

        # noise floor
        try:
            xi_ols, _, _, _ = np.linalg.lstsq(Ts, ys, rcond=None)
            ols_res = float(np.linalg.norm(ys - Ts @ xi_ols))
        except np.linalg.LinAlgError:
            ols_res = float(np.linalg.norm(ys))
        noise_floor = max(float(np.linalg.norm(y, 1)) * 0.1, ols_res * 0.5)

        # ── SDP variables ────────────────────────────────────────────
        yy = cp.Variable(N4)
        eps_param = cp.Parameter(nonneg=True)
        cons = [yy[d4_map[(0,) * nv]] == 1.0]

        # moment matrix M₂ ≽ 0  (indexed by d2)
        m2 = {}
        for r, mr in enumerate(d2):
            for c, mc in enumerate(d2):
                k = d4_map.get(_add(mr, mc))
                if k is not None:
                    m2[(r, c)] = m2.get((r, c), 0) + yy[k]
        M2 = cp.bmat([[m2.get((r, c), 0) for c in range(N2)] for r in range(N2)])
        cons.append(M2 >> 0)

        # helper: build M₁(g·y) localising matrix for constraint polynomial g
        def _m1_matrix(poly_coeffs):
            """poly_coeffs: dict {monomial_tuple → scalar_coefficient}.
               Returns an N1×N1 cvxpy expression matrix for M₁(g·y)."""
            mat = [[0] * N1 for _ in range(N1)]
            for mr, mono_c in zip(d1, range(N1)):
                for mc, mono_r in zip(d1, range(N1)):
                    expr = 0
                    for mono_g, coeff in poly_coeffs.items():
                        k = d4_map.get(_add(_add(mr, mc), mono_g))
                        if k is not None:
                            expr += coeff * yy[k]
                    mat[mono_r][mono_c] = expr
            return cp.bmat(mat)

        # binary:  z_i - z_i² = 0
        z0 = tuple(0 for _ in range(nv))
        for i in range(P):
            g = {z0: 0}  # zero polynomial start
            zi_mono = tuple(1 if k == _zi(i) else 0 for k in range(nv))
            zi2_mono = tuple(2 if k == _zi(i) else 0 for k in range(nv))
            g[zi_mono] = g.get(zi_mono, 0) + 1
            g[zi2_mono] = g.get(zi2_mono, 0) - 1
            Mi = _m1_matrix(g)
            cons.append(Mi >> 0)
            cons.append(cp.trace(Mi) == 0)

        # complementarity:  ξ_i z_i = 0
        for i in range(P):
            g = {}
            xz_mono = tuple((1 if k == _xi(i) else 0) + (1 if k == _zi(i) else 0) for k in range(nv))
            g[xz_mono] = 1
            Mi = _m1_matrix(g)
            cons.append(Mi >> 0)
            cons.append(cp.trace(Mi) == 0)

        # L2 constraint:  ε - c2 + 2aᵀξ - ξᵀQξ ≥ 0
        # Polynomial  h(ξ,z) = ε + (2aᵀξ - c2 - ξᵀQξ)
        # M₁(h·y) = ε·M₁(1·y) + M₁((2aᵀξ - c2 - ξᵀQξ)·y)
        # M₁(1·y) has entry (r,c) = y_{mr·mc}  (just the moment variable)
        # Build the non-ε part via the helper, then add ε on the diagonal.
        g_nop = {z0: -c2}
        for j, aj in enumerate(a):
            if abs(aj) > 1e-14:
                xi_mono = tuple(1 if k == _xi(j) else 0 for k in range(nv))
                g_nop[xi_mono] = g_nop.get(xi_mono, 0) + 2.0 * aj
        for i in range(P):
            for j in range(P):
                if abs(Q[i, j]) > 1e-14:
                    xij_mono = tuple((1 if k == _xi(i) else 0) + (1 if k == _xi(j) else 0) for k in range(nv))
                    g_nop[xij_mono] = g_nop.get(xij_mono, 0) - Q[i, j]

        # Build M₁(h·y) manually so we can inject ε on the diagonal
        m1_l2 = [[0] * N1 for _ in range(N1)]
        for r, mr in enumerate(d1):
            for c, mc in enumerate(d1):
                expr = 0
                # non-ε part
                for mono_g, coeff in g_nop.items():
                    k = d4_map.get(_add(_add(mr, mc), mono_g))
                    if k is not None:
                        expr += coeff * yy[k]
                # ε part: ε * y_{mr·mc}
                expr += eps_param * yy[d4_map[_add(mr, mc)]]
                m1_l2[r][c] = expr
        cons.append(cp.bmat(m1_l2) >> 0)

        # objective:  Σ z_i
        obj = cp.Minimize(cp.sum([yy[d4_map[tuple(1 if k == _zi(i) else 0 for k in range(nv))]] for i in range(P)]))
        prob = cp.Problem(obj, cons)

        # ── Pareto sweep ─────────────────────────────────────────────
        eps_hi = noise_floor * self.eps_factor_hi
        eps_lo = noise_floor * self.eps_factor_lo

        def _solve(eps_val):
            eps_param.value = eps_val
            try:
                prob.solve(solver=cp.SCS, warm_start=False, max_iters=5000)
            except Exception:
                return None, None
            if prob.status in ("optimal", "optimal_inaccurate"):
                zv = [float(yy[d4_map[tuple(1 if k == _zi(i) else 0 for k in range(nv))]].value or 1)
                      for i in range(P)]
                supp = frozenset(i for i in range(P) if zv[i] < 0.5)
                xv = np.array([float(yy[d4_map[tuple(1 if k == _xi(j) else 0 for k in range(nv))]].value or 0)
                               for j in range(P)])
                return supp, xv
            return None, None

        sl, _ = _solve(eps_lo)
        while sl is None and eps_lo < eps_hi * 0.9:
            eps_lo *= 2.0
            sl, _ = _solve(eps_lo)
            if self.verbose:
                print(f"  [sdp2] eps_lo → {eps_lo:.3e} {'✓' if sl else '✗'}")

        sh, _ = _solve(eps_hi)
        k_hi = len(sh) if sh is not None else 0
        k_lo = len(sl) if sl is not None else P
        if self.verbose:
            print(f"  [sdp2] eps [{eps_lo:.1e}, {eps_hi:.1e}]  k_lo={k_lo}  k_hi={k_hi}")

        frontier = {k_lo: eps_lo, k_hi: eps_hi}
        for k in range(k_hi + 1, k_lo):
            left  = frontier.get(k + 1, eps_lo)
            right = frontier.get(k - 1, eps_hi)
            if left >= right:
                frontier[k] = right; continue
            for _ in range(max(self.n_eps // max(k_lo - k_hi, 1), 3)):
                mid = np.sqrt(left * right)
                sm, _ = _solve(mid)
                if sm is None: break
                if len(sm) <= k: right = mid
                else:             left  = mid
                if right / left < 1.02: break
            frontier[k] = right

        pairs = [(e, s) for eps in sorted(set(frontier.values()))
                 for e, s in [(eps, _solve(eps)[0])] if s is not None]
        pairs.sort(key=lambda p: p[0])
        if not pairs:
            return RecoveryResult(method="l0_sdp2", coef=np.zeros(P),
                support=[], names=[], active_coef=[],
                residual=float(np.linalg.norm(y)),
                meta={"warning": "no_feasible_sdp2"})

        valid_eps = [p[0] for p in pairs]
        supports_raw = [p[1] for p in pairs]
        if self.verbose:
            print(f"  [sdp2] {len(pairs)} sparsity levels")

        runs = self._find_plateau_runs(supports_raw, valid_eps)
        non_trivial = [(s, e, supp) for s, e, supp in runs if len(supp) > 0] or runs
        rec_s, rec_e, rec_supp = min(non_trivial, key=lambda r: valid_eps[r[0]])
        if self.verbose:
            print(f"  [sdp2] selected: k={len(rec_supp)}  "
                  f"eps=[{valid_eps[rec_s]:.3e}, {valid_eps[rec_e]:.3e}]")

        rec_cols = sorted(rec_supp)
        if rec_cols:
            ols_coef, _, _, _ = np.linalg.lstsq(Theta[:, rec_cols], y, rcond=None)
        else:
            ols_coef = np.array([])
        coef = np.zeros(P)
        for j, col in enumerate(rec_cols):
            coef[col] = float(ols_coef[j])

        return RecoveryResult(method="l0_sdp2", coef=coef,
            support=rec_cols, names=[names[i] for i in rec_cols],
            active_coef=[float(coef[i]) for i in rec_cols],
            residual=float(np.linalg.norm(y - Theta @ coef)),
            meta={"plateau_eps_range": (valid_eps[rec_s], valid_eps[rec_e])},
        )

    @staticmethod
    def _mons(dim, deg):
        """All multi-indices of length *dim* with sum ≤ *deg*."""
        out = []
        for d in range(deg + 1):
            stack = [([], d, dim - 1)]
            while stack:
                pref, rem, k = stack.pop()
                if k == 0:
                    out.append(tuple(pref + [rem]))
                else:
                    for e in range(rem + 1):
                        stack.append((pref + [e], rem - e, k - 1))
        return out

    @staticmethod
    def _solve_order2_single(Theta, y, eps_val):
        """Single-epsilon order-2 Lasserre SDP. Returns (support, xi) or (None, None)."""
        try:
            import cvxpy as cp
        except ImportError:
            return None, None

        P = Theta.shape[1]
        nv = 2 * P
        a = Theta.T @ y; Q = Theta.T @ Theta; c2 = float(y @ y)

        d1 = OperatorIdentifier._mons(nv, 1)
        d2 = OperatorIdentifier._mons(nv, 2)
        d4_map = {t: i for i, t in enumerate(OperatorIdentifier._mons(nv, 4))}
        N1, N2, N4 = len(d1), len(d2), len(d4_map)
        _xi = lambda i: i
        _zi = lambda i: P + i
        _add = lambda t1, t2: tuple(t1[k] + t2[k] for k in range(nv))
        z0 = (0,) * nv

        yy = cp.Variable(N4)
        cons = [yy[d4_map[z0]] == 1.0]

        # M₂ ≽ 0
        m2d = {}
        for r, mr in enumerate(d2):
            for c, mc in enumerate(d2):
                k = d4_map.get(_add(mr, mc))
                if k is not None:
                    m2d[(r, c)] = m2d.get((r, c), 0) + yy[k]
        cons.append(cp.bmat([[m2d.get((r, c), 0) for c in range(N2)] for r in range(N2)]) >> 0)

        def _m1m(poly):
            m = [[0] * N1 for _ in range(N1)]
            for r, mr in enumerate(d1):
                for c, mc in enumerate(d1):
                    e = 0
                    for mg, coeff in poly.items():
                        k = d4_map.get(_add(_add(mr, mc), mg))
                        if k is not None:
                            e += coeff * yy[k]
                    m[r][c] = e
            return cp.bmat(m)

        # binary: z_i² = z_i
        for i in range(P):
            g = {z0: 0}
            g[tuple(1 if k == _zi(i) else 0 for k in range(nv))] = 1
            g[tuple(2 if k == _zi(i) else 0 for k in range(nv))] = -1
            Mi = _m1m(g); cons.append(Mi >> 0); cons.append(cp.trace(Mi) == 0)

        # complementarity: ξ_i z_i = 0
        for i in range(P):
            g = {tuple((1 if k == _xi(i) else 0) + (1 if k == _zi(i) else 0) for k in range(nv)): 1}
            Mi = _m1m(g); cons.append(Mi >> 0); cons.append(cp.trace(Mi) == 0)

        # L2 constraint: ε - c2 + 2aᵀξ - ξᵀQξ ≥ 0
        gl2 = {z0: -c2}
        for j, aj in enumerate(a):
            if abs(aj) > 1e-14:
                gl2[tuple(1 if k == _xi(j) else 0 for k in range(nv))] = gl2.get(
                    tuple(1 if k == _xi(j) else 0 for k in range(nv)), 0) + 2.0 * aj
        for i in range(P):
            for j in range(P):
                if abs(Q[i, j]) > 1e-14:
                    m = tuple((1 if k == _xi(i) else 0) + (1 if k == _xi(j) else 0) for k in range(nv))
                    gl2[m] = gl2.get(m, 0) - Q[i, j]
        eps_p = cp.Parameter(nonneg=True); eps_p.value = eps_val
        ml2 = [[0] * N1 for _ in range(N1)]
        for r, mr in enumerate(d1):
            for c, mc in enumerate(d1):
                e = 0
                for mg, coeff in gl2.items():
                    k = d4_map.get(_add(_add(mr, mc), mg))
                    if k is not None:
                        e += coeff * yy[k]
                e += eps_p * yy[d4_map[_add(mr, mc)]]
                ml2[r][c] = e
        cons.append(cp.bmat(ml2) >> 0)

        obj = cp.Minimize(cp.sum([yy[d4_map[tuple(1 if k == _zi(i) else 0 for k in range(nv))]] for i in range(P)]))
        try:
            cp.Problem(obj, cons).solve(solver=cp.SCS, warm_start=False, max_iters=5000)
        except Exception:
            return None, None

        # extract support
        supp = frozenset(i for i in range(P)
                         if float(yy[d4_map[tuple(1 if k == _zi(i) else 0 for k in range(nv))]].value or 1) < 0.5)
        xi_vals = np.array([float(yy[d4_map[tuple(1 if k == _xi(j) else 0 for k in range(nv))]].value or 0)
                            for j in range(P)])
        return supp, xi_vals

    def _fit_l0_sdp2_pursuit(self, Theta, y, names) -> RecoveryResult:
        """Hierarchical correlation-cut pursuit with iterative voting.

          1. Recursive spectral cut on the correlation graph → groups.
          2. Cross-group OLS + voting: feature survives if it gets ≥ N-1
             votes (appears in all cross-group OLS runs involving its group).
          3. Re-cluster the survivors, repeat until the support stabilises
             or falls below *cluster_size*.
          4. Final OLS + threshold debiasing.
        """
        P = Theta.shape[1]
        n = len(y)

        col_norms = np.linalg.norm(Theta, axis=0)
        col_norms[col_norms < 1e-14] = 1.0
        Tn = Theta / col_norms

        from scipy.sparse.csgraph import laplacian as _splap
        from scipy.linalg import eigh

        active = np.arange(P)                          # current survivor set
        cs = self.cluster_size
        max_rounds = 10
        prev_n = P + 1

        for rnd in range(max_rounds):
            current_n = len(active)
            if current_n <= cs or current_n == prev_n:
                break
            prev_n = current_n

            # correlation on active set
            C = np.abs((Tn[:, active].T @ Tn[:, active]) / n)
            np.fill_diagonal(C, 0)

            # spectral cut → groups
            L = _splap(C, normed=False)
            idx = np.arange(current_n)

            def _recurse(inds):
                if len(inds) <= cs:
                    return [inds]
                Lb = L[np.ix_(inds, inds)]
                _, eb = eigh(Lb, subset_by_index=[1, 1])
                fb = eb[:, 0]
                sp = np.argsort(fb)
                mid = len(sp) // 2
                return _recurse(inds[sp[:mid]]) + _recurse(inds[sp[mid:]])

            groups = [g.tolist() for g in _recurse(idx)]
            ng = len(groups)
            min_votes = max(1, ng // 2)

            if self.verbose:
                print(f"  [curs] r{rnd} n={current_n} {ng} groups "
                      f"sizes={[len(g) for g in groups]} min_votes={min_votes}")

            # cross-group OLS + voting
            votes = {gidx: 0 for gidx in range(current_n)}
            for i in range(ng):
                for j in range(i + 1, ng):
                    cols = np.concatenate([groups[i], groups[j]])
                    try:
                        xi, _, _, _ = np.linalg.lstsq(Theta[:, active[cols]], y, rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    for k, col in enumerate(cols):
                        if abs(float(xi[k])) > 1e-14:
                            votes[col] += 1

            survivors = [gidx for gidx, v in votes.items() if v >= min_votes]
            if self.verbose:
                vc = sorted([(v, names[active[gi]]) for gi, v in votes.items() if v >= min_votes],
                            reverse=True)
                print(f"    votes satisfied: {vc}")

            active = active[survivors]
            if self.verbose:
                print(f"    survivors: {len(active)}")

        if self.verbose:
            print(f"  [curs] final candidates: {len(active)} "
                  f"{[names[i] for i in sorted(active)]}")

        # ── Final OMP on survivors, then OLS + threshold ────────────────
        if len(active) == 0:
            rec_cols = np.arange(P).tolist()
        else:
            rec_cols = sorted(active)

        coef = np.zeros(P)
        if rec_cols:
            if len(rec_cols) >= 6:
                from sklearn.linear_model import OrthogonalMatchingPursuit
                s_target = min(len(rec_cols) * 2 // 3, 10)
                cn = np.linalg.norm(Theta[:, rec_cols], axis=0)
                cn[cn < 1e-14] = 1.0
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=s_target)
                omp.fit(Theta[:, rec_cols] / cn, y)
                omp_coef = omp.coef_ / cn
                omp_idx = [j for j, c in enumerate(omp_coef) if abs(c) > 1e-10]
                rec_cols = [rec_cols[j] for j in omp_idx]

            ols, _, _, _ = np.linalg.lstsq(Theta[:, rec_cols], y, rcond=None)
            for j, col in enumerate(rec_cols):
                coef[col] = float(ols[j])

            thresh = self.threshold_coef if self.threshold_coef is not None else 1e-6
            m = float(np.max(np.abs(coef)))
            if m > 0:
                keep = [c for c in rec_cols if abs(coef[c]) > thresh * m]
                if self.verbose:
                    print(f"  [curs] omp-thresh keep={len(keep)}/{len(rec_cols)} th={thresh:.0e}×{m:.4f}")
                if 1 <= len(keep) < len(rec_cols):
                    ols2, _, _, _ = np.linalg.lstsq(Theta[:, keep], y, rcond=None)
                    coef = np.zeros(P)
                    for j, col in enumerate(keep):
                        coef[col] = float(ols2[j])
                    rec_cols = keep

        return RecoveryResult(
            method="l0_sdp2p",
            coef=coef,
            support=rec_cols,
            names=[names[i] for i in rec_cols],
            active_coef=[float(coef[i]) for i in rec_cols],
            residual=float(np.linalg.norm(y - Theta @ coef)),
            meta={"final_support_size": len(rec_cols)},
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
