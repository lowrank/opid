"""Mixed-Integer L0 Pareto recovery via cvxpy MILP."""

from __future__ import annotations

import numpy as np

from .base import BaseRecovery, RecoveryResult
from ._utils import _find_plateau_runs


class L0ParetoRecovery(BaseRecovery):
    """L0 minimisation with Pareto sweep over error tolerance epsilon.

    Uses a MILP with big-M big‑M bounding, a bisection-based Pareto sweep
    to find the sparsity-epsilon frontier, plateau detection for stable
    support selection, and final OLS debiasing on the full data.
    """

    def __init__(
        self,
        n_eps=30,
        eps_factor_hi=100.0,
        eps_factor_lo=0.01,
        max_samples=5000,
        milp_solver="GLPK_MI",
        threshold_coef=None,
        random_state=42,
        max_rounds=3,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_eps = n_eps
        self.eps_factor_hi = eps_factor_hi
        self.eps_factor_lo = eps_factor_lo
        self.max_samples = max_samples
        self.milp_solver = milp_solver
        self.threshold_coef = threshold_coef
        self.random_state = random_state
        self.max_rounds = max_rounds
        self.verbose = verbose

    def _fit(self, Theta, y, names):
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError("cvxpy is required for the 'l0_pareto' method.") from e

        n = len(y)
        m = min(self.max_samples, n)
        idx = self.subsampler.select(Theta, m)
        Ts = Theta[idx]
        ys = y[idx]

        col_scales = np.linalg.norm(Ts, axis=0, keepdims=True)
        col_scales[col_scales < 1e-14] = 1.0
        Ts_norm = Ts / col_scales

        try:
            xi_ls, _, _, _ = np.linalg.lstsq(Ts_norm, ys, rcond=None)
            ols_residual = float(np.linalg.norm(ys - Ts_norm @ xi_ls, 1))
        except np.linalg.LinAlgError:
            reg = 1e-6 * np.eye(Ts_norm.shape[1])
            xi_ls = np.linalg.solve(Ts_norm.T @ Ts_norm + reg, Ts_norm.T @ ys)
            ols_residual = float(np.linalg.norm(ys - Ts_norm @ xi_ls, 1))

        y_scale_full = float(np.linalg.norm(y, 1))
        noise_floor = y_scale_full * 0.1
        # Floor for adaptive tightening: no model can fit tighter than the
        # OLS residual on the full data.  Below this, MILP solvers return
        # garbage rather than declaring infeasibility.
        full_cn = np.linalg.norm(Theta, axis=0, keepdims=True)
        full_cn[full_cn < 1e-14] = 1.0
        try:
            xi_full, _, _, _ = np.linalg.lstsq(Theta / full_cn, y, rcond=None)
            ols_full_resid = float(np.linalg.norm(y - (Theta / full_cn) @ xi_full, 1))
        except np.linalg.LinAlgError:
            ols_full_resid = float(np.linalg.norm(y, 1)) * 0.01
        min_eps_tight = max(ols_full_resid * 0.5, 1e-6)

        P = Ts_norm.shape[1]
        # Per-column Big-M: each term gets its own bound based on its OLS coefficient.
        # Fixes CoinError crashes when coefficients have vastly different magnitudes
        # (e.g. AC: u³ at 382 vs u_xx at 21 with uniform M=764).
        M_vec = np.maximum(2.0 * np.abs(xi_ls), 0.1)

        xi_var = cp.Variable(P)
        z_var = cp.Variable(P, boolean=True)
        eps_param = cp.Parameter(nonneg=True)

        prob = cp.Problem(
            cp.Minimize(cp.sum(z_var)),
            [
                cp.norm(ys - Ts_norm @ xi_var, 1) <= eps_param,
                xi_var <= cp.multiply(M_vec, z_var),
                xi_var >= cp.multiply(-M_vec, z_var),
            ],
        )

        # CoinError-safe solve for CBC: run the entire Pareto sweep
        # in a single subprocess via _milp_worker.  The worker solves
        # all epsilon values internally and returns the frontier.
        # SCIP and other solvers are safe in-process.
        if self.milp_solver == "CBC":
            import subprocess, sys, json, tempfile

            pad_data = {
                "Theta": Ts_norm.tolist(),
                "y": ys.tolist(),
                "M": M_vec.tolist(),
                "col_scales": col_scales.flatten().tolist(),
                "noise_floor": noise_floor,
                "eps_lo_factor": self.eps_factor_lo,
                "eps_hi_factor": self.eps_factor_hi,
                "n_eps": self.n_eps,
                "solver": "CBC",
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump(pad_data, tmp)
                worker_file = tmp.name

            try:
                result = subprocess.run(
                    [sys.executable, "-m", "opid.recovery._milp_worker", worker_file],
                    capture_output=True, text=True, timeout=600,
                )
                frontiers = json.loads(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip() else []
            except (subprocess.TimeoutExpired, json.JSONDecodeError):
                frontiers = []
            finally:
                import os; os.unlink(worker_file)

            if not frontiers:
                return RecoveryResult(
                    method="l0_pareto",
                    coef=xi_ls / col_scales.flatten(),
                    support=list(range(P)), names=names,
                    active_coef=list(xi_ls / col_scales.flatten()),
                    residual=float(np.linalg.norm(y - Theta @ (xi_ls / col_scales.flatten()))),
                    meta={"warning": "no_feasible_milp"},
                )

            # Convert frontiers to (eps, support) pairs
            pairs = []
            for f in frontiers:
                eps_val = f["eps"]
                z_vals = f["z"]
                supp = frozenset(i for i in range(len(z_vals)) if z_vals[i] > 0.5)
                pairs.append((eps_val, supp))

            pairs.sort(key=lambda p: p[0])
            valid_eps = [p[0] for p in pairs]
            supports_raw = [p[1] for p in pairs]

            if self.verbose:
                print(f"  [CBC subprocess] {len(pairs)} sparsity levels")

            runs = _find_plateau_runs(supports_raw, valid_eps)
            non_trivial = [(s, e, supp) for s, e, supp in runs if len(supp) > 0] or runs
            rec_s, rec_e, rec_supp = min(non_trivial, key=lambda r: valid_eps[r[0]])

            # Adaptive tightening (re-run in subprocess if needed)
            eps_tight = valid_eps[rec_s]
            stable_count = 0
            while stable_count < 3:
                eps_tight /= 2.0
                if eps_tight < min_eps_tight: break
                # ... subprocess call ...
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                    json.dump({"eps": float(eps_tight), "solver": "CBC", **{k: v for k, v in pad_data.items() if k in ("Theta", "y", "M")}}, tmp)
                    tight_file = tmp.name
                try:
                    r2 = subprocess.run(
                        [sys.executable, "-m", "opid.recovery._milp_worker", tight_file],
                        capture_output=True, text=True, timeout=120,
                    )
                    ftight = json.loads(r2.stdout.strip()) if r2.returncode == 0 and r2.stdout.strip() else [{"z": []}]
                    sm_new = frozenset(i for i in range(len(ftight[0]["z"])) if ftight[0]["z"][i] > 0.5) if ftight else None
                except Exception:
                    sm_new = None
                finally:
                    import os; os.unlink(tight_file)
                if sm_new is None: break
                if sm_new == rec_supp:
                    stable_count += 1
                else:
                    rec_supp = sm_new
                    stable_count = 0

            rec_cols = sorted(rec_supp)
            if rec_cols:
                ols_coef, _, _, _ = np.linalg.lstsq(Theta[:, rec_cols], y, rcond=None)
            else:
                ols_coef = np.array([])

            thresh = self.threshold_coef if self.threshold_coef is not None else 1e-4
            coef = np.zeros(P)
            for j, col in enumerate(rec_cols):
                coef[col] = float(ols_coef[j])
            if thresh > 0 and rec_cols:
                max_abs = max(abs(coef[c]) for c in rec_cols)
                keep = [c for c in rec_cols if abs(coef[c]) > thresh * max_abs]
                if keep:
                    rec_cols = keep
                    ols_coef2, _, _, _ = np.linalg.lstsq(Theta[:, rec_cols], y, rcond=None)
                    coef = np.zeros(P)
                    for j, col in enumerate(rec_cols):
                        coef[col] = float(ols_coef2[j])

            return RecoveryResult(
                method="l0_pareto", coef=coef,
                support=rec_cols, names=[names[i] for i in rec_cols],
                active_coef=[float(coef[i]) for i in rec_cols],
                residual=float(np.linalg.norm(y - Theta @ coef)),
            )

        # ── In-process solve (SCIP, GLPK, etc.) ────────────────────

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

        eps_hi = noise_floor * self.eps_factor_hi
        eps_lo = noise_floor * self.eps_factor_lo

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

        frontier = {k_lo: eps_lo, k_hi: eps_hi}

        for k in range(k_hi + 1, k_lo):
            left = frontier.get(k + 1, eps_lo)
            right = frontier.get(k - 1, eps_hi)
            if left >= right:
                frontier[k] = right
                continue
            for _ in range(self.n_eps // max(k_lo - k_hi, 1)):
                mid = np.sqrt(left * right)
                sm, _ = _solve(mid)
                if sm is None:
                    break
                km = len(sm)
                if self.verbose:
                    print(f"  k={k} mid={mid:.3e} → k={km}")
                if km <= k:
                    right = mid
                else:
                    left = mid
                if right / left < 1.02:
                    break
            frontier[k] = right

        pairs = []
        for eps in sorted(frontier.values()):
            sm, _ = _solve(eps)
            if sm is not None:
                pairs.append((eps, sm))

        pairs = [(e, s) for e, s in pairs if s is not None]
        pairs.sort(key=lambda p: p[0])
        if not pairs:
            return RecoveryResult(
                method="l0_pareto",
                coef=xi_ls / col_scales.flatten(),
                support=list(range(P)),
                names=names,
                active_coef=list(xi_ls / col_scales.flatten()),
                residual=float(np.linalg.norm(y - Theta @ (xi_ls / col_scales.flatten()))),
                meta={"warning": "no_feasible_milp"},
            )

        valid_eps = [p[0] for p in pairs]
        supports_raw = [p[1] for p in pairs]

        if self.verbose:
            print(f"  {len(pairs)} distinct sparsity levels found")

        runs = _find_plateau_runs(supports_raw, valid_eps)

        if self.verbose and len(runs) > 1:
            for s, e, supp in runs:
                print(f"    k={len(supp)}  eps=[{valid_eps[s]:.3e}, {valid_eps[e]:.3e}]  len={e-s+1}")

        non_trivial = [(s, e, supp) for s, e, supp in runs if len(supp) > 0]
        if not non_trivial:
            non_trivial = runs
        rec_s, rec_e, rec_supp = min(non_trivial, key=lambda r: valid_eps[r[0]])

        if self.verbose:
            print(f"  selected: k={len(rec_supp)}  "
                  f"eps=[{valid_eps[rec_s]:.3e}, {valid_eps[rec_e]:.3e}]")

        eps_tight = valid_eps[rec_s]
        stable_count = 0
        while stable_count < 3:
            eps_tight /= 2.0
            if eps_tight < min_eps_tight: break
            sm_new, _ = _solve(eps_tight)
            if sm_new is None:
                break
            if sm_new == rec_supp:
                stable_count += 1
                if self.verbose:
                    print(f"  tight eps={eps_tight:.3e}: k={len(sm_new)} (stable {stable_count}/3)")
            else:
                rec_supp = sm_new
                stable_count = 0
                if self.verbose:
                    print(f"  tight eps={eps_tight:.3e}: k={len(sm_new)} (changed, reset)")

        rec_cols = sorted(rec_supp)
        if rec_cols:
            ols_coef, _, _, _ = np.linalg.lstsq(Theta[:, rec_cols], y, rcond=None)
        else:
            ols_coef = np.array([])

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
                "all_runs": [
                    (s, e, [names[i] for i in supp])
                    for s, e, supp in _find_plateau_runs(supports_raw, valid_eps)
                ],
                "M_vec": M_vec.tolist(),
                "noise_floor": noise_floor,
            },
        )
