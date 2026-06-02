"""Order-2 Lasserre SDP relaxation for L0 recovery via cvxpy."""

from __future__ import annotations

import numpy as np

from .base import BaseRecovery, RecoveryResult
from ._utils import _mons, _find_plateau_runs


class L0SDP2Recovery(BaseRecovery):
    """L0 via Lasserre order-2 SDP relaxation with L2 constraint.

    Moment matrix over all degree-2 monomials in (ϟ, z) with order-1
    localising matrices enforcing binary and complementarity constraints.
    Tractable for P ≤ ∼9.
    """

    def __init__(
        self,
        n_eps=30,
        eps_factor_hi=100.0,
        eps_factor_lo=0.01,
        max_samples=5000,
        threshold_coef=None,
        random_state=42,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_eps = n_eps
        self.eps_factor_hi = eps_factor_hi
        self.eps_factor_lo = eps_factor_lo
        self.max_samples = max_samples
        self.threshold_coef = threshold_coef
        self.random_state = random_state
        self.verbose = verbose

    def _fit(self, Theta, y, names):
        try:
            import cvxpy as cp
        except ImportError as e:
            raise ImportError("cvxpy is required for the 'l0_sdp2' method.") from e

        n = len(y)
        m = min(self.max_samples, n)
        s = self.subsampler
        if s is not None:
            idx = s.select(Theta, m)
        else:
            idx = np.arange(m)
        Ts = Theta[idx]
        ys = y[idx]

        col_scales = np.linalg.norm(Ts, axis=0, keepdims=True)
        col_scales[col_scales < 1e-14] = 1.0
        Ts = Ts / col_scales

        P = Ts.shape[1]
        nv = 2 * P
        a = Ts.T @ ys
        Q = Ts.T @ Ts
        c2 = float(ys @ ys)

        d1 = _mons(nv, 1)
        d2 = _mons(nv, 2)
        d4_map = {t: i for i, t in enumerate(_mons(nv, 4))}

        N2 = len(d2)
        N1 = len(d1)
        N4 = len(d4_map)

        _xi = lambda i: i
        _zi = lambda i: P + i
        _add = lambda t1, t2: tuple(t1[k] + t2[k] for k in range(nv))

        try:
            xi_ols, _, _, _ = np.linalg.lstsq(Ts, ys, rcond=None)
            ols_res = float(np.linalg.norm(ys - Ts @ xi_ols))
        except np.linalg.LinAlgError:
            ols_res = float(np.linalg.norm(ys))
        noise_floor = max(float(np.linalg.norm(y, 1)) * 0.1, ols_res * 0.5)

        yy = cp.Variable(N4)
        eps_param = cp.Parameter(nonneg=True)
        cons = [yy[d4_map[(0,) * nv]] == 1.0]

        m2 = {}
        for r, mr in enumerate(d2):
            for c, mc in enumerate(d2):
                k = d4_map.get(_add(mr, mc))
                if k is not None:
                    m2[(r, c)] = m2.get((r, c), 0) + yy[k]
        M2 = cp.bmat([[m2.get((r, c), 0) for c in range(N2)] for r in range(N2)])
        cons.append(M2 >> 0)

        def _m1_matrix(poly_coeffs):
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

        z0 = tuple(0 for _ in range(nv))
        for i in range(P):
            g = {z0: 0}
            zi_mono = tuple(1 if k == _zi(i) else 0 for k in range(nv))
            zi2_mono = tuple(2 if k == _zi(i) else 0 for k in range(nv))
            g[zi_mono] = g.get(zi_mono, 0) + 1
            g[zi2_mono] = g.get(zi2_mono, 0) - 1
            Mi = _m1_matrix(g)
            cons.append(Mi >> 0)
            cons.append(cp.trace(Mi) == 0)

        for i in range(P):
            g = {}
            xz_mono = tuple((1 if k == _xi(i) else 0) + (1 if k == _zi(i) else 0) for k in range(nv))
            g[xz_mono] = 1
            Mi = _m1_matrix(g)
            cons.append(Mi >> 0)
            cons.append(cp.trace(Mi) == 0)

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

        m1_l2 = [[0] * N1 for _ in range(N1)]
        for r, mr in enumerate(d1):
            for c, mc in enumerate(d1):
                expr = 0
                for mono_g, coeff in g_nop.items():
                    k = d4_map.get(_add(_add(mr, mc), mono_g))
                    if k is not None:
                        expr += coeff * yy[k]
                expr += eps_param * yy[d4_map[_add(mr, mc)]]
                m1_l2[r][c] = expr
        cons.append(cp.bmat(m1_l2) >> 0)

        obj = cp.Minimize(cp.sum([yy[d4_map[tuple(1 if k == _zi(i) else 0 for k in range(nv))]] for i in range(P)]))
        prob = cp.Problem(obj, cons)

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
            left = frontier.get(k + 1, eps_lo)
            right = frontier.get(k - 1, eps_hi)
            if left >= right:
                frontier[k] = right
                continue
            for _ in range(max(self.n_eps // max(k_lo - k_hi, 1), 3)):
                mid = np.sqrt(left * right)
                sm, _ = _solve(mid)
                if sm is None:
                    break
                if len(sm) <= k:
                    right = mid
                else:
                    left = mid
                if right / left < 1.02:
                    break
            frontier[k] = right

        pairs = [
            (e, s) for eps in sorted(set(frontier.values()))
            for e, s in [(eps, _solve(eps)[0])] if s is not None
        ]
        pairs.sort(key=lambda p: p[0])
        if not pairs:
            return RecoveryResult(
                method="l0_sdp2", coef=np.zeros(P),
                support=[], names=[], active_coef=[],
                residual=float(np.linalg.norm(y)),
                meta={"warning": "no_feasible_sdp2"},
            )

        valid_eps = [p[0] for p in pairs]
        supports_raw = [p[1] for p in pairs]
        if self.verbose:
            print(f"  [sdp2] {len(pairs)} sparsity levels")

        runs = _find_plateau_runs(supports_raw, valid_eps)
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

        # Threshold truncation (same as l0_pareto)
        thresh = self.threshold_coef if self.threshold_coef is not None else 1e-4
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
            method="l0_sdp2", coef=coef,
            support=rec_cols, names=[names[i] for i in rec_cols],
            active_coef=[float(coef[i]) for i in rec_cols],
            residual=float(np.linalg.norm(y - Theta @ coef)),
            meta={"plateau_eps_range": (valid_eps[rec_s], valid_eps[rec_e])},
        )
