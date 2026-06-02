"""Correlation-Cut Pursuit recovery with cross-group OLS voting."""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import laplacian as _splap
from scipy.linalg import eigh

from .base import BaseRecovery, RecoveryResult
from ._utils import _column_normalise
from .subsample import Subsampler, RandomSubsampler, FullSubsampler, QRSubsampler, SignalQRSubsampler


class CCPRecovery(BaseRecovery):
    """CCP: spectral clustering + cross-group OLS voting + iterative pruning."""

    def __init__(self, cluster_size=8, adaptive_groups_lambda=0.5,
                 threshold_coef=None, subsampler=None, max_samples=4000,
                 verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.cluster_size = cluster_size
        self.adaptive_groups_lambda = adaptive_groups_lambda
        self.threshold_coef = threshold_coef
        self.subsampler = subsampler
        self.max_samples = max_samples
        self.verbose = verbose

    def _fit(self, Theta, y, names):
        P = Theta.shape[1]
        n = len(y)

        col_norms = np.linalg.norm(Theta, axis=0)
        col_norms[col_norms < 1e-14] = 1.0
        Tn = Theta / col_norms

        active = np.arange(P)
        cs = self.cluster_size
        max_rounds = 10
        prev_n = P + 1

        # Row subsampling (default: signal-aware QR — filter tail, then max-volume)
        s = self.subsampler or SignalQRSubsampler()
        ridx = s.select(Tn[:, active], self.max_samples)

        for rnd in range(max_rounds):
            current_n = len(active)
            if current_n <= cs or current_n == prev_n:
                break
            prev_n = current_n

            Tn_sub = Tn[ridx]
            ys = y[ridx]
            ns = len(ridx)

            C = np.abs((Tn_sub[:, active].T @ Tn_sub[:, active]) / ns)
            np.fill_diagonal(C, 0)
            C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

            L = _splap(C, normed=False)
            L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
            idx = np.arange(current_n)

            def _recurse(inds):
                if len(inds) <= cs:
                    return [inds]
                Lb = L[np.ix_(inds, inds)]
                try:
                    lambda2, eb = eigh(np.nan_to_num(Lb, nan=0.0), subset_by_index=[1, 1])
                    fb = eb[:, 0]
                except Exception:
                    return [inds]
                if lambda2[0] > self.adaptive_groups_lambda:
                    return [inds]
                sp = np.argsort(fb)
                mid = len(sp) // 2
                return _recurse(inds[sp[:mid]]) + _recurse(inds[sp[mid:]])

            groups = [g.tolist() for g in _recurse(idx)]
            ng = len(groups)

            # ── cross-group OLS voting ─────────────────────────────────
            votes = {gidx: 0 for gidx in range(current_n)}
            for i in range(ng):
                for j in range(i + 1, ng):
                    cols = np.concatenate([groups[i], groups[j]])
                    Tsub = Theta[:, active[cols]]
                    cn = np.linalg.norm(Tsub, axis=0, keepdims=True)
                    cn[cn < 1e-14] = 1.0
                    try:
                        xi, _, _, _ = np.linalg.lstsq(Tsub[ridx] / cn, ys, rcond=None)
                    except np.linalg.LinAlgError: continue
                    thresh = max(float(np.max(np.abs(xi))) * 1e-3, 1e-10)
                    for k, col in enumerate(cols):
                        if abs(float(xi[k])) > thresh:
                            votes[col] += 1

            # ── survivors ──────────────────────────────────────────────
            min_votes = max(1, ng // 2)
            survivors = [gidx for gidx, v in votes.items() if v >= min_votes]
            if len(survivors) == current_n and ng > 1:
                min_votes = max(1, ng - 1)
                survivors = [gidx for gidx, v in votes.items() if v >= min_votes]

            if self.verbose:
                print(f"  [curs] r{rnd} n={current_n} {ng} groups "
                      f"sizes={[len(g) for g in groups]} min_votes={min_votes}")

            active = active[survivors]
            if self.verbose:
                print(f"    survivors: {len(active)}")

        if self.verbose:
            print(f"  [curs] final candidates: {len(active)} "
                  f"{[names[i] for i in sorted(active)]}")

        # ── final OMP + OLS debiasing ──────────────────────────────────
        if len(active) == 0:
            rec_cols = np.arange(P).tolist()
        else:
            rec_cols = sorted(active)

        coef = np.zeros(P)
        if len(rec_cols) > 6:
            from sklearn.linear_model import OrthogonalMatchingPursuit
            Tn_full = _column_normalise(Theta[:, rec_cols])[0]
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(len(rec_cols), 6))
            omp.fit(Tn_full, y)
            rec_cols = [rec_cols[i] for i, c in enumerate(omp.coef_) if abs(c) > 1e-6]

        if rec_cols:
            ols_coef, _, _, _ = np.linalg.lstsq(Theta[:, rec_cols], y, rcond=None)
            for j, col in enumerate(rec_cols):
                coef[col] = float(ols_coef[j])

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

        residual = float(np.linalg.norm(y - Theta @ coef))
        return RecoveryResult(
            method="ccp", coef=coef, support=rec_cols,
            names=[names[i] for i in rec_cols],
            active_coef=[float(coef[i]) for i in rec_cols],
            residual=residual,
        )
