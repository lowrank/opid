"""Shared utility functions for sparse recovery."""

from __future__ import annotations

import numpy as np


def _column_normalise(Theta):
    col_norms = np.linalg.norm(Theta, axis=0)
    col_norms[col_norms < 1e-14] = 1.0
    return Theta / col_norms, col_norms


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
