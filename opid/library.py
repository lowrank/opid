"""
FeatureLibrary — candidate term dictionary for operator identification.

Dictionary design
-----------------
All monomials of the form

    φ_{p, r}(u, D^r u) = u^p · (D^r u)^q

with  p ∈ {0,…,poly_degree},  r ∈ {0,…,max_deriv},  p + q ≤ max_cross_degree
are included, together with optional trigonometric and absolute-value terms.

Spatial derivatives are computed spectrally (FFT), so the library is
machine-precision accurate even at high derivative orders.

B-spline smoothing
------------------
When the optional B-spline smoother is available (via opid._bspline) the library
can optionally smooth noisy u-data before computing derivatives, improving
derivative quality for collocated (non-spectral) problems.  Pass
``smooth=True`` and ``smooth_degree=k``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Try bundled B-spline smoother ────────────────────────────────────────
_FunctionRepr = None
try:
    from opid._bspline import FunctionRepr as _FunctionRepr  # type: ignore
except Exception:
    pass


def _spectral_deriv(U: np.ndarray, order: int) -> np.ndarray:
    """r-th spatial derivative in physical space via rfft (axis 0)."""
    N = U.shape[0]
    k = np.fft.rfftfreq(N, d=1.0 / N)   # 0, 1, ..., N//2
    U_hat = np.fft.rfft(U, axis=0)
    U_hat *= (1j * k[:, None]) ** order if U.ndim > 1 else (1j * k) ** order
    return np.fft.irfft(U_hat, n=N, axis=0)


def _deriv_name(r: int) -> str:
    if r == 0:
        return "u"
    sub = "x" * r if r <= 4 else f"x^{{{r}}}"
    return f"u_{sub}"


class FeatureLibrary:
    """
    Build a candidate feature matrix Θ from PDE trajectory snapshots.

    Parameters
    ----------
    poly_degree : int
        Maximum pure polynomial degree in u (default 3).
    max_deriv : int
        Maximum spatial derivative order (default 4).
    cross_terms : bool
        Include u^p · D^r u mixed products (default True).
    max_cross_degree : int
        Maximum total degree p+q in cross-products (default 4).
    include_const : bool
        Include the constant feature "1" (default True).
    trig_terms : bool
        Include sin(u) and cos(u) (default False).
    smooth : bool
        Smooth u with a B-spline before differentiating (default False).
        Requires opid._bspline (installed with fftw extras).
    smooth_degree : int
        B-spline degree used for smoothing (default 5).
    smooth_n_knots : int
        Number of uniform knot intervals for smoothing (default 20).
    """

    def __init__(
        self,
        poly_degree: int = 3,
        max_deriv: int = 4,
        cross_terms: bool = True,
        max_cross_degree: int = 4,
        include_const: bool = True,
        trig_terms: bool = False,
        smooth: bool = False,
        smooth_degree: int = 5,
        smooth_n_knots: int = 20,
    ):
        self.poly_degree = poly_degree
        self.max_deriv = max_deriv
        self.cross_terms = cross_terms
        self.max_cross_degree = max_cross_degree
        self.include_const = include_const
        self.trig_terms = trig_terms
        self.smooth = smooth
        self.smooth_degree = smooth_degree
        self.smooth_n_knots = smooth_n_knots

        if smooth and _FunctionRepr is None:
            raise ImportError(
                "smooth=True requires the compiled design_matrix Cython extension. "
                "Run:  pip install -e '.[upstream]'  (or python setup.py build_ext --inplace) "
                "from the opid/ package root."
            )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def build(
        self,
        U: np.ndarray,
        k: np.ndarray = None,
        U_hat: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Construct the feature matrix and return ``(Theta, names)``.

        Parameters
        ----------
        U     : ndarray (N, n_t)   Physical-space solution snapshots.
        k     : ndarray or None    Ignored (kept for backward-compatibility).
                                   Wavenumbers are computed internally from N.
        U_hat : ndarray or None    Ignored (kept for backward-compatibility).

        Returns
        -------
        Theta : ndarray (N·n_t, P)  Feature matrix.
        names : list[str]           Column labels.
        """
        N, n_t = U.shape

        # Optionally smooth each time-slice with B-splines
        if self.smooth:
            x = np.linspace(0, 2 * np.pi, N, endpoint=False)
            U_smooth = self._bspline_smooth(U, x)
        else:
            U_smooth = U

        # Pre-compute derivative fields using rfft (no k argument needed)
        derivs: Dict[int, np.ndarray] = {0: U_smooth}
        for r in range(1, self.max_deriv + 1):
            derivs[r] = _spectral_deriv(U_smooth, r)

        terms: List[Tuple[str, np.ndarray]] = []

        # 1. Constant
        if self.include_const:
            terms.append(("1", np.ones((N, n_t))))

        # 2. Pure polynomial: u, u², …, u^p
        for p in range(1, self.poly_degree + 1):
            name = "u" if p == 1 else f"u^{p}"
            terms.append((name, U_smooth ** p))

        # 3. Pure derivative: u_x, u_xx, …
        for r in range(1, self.max_deriv + 1):
            terms.append((_deriv_name(r), derivs[r]))

        # 4. Cross terms: u^p · D^r u
        if self.cross_terms:
            for r in range(1, self.max_deriv + 1):
                for p in range(1, self.poly_degree + 1):
                    total = p + 1   # degree of D^r u is 1, so total = p+1
                    if total > self.max_cross_degree:
                        continue
                    u_label = "u" if p == 1 else f"u^{p}"
                    d_label = _deriv_name(r)
                    name = f"{u_label} {d_label}"
                    terms.append((name, U_smooth ** p * derivs[r]))

            # 5. Higher-power derivative cross terms: u^p · (D^r u)^q, q≥2
            for r in range(1, self.max_deriv + 1):
                for q in range(2, self.poly_degree + 1):
                    for p in range(0, self.poly_degree + 1):
                        total = p + q
                        if total > self.max_cross_degree or total < 2:
                            continue
                        d_label = _deriv_name(r)
                        if p == 0:
                            name = f"({d_label})^{q}"
                            feat = derivs[r] ** q
                        elif p == 1:
                            name = f"u ({d_label})^{q}"
                            feat = U_smooth * derivs[r] ** q
                        else:
                            name = f"u^{p} ({d_label})^{q}"
                            feat = U_smooth ** p * derivs[r] ** q
                        terms.append((name, feat))

        # 6. Trigonometric terms
        if self.trig_terms:
            terms.append(("sin(u)", np.sin(U_smooth)))
            terms.append(("cos(u)", np.cos(U_smooth)))
            for r in range(1, min(3, self.max_deriv) + 1):
                d_label = _deriv_name(r)
                terms.append((f"sin(u) {d_label}", np.sin(U_smooth) * derivs[r]))
                terms.append((f"cos(u) {d_label}", np.cos(U_smooth) * derivs[r]))

        # Stack: each field is (N, n_t) → flatten to (N·n_t,)
        Theta = np.column_stack([f.ravel() for _, f in terms])
        names = [n for n, _ in terms]

        return Theta, names

    # ------------------------------------------------------------------ #
    #  B-spline smoother (upstream)                                        #
    # ------------------------------------------------------------------ #

    def _bspline_smooth(self, U: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Smooth each time slice of U with a periodic B-spline."""
        assert _FunctionRepr is not None
        N, n_t = U.shape
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        deg = self.smooth_degree
        n_knots = self.smooth_n_knots
        t_knots = (
            np.arange(-deg, (n_knots + 1) * (deg + 1))
            * (2 * np.pi) / n_knots / (deg + 1)
        )
        func_repr = _FunctionRepr("b")
        U_smooth = np.empty_like(U)
        for j in range(n_t):
            c = func_repr.b_1d_solve(x, U[:, j], t_knots, deg, periodic=True)
            dm = func_repr.b_construct_1d_design_matrix(
                x, t_knots, deg, 0, False, periodic=True
            )
            U_smooth[:, j] = (dm @ c).ravel()
        return U_smooth

    # ------------------------------------------------------------------ #
    #  Info                                                                #
    # ------------------------------------------------------------------ #

    def describe(self, names: List[str]) -> str:
        """Return a formatted summary of the library."""
        lines = [f"FeatureLibrary  ({len(names)} terms)"]
        for i, n in enumerate(names):
            lines.append(f"  [{i:3d}]  {n}")
        return "\n".join(lines)
