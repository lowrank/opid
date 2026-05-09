"""
PDESimulator — unified spectral PDE integrator backed by SpectralEngine.

All simulations use periodic boundary conditions on [0, 2π].

The ``SpectralEngine`` (in ``opid._backend``) handles:
  * Compiling the Mathematica-string RHS to a FFTW shared library (fast path).
  * Falling back to a pure-numpy spectral RHS when g++/fftw3 are absent.

Both paths share the same physical-space API:
    engine.solve(u0, t_span)   → ndarray (n_t, N)  physical-space snapshots
    engine.rhs_physical(u)     → ndarray (N,)       d/dt in physical space

Supported equations (factory methods)
--------------------------------------
* ``kdv``        — Korteweg–de Vries:           u_t = -6 u u_x - u_xxx
* ``ks``         — Kuramoto–Sivashinsky:         u_t = -u u_x - u_xx - u_xxxx
* ``burgers``    — Viscous Burgers:              u_t = -u u_x + ν u_xx
* ``allen_cahn`` — Allen–Cahn:                   u_t = ε u_xx + u - u³
* ``fisher_kpp`` — Fisher–KPP:                   u_t = D u_xx + r u(1-u)
* ``nls``        — Focusing NLS (real+imag):     i ψ_t + ψ_xx + σ|ψ|² ψ = 0
* ``custom``     — Arbitrary Mathematica-string RHS
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from opid._backend import SpectralEngine


# ── Initial-condition helpers ────────────────────────────────────────────

def _random_ic(N: int, n_modes: int, seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u0 = np.zeros(N)
    for m in range(1, n_modes + 1):
        a, b = rng.standard_normal(2)
        u0 += a * np.cos(m * x) + b * np.sin(m * x)
    return u0


# ── Numpy RHS builders (physical space) ─────────────────────────────────
# Each builder returns a callable  f(u) -> du/dt  operating in physical
# space (real ndarray of length N or 2N for NLS).  These are passed as
# the numpy_rhs argument of SpectralEngine so the fallback is exact.

def _build_kdv_rhs(N: int):
    k = np.fft.rfftfreq(N, d=1.0 / N)
    def rhs(u):
        U  = np.fft.rfft(u)
        ux = np.fft.irfft(1j * k * U, n=N)
        uxxx = np.fft.irfft(-(1j * k) ** 3 * U, n=N)
        return -6.0 * u * ux + uxxx
    return rhs


def _build_ks_rhs(N: int):
    k = np.fft.rfftfreq(N, d=1.0 / N)
    def rhs(u):
        U    = np.fft.rfft(u)
        ux   = np.fft.irfft(1j * k * U, n=N)
        uxx  = np.fft.irfft(-(k ** 2) * U, n=N)
        uxxxx = np.fft.irfft((k ** 4) * U, n=N)
        return -u * ux - uxx - uxxxx
    return rhs


def _build_burgers_rhs(N: int, nu: float):
    k = np.fft.rfftfreq(N, d=1.0 / N)
    def rhs(u):
        U   = np.fft.rfft(u)
        ux  = np.fft.irfft(1j * k * U, n=N)
        uxx = np.fft.irfft(-(k ** 2) * U, n=N)
        return -u * ux + nu * uxx
    return rhs


def _build_allen_cahn_rhs(N: int, eps: float):
    k = np.fft.rfftfreq(N, d=1.0 / N)
    def rhs(u):
        U   = np.fft.rfft(u)
        uxx = np.fft.irfft(-(k ** 2) * U, n=N)
        return eps * uxx + u - u ** 3
    return rhs


def _build_fisher_kpp_rhs(N: int, D: float, r: float):
    k = np.fft.rfftfreq(N, d=1.0 / N)
    def rhs(u):
        U   = np.fft.rfft(u)
        uxx = np.fft.irfft(-(k ** 2) * U, n=N)
        return D * uxx + r * u * (1.0 - u)
    return rhs


def _build_nls_rhs(N: int, sigma: float):
    k = np.fft.rfftfreq(N, d=1.0 / N)
    def rhs(uv):
        u = uv[:N];  v = uv[N:]
        U = np.fft.rfft(u);  V = np.fft.rfft(v)
        uxx = np.fft.irfft(-(k ** 2) * U, n=N)
        vxx = np.fft.irfft(-(k ** 2) * V, n=N)
        mod2 = u ** 2 + v ** 2
        du = -vxx - sigma * mod2 * v
        dv =  uxx + sigma * mod2 * u
        return np.concatenate([du, dv])
    return rhs


# ═══════════════════════════════════════════════════════════════════════ #
#  PDESimulator                                                            #
# ═══════════════════════════════════════════════════════════════════════ #

class PDESimulator:
    """
    Spectral PDE simulator on [0, 2π] with periodic boundary conditions.

    Do not construct directly — use a factory classmethod.

    Parameters
    ----------
    N : int          Number of spatial grid points (power of 2 recommended).
    T : float        Final integration time.
    n_t : int        Number of output time snapshots (including t=0).
    n_modes : int    Number of random Fourier modes in the initial condition.
    seed : int|None  RNG seed for the initial condition.
    backend : str    ``'auto'`` (default), ``'numpy'``, or ``'fftw'``.
                     ``'auto'`` tries FFTW compilation first, falls back to numpy.
                     ``'numpy'`` forces the pure-numpy integrator.
                     ``'fftw'`` forces FFTW compilation (raises if unavailable).
    """

    def __init__(
        self,
        N: int = 256,
        T: float = 0.05,
        n_t: int = 100,
        n_modes: int = 8,
        seed: Optional[int] = 42,
        backend: str = "auto",
    ):
        self.N       = N
        self.T       = T
        self.n_t     = n_t
        self.n_modes = n_modes
        self.seed    = seed
        self.backend = backend

        self.x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        self.k = np.fft.rfftfreq(N, d=1.0 / N)   # length N//2+1

        # Filled by factory methods
        self._name:        str           = "unknown"
        self._rhs_str:     Optional[str] = None    # Mathematica string (single)
        self._numpy_rhs                  = None    # f(u_physical) -> du/dt
        self._is_complex:  bool          = False   # two-component system?
        self._rescale_ic:  bool          = False   # rescale IC to [0,1]?
        self._engine: Optional[SpectralEngine] = None

    # ------------------------------------------------------------------ #
    #  Factory methods                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def kdv(cls, **kwargs) -> "PDESimulator":
        """KdV: u_t = -6 u u_x - u_xxx."""
        sim = cls(**kwargs)
        sim._name     = "KdV"
        sim._rhs_str  = "-6*D[u]*u - D[D[D[u]]]"
        sim._numpy_rhs = _build_kdv_rhs(sim.N)
        return sim

    @classmethod
    def ks(cls, **kwargs) -> "PDESimulator":
        """Kuramoto–Sivashinsky: u_t = -u u_x - u_xx - u_xxxx."""
        sim = cls(**kwargs)
        sim._name     = "KS"
        # conservative form: -u u_x = -d/dx(u²/2)
        sim._rhs_str  = "-D[u]*u/2 + D[D[u]] - D[D[D[D[u]]]]"
        sim._numpy_rhs = _build_ks_rhs(sim.N)
        return sim

    @classmethod
    def burgers(cls, nu: float = 0.05, **kwargs) -> "PDESimulator":
        """Viscous Burgers: u_t = -u u_x + ν u_xx."""
        sim = cls(**kwargs)
        sim._name     = "Burgers"
        sim._rhs_str  = f"-D[u]*u + {nu}*D[D[u]]"
        sim._numpy_rhs = _build_burgers_rhs(sim.N, nu)
        sim.nu        = nu
        return sim

    @classmethod
    def allen_cahn(cls, eps: float = 0.01, **kwargs) -> "PDESimulator":
        """Allen–Cahn: u_t = ε u_xx + u - u³."""
        sim = cls(**kwargs)
        sim._name     = "Allen-Cahn"
        sim._rhs_str  = f"{eps}*D[D[u]] + u - u^3"
        sim._numpy_rhs = _build_allen_cahn_rhs(sim.N, eps)
        sim.eps       = eps
        return sim

    @classmethod
    def fisher_kpp(cls, D: float = 0.01, r: float = 1.0, **kwargs) -> "PDESimulator":
        """Fisher–KPP: u_t = D u_xx + r u(1 - u).

        The initial condition is rescaled to [0, 1] because FKPP models
        a population density; negative values make the reaction term
        r u(1-u) a positive feedback, destabilising the integration.
        """
        sim = cls(**kwargs)
        sim._name     = "Fisher-KPP"
        sim._rhs_str  = f"{D}*D[D[u]] + {r}*u*(1 - u)"
        sim._numpy_rhs = _build_fisher_kpp_rhs(sim.N, D, r)
        sim.D         = D
        sim.r         = r
        sim._rescale_ic = True
        return sim

    @classmethod
    def nls(cls, sigma: float = 1.0, **kwargs) -> "PDESimulator":
        """
        Focusing NLS: i ψ_t + ψ_xx + σ|ψ|² ψ = 0.

        State vector: uv = [Re(ψ), Im(ψ)], shape (2N,).
        Returns U of shape (N, n_t) (real part), and U_t likewise.
        """
        sim = cls(**kwargs)
        sim._name       = "NLS"
        sim._is_complex = True
        sim._rhs_str    = None   # two-component; no single Mathematica string
        sim._numpy_rhs  = _build_nls_rhs(sim.N, sigma)
        sim.sigma       = sigma
        return sim

    @classmethod
    def custom(cls, rhs_str: str, name: str = "custom", **kwargs) -> "PDESimulator":
        """
        Arbitrary single-component PDE via Mathematica-string RHS.

        The RHS is compiled to a FFTW shared library; falls back to a
        symbolic numpy RHS if compilation is unavailable.

        Parameters
        ----------
        rhs_str : str
            e.g. ``"-D[u]*u - D[D[D[u]]]"``
        """
        sim = cls(**kwargs)
        sim._name    = name
        sim._rhs_str = rhs_str
        # No pre-built numpy_rhs; SpectralEngine will build one symbolically
        # via sympy if g++/fftw3 is unavailable.
        return sim

    # ------------------------------------------------------------------ #
    #  Public properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------ #
    #  Integration                                                         #
    # ------------------------------------------------------------------ #

    def _get_engine(self) -> SpectralEngine:
        """Lazily construct (and cache) the SpectralEngine."""
        if self._engine is not None:
            return self._engine

        numpy_rhs = self._numpy_rhs

        # Respect backend override
        rhs_str = self._rhs_str
        if self.backend == "numpy":
            rhs_str = None   # suppress compilation

        self._engine = SpectralEngine(
            rhs_str=rhs_str,
            numpy_rhs=numpy_rhs,
            is_complex=self._is_complex,
        )

        if self.backend == "fftw" and self._engine.backend != "fftw_compiled":
            raise RuntimeError(
                f"backend='fftw' requested but FFTW compilation failed for {self._name}."
            )
        return self._engine

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the PDE and return ``(U, U_t)``.

        Returns
        -------
        U   : ndarray (N, n_t)   Physical-space snapshots u(x, tⱼ).
        U_t : ndarray (N, n_t)   Time-derivative  u_t(x, tⱼ) (from RHS).
        """
        engine = self._get_engine()
        t_span = np.linspace(0.0, self.T, self.n_t)

        # Build initial condition
        u0 = _random_ic(self.N, self.n_modes, self.seed)
        if self._rescale_ic:
            u0 = u0 - u0.min()
            u0 = u0 / u0.max() if u0.max() > 0 else u0
        if self._is_complex:
            rng = np.random.default_rng(self.seed)
            v0 = np.zeros(self.N)
            for m in range(1, self.n_modes + 1):
                a, b = rng.standard_normal(2)
                v0 += a * np.cos(m * self.x) + b * np.sin(m * self.x)
            ic = np.concatenate([u0, v0])
        else:
            ic = u0

        # Integrate (physical space throughout)
        sol = engine.solve(ic, t_span)   # (n_t, N) or (n_t, 2N)

        if self._is_complex:
            N = self.N
            U   = sol[:, :N].T           # (N, n_t) — real part
            # Compute U_t via RHS on each snapshot
            U_t = np.column_stack([
                engine.rhs_physical(sol[j])[:N] for j in range(self.n_t)
            ])
        else:
            U   = sol.T                  # (N, n_t)
            U_t = np.column_stack([
                engine.rhs_physical(sol[j]) for j in range(self.n_t)
            ])

        return U, U_t
