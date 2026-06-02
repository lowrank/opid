"""
opid._backend
=============
Self-contained spectral PDE engine on the periodic domain [0, 2π].

This module provides a complete spectral PDE solver with optional FFTW acceleration.
All C++ headers, Jinja2 templates, and Python logic are bundled within opid.

Architecture
------------
``SpectralEngine``
    Low-level engine.  Accepts a Mathematica-string RHS, compiles it to a
    FFTW shared library (fast path), or falls back to a pure-numpy
    pseudo-spectral integrator.  Both paths share the same physical-space
    API::

        engine.solve(u0, t_span)      → ndarray (n_t, N)
        engine.rhs_physical(u)        → ndarray (N,)
        engine.eval_symbolic(func_u)  → sympy expression
        engine.latex_repr()           → str (LaTeX)

``EvolutionEquation``
    High-level convenience class providing a symbolic equation interface.
    Useful for direct instantiation and symbolic inspection::

        eq = EvolutionEquation(["-6*D[u]*u - D[D[D[u]]]"])
        eq.wrap()          # compile FFTW library
        eq.solve(u0, t)    # integrate
        eq.compute(u)      # evaluate RHS at a state vector
        repr(eq)           # LaTeX string

C++ backend layout
------------------
``opid/_backend/src/``        — extra.h, linalg.h, utils.h
``opid/_backend/templates/``  — template_real.cpp, template_complex.cpp
Compiled ``.so`` files go to ``~/.cache/opid/``, keyed by MD5(rhs_str).

Compilation requirements
------------------------
* g++ (any version supporting ``-std=gnu++11``)
* libfftw3-dev  (``apt install libfftw3-dev``)

If either is absent the module falls back to a pure-numpy integrator
with the **identical public API** — no code change is needed by callers.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from scipy.integrate import odeint

logger = logging.getLogger(__name__)

# ── Paths to bundled resources ────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent          # opid/_backend/
_SRC_DIR     = _BACKEND_DIR / "src"
_TPL_DIR     = _BACKEND_DIR / "templates"
_CACHE_DIR   = Path.home() / ".cache" / "opid"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════ #
#  FFTW half-complex helpers                                               #
# ═══════════════════════════════════════════════════════════════════════ #

def _hc_wavenumbers(N: int) -> np.ndarray:
    """
    Return the signed wavenumber array for FFTW's HC (half-complex) packed
    layout for a real array of length N.

    FFTW_R2HC packs:
        index 0          : DC  (k = 0)
        index 1..N//2-1  : real parts   of k = 1..N//2-1
        index N//2       : Nyquist (real only)
        index N//2+1..N-1: imaginary parts of k = N//2-1..1 (descending)
    """
    k = np.zeros(N, dtype=int)
    for i in range(1, N // 2):
        k[i] = i
    k[N // 2] = N // 2
    for i in range(N // 2 + 1, N):
        k[i] = N - i
    return k


def _hc_diff_inplace(hc: np.ndarray) -> np.ndarray:
    """
    Apply spectral first-derivative filter to a half-complex array,
    replicating extra.h::D() exactly in numpy.
    Returns a new array (hc is not modified).
    """
    N   = len(hc)
    out = hc.copy()
    out[0] = 0.0
    for i in range(1, N // 2):
        tmp      =  out[i]
        out[i]   = -out[N - i] * i
        out[N-i] =  tmp        * i
    if N % 2 == 0:
        out[N // 2] = 0.0
    return out


def _hc_deriv_numpy(u: np.ndarray, order: int = 1) -> np.ndarray:
    """r-th spectral derivative of a real 1-D array using numpy rfft."""
    N   = len(u)
    k   = np.fft.rfftfreq(N, d=1.0 / N)
    Uf  = np.fft.rfft(u)
    Uf *= (1j * k) ** order
    return np.fft.irfft(Uf, n=N)


# ═══════════════════════════════════════════════════════════════════════ #
#  Sympy → C codegen                                                       #
# ═══════════════════════════════════════════════════════════════════════ #

def _rhs_to_c_code(rhs_str: Union[str, Tuple[str, str]], is_complex: bool) -> str:
    """
    Parse a Mathematica-notation RHS string with sympy and emit C++ source
    rendered from the bundled Jinja2 templates.

    The raw parsed expression (``D`` still as ``sympy.Function``, not
    ``Derivative``) is passed to ``codegen`` so the emitted C code contains
    ``D(u)`` calls, which map to the FFTW spectral derivative defined in
    ``extra.h``.

    Parameters
    ----------
    rhs_str : str or (str, str)
        Mathematica-notation RHS.  Two-tuple for two-component systems.
    is_complex : bool
        True for two-component (NLS-style) systems.

    Returns
    -------
    str — Complete C++ source ready for g++.
    """
    from sympy.parsing.mathematica import parse_mathematica
    from sympy.utilities.codegen import codegen
    import jinja2

    tpl_env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(_TPL_DIR)))

    if not is_complex:
        base = parse_mathematica(rhs_str)
        [(_, c_code), _] = codegen(
            ("model", base), "C99", "extra", header=False, empty=False
        )
        lines = c_code.replace("double", "Vector").split("\n")
        model_code = "".join(lines[3:5])
        return tpl_env.get_template("template_real.cpp").render(model_code=model_code)
    else:
        rhs_real_str, rhs_complex_str = rhs_str
        base_r = parse_mathematica(rhs_real_str)
        base_c = parse_mathematica(rhs_complex_str)
        [(_, c_code), _] = codegen(
            [("model_real", base_r), ("model_complex", base_c)],
            "C99", "extra", header=False, empty=False,
        )
        lines = c_code.replace("double", "Vector").split("\n")
        return tpl_env.get_template("template_complex.cpp").render(
            model_real_code="".join(lines[3:5]),
            model_complex_code="".join(lines[8:10]),
        )


# ═══════════════════════════════════════════════════════════════════════ #
#  Compilation cache                                                       #
# ═══════════════════════════════════════════════════════════════════════ #

def _compile_rhs(rhs_str: Union[str, Tuple[str, str]], is_complex: bool) -> Optional[Path]:
    """
    Compile the RHS to a shared library.  Caches by MD5(repr(rhs_str)).
    Returns the Path to the .so, or None on failure.
    """
    key    = hashlib.md5(repr(rhs_str).encode()).hexdigest()[:16]
    so_path = _CACHE_DIR / f"opid_rhs_{key}.so"
    if so_path.exists():
        logger.debug("Using cached shared library: %s", so_path)
        return so_path

    try:
        src = _rhs_to_c_code(rhs_str, is_complex)
    except Exception as e:
        logger.warning("Sympy codegen failed: %s", e)
        return None

    cpp_path = _CACHE_DIR / f"opid_rhs_{key}.cpp"
    cpp_path.write_text(src)

    c_flags = ["-shared", "-Ofast", "-march=native", "-std=gnu++11", "-ffast-math",
               f"-I{_SRC_DIR}"]
    l_flags = ["-lm", "-lfftw3"]
    cmd     = ["g++", str(cpp_path)] + c_flags + ["-o", str(so_path)] + l_flags

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        logger.warning("g++ compilation failed:\n%s", result.stderr)
        cpp_path.unlink(missing_ok=True)
        return None

    logger.debug("Compiled RHS to %s", so_path)
    return so_path


# ═══════════════════════════════════════════════════════════════════════ #
#  ctypes wrappers                                                         #
# ═══════════════════════════════════════════════════════════════════════ #

def _load_real_lib(so_path: Path):
    lib = np.ctypeslib.load_library(so_path.stem, str(so_path.parent))
    lib.model.restype = None
    lib.model.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags=["C_CONTIGUOUS", "ALIGNED"]),
        np.ctypeslib.ndpointer(dtype=np.float64, flags=["C_CONTIGUOUS", "ALIGNED", "WRITEABLE"]),
        ctypes.c_long,
    ]
    def call(u: np.ndarray) -> np.ndarray:
        u_c = np.ascontiguousarray(u, dtype=np.float64)
        res = np.empty_like(u_c)
        lib.model(u_c, res, ctypes.c_long(len(u_c)))
        return res
    return call


def _load_complex_lib(so_path: Path):
    lib = np.ctypeslib.load_library(so_path.stem, str(so_path.parent))
    for fname in ("model_real", "model_complex"):
        fn = getattr(lib, fname)
        fn.restype = None
        fn.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, flags=["C_CONTIGUOUS", "ALIGNED"]),
            np.ctypeslib.ndpointer(dtype=np.float64, flags=["C_CONTIGUOUS", "ALIGNED"]),
            np.ctypeslib.ndpointer(dtype=np.float64, flags=["C_CONTIGUOUS", "ALIGNED", "WRITEABLE"]),
            ctypes.c_long,
        ]
    def call(uv: np.ndarray) -> np.ndarray:
        N2  = len(uv) // 2
        u_c = np.ascontiguousarray(uv[:N2], dtype=np.float64)
        v_c = np.ascontiguousarray(uv[N2:], dtype=np.float64)
        res_r = np.empty_like(u_c)
        res_c = np.empty_like(v_c)
        lib.model_real   (u_c, v_c, res_r, ctypes.c_long(N2))
        lib.model_complex(u_c, v_c, res_c, ctypes.c_long(N2))
        return np.concatenate([res_r, res_c])
    return call


# ═══════════════════════════════════════════════════════════════════════ #
#  SpectralEngine                                                          #
# ═══════════════════════════════════════════════════════════════════════ #

class SpectralEngine:
    """
    Spectral PDE engine on [0, 2π] (periodic BCs).

    Accepts a Mathematica-string RHS and:
    1. Attempts g++/FFTW compilation → fast path.
    2. Falls back to pure-numpy rfft integrator → slow path.

    Both paths expose the **same** API:

        engine.solve(u0, t_span)          → ndarray (n_t, N)   physical space
        engine.rhs_physical(u)            → ndarray (N,)        d/dt
        engine.eval_symbolic(func_u)      → sympy Expr
        engine.latex_repr()               → str

    Parameters
    ----------
    rhs_str : str or (str, str) or None
        Mathematica-notation RHS.  ``None`` means a numpy_rhs must be
        supplied.  Two-tuple for two-component (NLS) systems.
    numpy_rhs : callable or None
        Pre-built numpy RHS  ``f(u_physical) -> du/dt``.
    is_complex : bool
        True for two-component systems.
    """

    def __init__(
        self,
        rhs_str: Optional[Union[str, Tuple[str, str]]],
        numpy_rhs=None,
        is_complex: bool = False,
        dt_max: Optional[float] = None,
    ):
        self.rhs_str    = rhs_str
        self.is_complex = is_complex
        self._compiled_call = None
        self._numpy_rhs     = numpy_rhs
        self._sympy_repr    = None   # cached sympy expression(s)
        self.dt_max     = dt_max

        if rhs_str is not None:
            self._try_compile()

    # ------------------------------------------------------------------ #
    #  Compilation                                                         #
    # ------------------------------------------------------------------ #

    def _try_compile(self):
        try:
            so = _compile_rhs(self.rhs_str, self.is_complex)
        except subprocess.TimeoutExpired:
            logger.warning("Compilation timed out, falling back to numpy.")
            return
        except Exception as e:
            logger.warning("Compilation failed: %s", e)
            return
        if so is None:
            logger.info("Falling back to numpy spectral RHS.")
            return
        try:
            loader = _load_complex_lib if self.is_complex else _load_real_lib
            self._compiled_call = loader(so)
            logger.debug("Using compiled FFTW RHS from %s", so)
        except Exception as e:
            logger.warning("Failed to load compiled library: %s", e)

    # ------------------------------------------------------------------ #
    #  RHS evaluation                                                      #
    # ------------------------------------------------------------------ #

    def rhs_physical(self, state: np.ndarray) -> np.ndarray:
        """
        Evaluate d(state)/dt in physical space.

        For single-component systems ``state`` has shape ``(N,)``.
        For two-component systems ``state`` has shape ``(2N,)`` with
        Re(ψ) in the first N elements and Im(ψ) in the last N.
        """
        if self._compiled_call is not None:
            return self._compiled_call(state)
        if self._numpy_rhs is not None:
            return self._numpy_rhs(state)
        raise RuntimeError("No RHS available: provide rhs_str or numpy_rhs.")

    # ------------------------------------------------------------------ #
    #  Integration                                                         #
    # ------------------------------------------------------------------ #

    def solve(
        self,
        u0: np.ndarray,
        t_span: np.ndarray,
        rtol: float = 1e-9,
        atol: float = 1e-9,
        mxstep: int = 20000,
        method: str = "odeint",
    ) -> np.ndarray:
        """
        Integrate the PDE from ``u0`` (physical space) over ``t_span``.

        method='odeint' uses scipy's classic wrapper; 'lsoda' uses
        solve_ivp with explicit jac=None (avoids known hang).
        """
        if method == "lsoda":
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                lambda t, y: self.rhs_physical(y),
                (t_span[0], t_span[-1]),
                u0, t_eval=t_span,
                method='LSODA', rtol=rtol, atol=atol,
                jac=None,  # explicit None avoids hang (#10309)
            )
            return sol.y.T
        if method == "rk4":
            return self._solve_rk4(u0, t_span)
        return odeint(
            lambda y, t: self.rhs_physical(y),
            u0, t_span,
            rtol=rtol, atol=atol,
            tfirst=False,
            mxstep=mxstep,
        )

    def _solve_rk4(self, u0: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        """PDE-aware fixed-step RK4 with CFL-based sub-stepping."""
        dt_out = t_span[1] - t_span[0]
        N = len(u0)
        if self.dt_max is not None:
            dt_max = self.dt_max
        else:
            dt_max = 0.5 / (max(N, 1) * max(N, 1)) / 4
        n_sub = max(1, int(np.ceil(dt_out / dt_max)))
        dt = dt_out / n_sub
        n_t = len(t_span)
        u = np.asarray(u0, dtype=np.float64)
        sol = np.zeros((n_t, len(u)), dtype=np.float64)
        sol[0] = u.copy()
        for i in range(1, n_t):
            for _ in range(n_sub):
                k1 = self.rhs_physical(u)
                k2 = self.rhs_physical(u + 0.5 * dt * k1)
                k3 = self.rhs_physical(u + 0.5 * dt * k2)
                k4 = self.rhs_physical(u + dt * k3)
                u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            if np.isnan(u).any():
                break
            sol[i] = u.copy()
        return sol

    # ------------------------------------------------------------------ #
    #  Symbolic / display                                                  #
    # ------------------------------------------------------------------ #

    def _get_sympy_repr(self):
        """Return sympy expression(s) with D replaced by Derivative."""
        if self._sympy_repr is not None:
            return self._sympy_repr
        if self.rhs_str is None:
            return None
        try:
            from sympy import Function, Derivative
            from sympy.abc import x
            from sympy.parsing.mathematica import parse_mathematica

            if not self.is_complex:
                base = parse_mathematica(self.rhs_str)
                expr = base.replace(Function("D"), lambda f_: Derivative(f_, x))
                self._sympy_repr = expr
            else:
                rhs_r, rhs_c = self.rhs_str
                base_r = parse_mathematica(rhs_r).replace(
                    Function("D"), lambda f_: Derivative(f_, x))
                base_c = parse_mathematica(rhs_c).replace(
                    Function("D"), lambda f_: Derivative(f_, x))
                self._sympy_repr = (base_r, base_c)
        except Exception as e:
            logger.warning("Sympy parsing failed: %s", e)
            return None
        return self._sympy_repr

    def eval_symbolic(self, func_u, func_v=None):
        """
        Evaluate the RHS symbolically at a given sympy function.

        Parameters
        ----------
        func_u : sympy Expr   Substituted for ``u``.
        func_v : sympy Expr   Substituted for ``v`` (two-component only).

        Returns
        -------
        sympy Expr or tuple of two sympy Expr.
        """
        from sympy.abc import u, v
        expr = self._get_sympy_repr()
        if expr is None:
            raise RuntimeError("No rhs_str set; cannot evaluate symbolically.")
        if self.is_complex:
            return (
                expr[0].subs(u, func_u).subs(v, func_v).doit(),
                expr[1].subs(u, func_u).subs(v, func_v).doit(),
            )
        return expr.subs(u, func_u).doit()

    def latex_repr(self) -> str:
        """Return a LaTeX string representation of the RHS."""
        try:
            from sympy import latex
            expr = self._get_sympy_repr()
            if expr is None:
                return "(no symbolic repr)"
            if self.is_complex:
                return (
                    r"$\partial_t u = %s$" % latex(expr[0]) + "\n"
                    r"$\partial_t v = %s$" % latex(expr[1])
                )
            return r"$\partial_t u = %s$" % latex(expr)
        except Exception:
            return str(self.rhs_str)

    def __repr__(self) -> str:
        return self.latex_repr()

    # ------------------------------------------------------------------ #
    #  Metadata                                                            #
    # ------------------------------------------------------------------ #

    @property
    def backend(self) -> str:
        """``'fftw_compiled'`` or ``'numpy'``."""
        return "fftw_compiled" if self._compiled_call is not None else "numpy"


# ═══════════════════════════════════════════════════════════════════════ #
#  EvolutionEquation — high-level compatibility class                      #
# ═══════════════════════════════════════════════════════════════════════ #

class EvolutionEquation:
    """
    High-level PDE evolution equation class.

    Provides a symbolic interface for defining, compiling, and solving
    evolution PDEs on the periodic domain [0, 2π].  All functionality is
    self-contained within opid.

    Parameters
    ----------
    funcs_list : list of str
        * One element  → single-component equation (u only).
        * Two elements → two-component system  (u = Re, v = Im).

    Mathematica-string syntax
    -------------------------
    * ``'D[u]'``           — first derivative
    * ``'D[D[u]]'``        — second derivative
    * ``'D[u]*u'``         — product
    * ``'D[u]*Sin[u]'``    — trig function of u

    Examples
    --------
    >>> eq = EvolutionEquation(["-6*D[u]*u - D[D[D[u]]]"])
    >>> eq.wrap()
    >>> sol = eq.solve(u0, t_span)

    >>> eq2 = EvolutionEquation(["-v_xx - sigma*(u^2+v^2)*v",
    ...                          "+u_xx + sigma*(u^2+v^2)*u"])
    """

    def __init__(self, funcs_list: list, dt_max: Optional[float] = None):
        if len(funcs_list) == 2:
            self.func_str_real    = funcs_list[0]
            self.func_str_complex = funcs_list[1]
            self.complex          = True
            rhs_str = (self.func_str_real, self.func_str_complex)
        else:
            self.func_str_real = funcs_list[0]
            self.complex       = False
            rhs_str = self.func_str_real

        self._dt_max = dt_max

        self._engine = SpectralEngine(rhs_str=rhs_str, is_complex=self.complex,
                                      dt_max=dt_max)
        self._engine._compiled_call = None
        self.model_lib = None

    # ------------------------------------------------------------------ #
    #  Compile & load                                                      #
    # ------------------------------------------------------------------ #

    def wrap(self, label: str = "model"):
        """Compile the FFTW shared library and bind it."""
        rhs_str = (
            (self.func_str_real, self.func_str_complex)
            if self.complex else self.func_str_real
        )
        self._engine.rhs_str = rhs_str
        self._engine._try_compile()

    # ------------------------------------------------------------------ #
    #  Symbolic evaluation                                                 #
    # ------------------------------------------------------------------ #

    def eval(self, func_u, func_v=None):
        """Evaluate the RHS symbolically."""
        return self._engine.eval_symbolic(func_u, func_v)

    # ------------------------------------------------------------------ #
    #  Numerical evaluation                                                #
    # ------------------------------------------------------------------ #

    def compute(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate the compiled RHS at a physical-space state vector.

        Parameters
        ----------
        data : ndarray (N,) for real, (2N,) for complex systems.
        """
        return self._engine.rhs_physical(data)

    # ------------------------------------------------------------------ #
    #  Integration                                                         #
    # ------------------------------------------------------------------ #

    def solve(
        self,
        initial: np.ndarray,
        t_span: np.ndarray,
        rtol: float = 1e-12,
        atol: float = 1e-12,
        dt_max: Optional[float] = None,
    ) -> np.ndarray:
        """
        Integrate from ``initial`` (physical space) over ``t_span``.

        Returns
        -------
        sol : ndarray (n_t, N) — physical-space snapshots (rows = time).
        """
        if dt_max is not None:
            self._engine.dt_max = dt_max
        return self._engine.solve(initial, t_span, rtol=rtol, atol=atol)

    # ------------------------------------------------------------------ #
    #  Display                                                             #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return self._engine.latex_repr()


# Backward-compat alias
EvolutionDiffEq = EvolutionEquation
