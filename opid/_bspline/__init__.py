"""
opid._bspline
-------------
B-spline design-matrix and function-representation utilities for smoothing
noisy trajectory data before computing spatial derivatives.

The ``design_matrix`` Cython extension must be compiled before use:

    cd opid/
    pip install -e .[fftw]

or equivalently:

    python setup.py build_ext --inplace

If the compiled extension is absent a ``RuntimeError`` with a clear message
is raised at the first ``smooth=True`` call.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent


def _try_compile_design_matrix() -> bool:
    """Attempt a just-in-time build of design_matrix.pyx."""
    setup_py = _HERE.parent.parent / "setup.py"
    if not setup_py.exists():
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(setup_py), "build_ext", "--inplace"],
            cwd=str(_HERE.parent.parent),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


def _import_design_matrix():
    """Import the compiled design_matrix extension, compiling if needed."""
    # Add _bspline dir to sys.path so Cython .so is findable
    _bspline_str = str(_HERE)
    if _bspline_str not in sys.path:
        sys.path.insert(0, _bspline_str)

    try:
        import design_matrix as _dm  # noqa: F401
        return _dm
    except ImportError:
        pass

    # Try JIT build
    if _try_compile_design_matrix():
        try:
            import importlib
            import design_matrix as _dm  # noqa: F401
            return _dm
        except ImportError:
            pass

    raise RuntimeError(
        "The 'design_matrix' Cython extension is not compiled.\n"
        "Run:  pip install -e '.[upstream]'  (or python setup.py build_ext --inplace)\n"
        "from the opid/ package root before using smooth=True."
    )


# Lazy attribute: only load design_matrix when actually needed
_design_matrix_mod = None


def _get_design_matrix():
    global _design_matrix_mod
    if _design_matrix_mod is None:
        _design_matrix_mod = _import_design_matrix()
    return _design_matrix_mod


# Expose design_matrix as a module-level attribute that triggers load on access
class _LazyModule:
    """Proxy that loads design_matrix on first attribute access."""
    def __getattr__(self, name):
        return getattr(_get_design_matrix(), name)


design_matrix = _LazyModule()

from opid._bspline.function_repr import FunctionRepr  # noqa: E402

__all__ = ["FunctionRepr", "design_matrix"]
