"""
setup.py — Cython extension build for opid._bspline.design_matrix

Usage:
    pip install -e .[upstream]          # recommended (also installs deps)
    python setup.py build_ext --inplace # direct build for development
"""

from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    import numpy as np

    extensions = cythonize(
        [
            Extension(
                name="opid._bspline.design_matrix",
                sources=["opid/_bspline/design_matrix.pyx"],
                include_dirs=[
                    np.get_include(),
                    "opid/_bspline",   # for __fitpack.h (direct)
                    "opid/src",        # for __fitpack.h (relative ../src path in .pyx)
                ],
                extra_compile_args=["-O2"],
            )
        ],
        compiler_directives={"language_level": "3"},
    )
except Exception:
    # Cython or numpy not available at build time: skip extension silently.
    # The package will still import; smooth=True will raise a clear error.
    extensions = []

setup(ext_modules=extensions)
