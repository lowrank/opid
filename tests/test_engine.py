"""
Tests for opid._backend.EvolutionEquation (formerly EvolutionDiffEq).

Ported from pde-identification/tests/evolution_diff_eq_test.py
"""
import unittest
import numpy as np
from sympy import simplify, Eq, sin, cos, pi, Function
from sympy.abc import x

from opid._backend import EvolutionEquation


class TestEvolutionEquation(unittest.TestCase):
    """Test symbolic evaluation and spectral solver."""

    def test_1st_derivatives(self):
        """Test D[u] symbolic evaluation."""
        model = EvolutionEquation(['D[u]'])
        self.assertEqual(simplify(Eq(model.eval(x ** 2 + 1), 2 * x)), True)
        self.assertEqual(simplify(Eq(model.eval(sin(x)), cos(x))), True)

    def test_2nd_derivatives(self):
        """Test D[D[u]] symbolic evaluation."""
        model = EvolutionEquation(['D[D[u]]'])
        self.assertEqual(simplify(Eq(model.eval(x ** 2 + 1), 2)), True)
        self.assertEqual(
            simplify(Eq(model.eval(x ** 6 + x ** 4), 30 * x ** 4 + 12 * x ** 2)),
            True
        )

    def test_multiplication(self):
        """Test D[D[u]] * u symbolic evaluation."""
        model = EvolutionEquation(['D[D[u]] * u'])
        self.assertEqual(simplify(Eq(model.eval(x ** 2 + 1), 2 * (x ** 2 + 1))), True)
        self.assertEqual(simplify(Eq(model.eval(sin(2 * x)), -4 * (sin(2 * x)) ** 2)), True)

    def test_trig(self):
        """Test D[u] * Sin[u] symbolic evaluation."""
        model = EvolutionEquation(['D[u] * Sin[u]'])
        self.assertEqual(simplify(Eq(model.eval(x ** 2 + 1), 2 * x * sin(x ** 2 + 1))), True)
        self.assertEqual(simplify(Eq(model.eval(pi * x), pi * sin(pi * x))), True)

    def test_transport_solver(self):
        """Test spectral solver for transport equation u_t = u_x."""
        N = 128
        dt, t_end = 0.001, 1.0
        t_span = np.arange(dt, t_end + dt, dt)
        x_axis = np.linspace(0, 2 * np.pi, N, endpoint=False)
        y0 = np.sin(x_axis)

        model = EvolutionEquation(['D[u]'])
        model.wrap(label="transport")

        sol = model.solve(y0, t_span, rtol=1e-12, atol=1e-12, dt_max=0.01)
        error = np.mean(np.abs(np.sin(x_axis + t_span[499]) - sol[500, :]))
        self.assertLess(error, 1e-4, f"Transport solver error {error:.2e} exceeds tolerance")

    def test_diffusion_solver(self):
        """Test spectral solver for diffusion equation u_t = u_xx."""
        N = 128
        dt, t_end = 0.001, 1.0
        t_span = np.arange(dt, t_end + dt, dt)
        x_axis = np.linspace(0, 2 * np.pi, N, endpoint=False)
        y0 = np.sin(x_axis)

        model = EvolutionEquation(['D[D[u]]'])
        model.wrap(label="diffusion")

        sol = model.solve(y0, t_span, rtol=1e-12, atol=1e-12, dt_max=5e-4)
        error = np.mean(np.abs(np.sin(x_axis) * np.exp(-t_span[499]) - sol[500, :]))
        self.assertLess(error, 1e-4, f"Diffusion solver error {error:.2e} exceeds tolerance")


if __name__ == '__main__':
    unittest.main()
