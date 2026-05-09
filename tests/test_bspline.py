"""
Tests for opid._bspline.FunctionRepr.

Ported from pde-identification/tests/func_repr_test.py
"""
import unittest
import numpy as np
from scipy.interpolate import BSpline

try:
    from opid._bspline import FunctionRepr
    BSPLINE_AVAILABLE = True
except (ImportError, RuntimeError):
    BSPLINE_AVAILABLE = False


def generate_bsp_basis(
    grid_points: np.ndarray,
    interval_bounds: tuple,
    grid_number_for_bases: int,
    order: int,
    der_order: int = 0
):
    """
    Generate B-spline basis functions for one dimension interval.
    
    Reference implementation using scipy.interpolate.BSpline for testing.
    """
    grid_width = (interval_bounds[1] - interval_bounds[0]) / grid_number_for_bases
    knot_width = grid_width / (order + 1)
    knot_points = (
        interval_bounds[0]
        + np.arange(-order, (grid_number_for_bases + 1) * (order + 1))
        * knot_width
    )
    knot_points_number = len(knot_points)
    basis_number = knot_points_number - order - 1
    basis_mat = np.zeros((len(grid_points), basis_number))
    for n in range(basis_number):
        local_knots = knot_points[n : (n + order + 2)]
        bs_elem = BSpline.basis_element(local_knots)
        bs_elem = bs_elem.derivative(der_order)
        non_support = (grid_points > local_knots[-1]) | (grid_points < local_knots[0])
        b = bs_elem(grid_points)
        b[non_support] = 0.0
        basis_mat[:, n] = b
    return basis_mat


@unittest.skipIf(not BSPLINE_AVAILABLE, "B-spline extension not available")
class TestFunctionRepr(unittest.TestCase):
    """Test B-spline design-matrix construction and projection."""

    def test_design_matrix(self):
        """Test that design matrix matches scipy reference implementation."""
        b_spline_func = FunctionRepr(basis_type='b')

        x = np.random.rand(500) * 2 * np.pi
        grid_num, k = 10, 3
        t = np.arange(-k, (grid_num + 1) * (k + 1)) * (2 * np.pi) / grid_num / (k + 1)

        for i in range(3):
            m_matrix = generate_bsp_basis(x, (0, 2 * np.pi), grid_num, k, i)
            n_matrix = b_spline_func.b_construct_1d_design_matrix(
                x, t, k, i, False
            )
            self.assertTrue(
                np.allclose(m_matrix, n_matrix.todense(), rtol=1e-14),
                f"Design matrix mismatch at derivative order {i}"
            )

    def test_projection(self):
        """Test that B-spline projection accurately recovers sin(3x)."""
        interval = [0, 2 * np.pi]
        x_ = np.linspace(interval[0], interval[1], 33)[0:-1]
        x_samples = np.linspace(interval[0], interval[1], 513)[0:-1]
        y_ = np.sin(3 * x_)
        y_samples = np.sin(3 * x_samples)
        
        # Degree
        k = 5
        # Knots in each interval
        grid_num = 13
        t = (
            interval[0]
            + np.arange(-k, (grid_num + 1) * (k + 1))
            * (interval[1] - interval[0])
            / grid_num
            / (k + 1)
        )
        nt = len(t)
        colloq = t[k : nt - k - 1]
        func_space = FunctionRepr('b')

        sample_matrix = func_space.b_construct_1d_design_matrix(
            x_samples, t, k, 0, False, True
        )
        
        # With collocation points
        c = func_space.b_1d_solve(x_, y_, t, k, True, True, colloq)
        res = sample_matrix @ c
        self.assertTrue(
            np.allclose(res, y_samples, atol=1e-4),
            "B-spline projection with collocation failed"
        )

        # Without collocation points
        c = func_space.b_1d_solve(x_, y_, t, k, True, False)
        res = sample_matrix @ c
        self.assertFalse(
            np.allclose(res, y_samples, atol=1e-2),
            "B-spline projection without collocation should be less accurate"
        )


if __name__ == '__main__':
    unittest.main()
