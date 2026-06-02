"""Tests for PDESimulator data generation."""
import unittest
import numpy as np
from opid import PDESimulator


class TestSimulator(unittest.TestCase):
    """Verify all built-in PDE simulators produce valid trajectories."""

    N = 64
    T = 0.01
    n_t = 10
    seed = 42

    def _check(self, sim):
        """Common assertions for any simulator output."""
        U, U_t = sim.run()
        self.assertEqual(U.shape, (self.N, self.n_t))
        self.assertEqual(U_t.shape, (self.N, self.n_t))
        self.assertFalse(np.isnan(U).any(), f"{sim.name}: U has NaN")
        self.assertFalse(np.isinf(U).any(), f"{sim.name}: U has Inf")
        self.assertFalse(np.isnan(U_t).any(), f"{sim.name}: U_t has NaN")
        self.assertFalse(np.isinf(U_t).any(), f"{sim.name}: U_t has Inf")

    def test_kdv(self):
        sim = PDESimulator.kdv(N=self.N, T=self.T, n_t=self.n_t,
                               n_modes=4, seed=self.seed, backend="numpy")
        self._check(sim)
        self.assertEqual(sim.name, "KdV")

    def test_ks(self):
        sim = PDESimulator.ks(N=self.N, T=self.T, n_t=self.n_t,
                              n_modes=4, seed=self.seed, backend="numpy")
        self._check(sim)
        self.assertEqual(sim.name, "KS")

    def test_burgers(self):
        sim = PDESimulator.burgers(nu=0.05, N=self.N, T=self.T, n_t=self.n_t,
                                   n_modes=4, seed=self.seed, backend="numpy")
        self._check(sim)
        self.assertEqual(sim.name, "Burgers")

    def test_allen_cahn(self):
        sim = PDESimulator.allen_cahn(eps=0.01, N=self.N, T=self.T, n_t=self.n_t,
                                      n_modes=4, seed=self.seed, backend="numpy")
        self._check(sim)
        self.assertEqual(sim.name, "Allen-Cahn")

    def test_fisher_kpp(self):
        sim = PDESimulator.fisher_kpp(D=0.01, r=1.0, N=self.N, T=self.T,
                                      n_t=self.n_t, n_modes=4, seed=self.seed,
                                      backend="numpy")
        self._check(sim)
        self.assertEqual(sim.name, "Fisher-KPP")
        # FKPP IC is rescaled to [0,1]
        U, _ = sim.run()
        self.assertAlmostEqual(U.min(), 0.0, delta=0.01)
        self.assertAlmostEqual(U.max(), 1.0, delta=0.01)

    def test_time_derivative_consistency(self):
        """u_t computed from RHS should be consistent with finite-difference
        time derivative of U."""
        sim = PDESimulator.kdv(N=self.N, T=0.05, n_t=30,
                               n_modes=4, seed=self.seed, backend="numpy")
        U, U_t = sim.run()
        dt = 0.05 / 30
        # Central difference for interior time points
        U_t_fd = np.zeros_like(U)
        U_t_fd[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2 * dt)
        # Forward/backward at boundaries
        U_t_fd[:, 0] = (U[:, 1] - U[:, 0]) / dt
        U_t_fd[:, -1] = (U[:, -1] - U[:, -2]) / dt
        # Relative error should be small for smooth solutions
        rel_err = np.linalg.norm(U_t - U_t_fd) / np.linalg.norm(U_t)
        self.assertLess(rel_err, 1.0, f"Time derivative mismatch: {rel_err:.3f}")

    def test_initial_condition_stats(self):
        """Initial conditions should have reasonable statistics."""
        sim = PDESimulator.kdv(N=256, T=0.01, n_t=5,
                               n_modes=8, seed=99, backend="numpy")
        U, _ = sim.run()
        u0 = U[:, 0]
        # Should have roughly zero mean for sufficiently many modes
        self.assertAlmostEqual(float(np.mean(u0)), 0.0, delta=0.5)
        # Should have non-zero std
        self.assertGreater(np.std(u0), 0.1)

    def test_reproducibility(self):
        """Same seed should give identical trajectories."""
        sim1 = PDESimulator.kdv(N=32, T=0.01, n_t=5,
                                n_modes=3, seed=42, backend="numpy")
        sim2 = PDESimulator.kdv(N=32, T=0.01, n_t=5,
                                n_modes=3, seed=42, backend="numpy")
        U1, U_t1 = sim1.run()
        U2, U_t2 = sim2.run()
        self.assertTrue(np.allclose(U1, U2, rtol=1e-14))
        self.assertTrue(np.allclose(U_t1, U_t2, rtol=1e-14))

    def test_different_seeds_different(self):
        """Different seeds should give different trajectories."""
        sim1 = PDESimulator.kdv(N=32, T=0.01, n_t=5,
                                n_modes=3, seed=1, backend="numpy")
        sim2 = PDESimulator.kdv(N=32, T=0.01, n_t=5,
                                n_modes=3, seed=2, backend="numpy")
        U1, _ = sim1.run()
        U2, _ = sim2.run()
        self.assertFalse(np.allclose(U1, U2, rtol=1e-14))

    def test_nls(self):
        """NLS should produce valid real/imag data."""
        sim = PDESimulator.nls(sigma=1.0, N=self.N, T=self.T, n_t=self.n_t,
                                n_modes=4, seed=self.seed, backend="numpy")
        self._check(sim)
        self.assertEqual(sim.name, "NLS")

    def test_custom_complex_wave_equation(self):
        """Custom two-component PDE (wave eq: u_t = v, v_t = u_xx)."""
        sim = PDESimulator.custom(
            ("v", "D[D[u]]"),
            name="wave-eq", N=self.N, T=self.T, n_t=self.n_t,
            n_modes=4, seed=self.seed, backend="numpy",
        )
        U, U_t = sim.run()
        self.assertEqual(U.shape, (self.N, self.n_t))
        self.assertEqual(U_t.shape, (self.N, self.n_t))
        self.assertFalse(np.isnan(U).any())
        self.assertFalse(np.isinf(U).any())
        self.assertFalse(np.isnan(U_t).any())
        self.assertFalse(np.isinf(U_t).any())
        self.assertEqual(sim.name, "wave-eq")
        self.assertTrue(sim._is_complex)

    def test_custom_complex_nls(self):
        """NLS via custom() should produce valid data."""
        sim = PDESimulator.custom(
            ("-D[D[v]] - 1.0*(u^2+v^2)*v", "D[D[u]] + 1.0*(u^2+v^2)*u"),
            name="NLS-custom", N=self.N, T=self.T, n_t=self.n_t,
            n_modes=4, seed=42, backend="numpy",
        )
        U, U_t = sim.run()
        self.assertEqual(U.shape, (self.N, self.n_t))
        self.assertEqual(U_t.shape, (self.N, self.n_t))
        self.assertFalse(np.isnan(U).any())
        self.assertFalse(np.isinf(U).any())
        self.assertFalse(np.isnan(U_t).any())
        self.assertFalse(np.isinf(U_t).any())
        self.assertEqual(sim.name, "NLS-custom")
        self.assertTrue(sim._is_complex)

    def test_custom_single_component_backward_compat(self):
        """custom() with a single string should still work (backward compat)."""
        sim = PDESimulator.custom(
            "-6*D[u]*u - D[D[D[u]]]",
            name="KdV-custom", N=self.N, T=self.T, n_t=self.n_t,
            n_modes=4, seed=42, backend="numpy",
        )
        U, U_t = sim.run()
        self.assertEqual(U.shape, (self.N, self.n_t))
        self.assertFalse(np.isnan(U).any())
        self.assertFalse(np.isinf(U).any())
        self.assertFalse(sim._is_complex)

    def test_custom_complex_fails_on_bad_tuple(self):
        """custom() with wrong-length tuple should raise ValueError."""
        with self.assertRaises(ValueError):
            PDESimulator.custom(("one", "two", "three"), N=16)


if __name__ == '__main__':
    unittest.main()
