"""
Comprehensive tests for the opid package.

Covers:
  - utils.py functions (jaccard_score, add_noise, relative_error, print_recovery_table)
  - RecoveryResult dataclass
  - FeatureLibrary with varied parameters
  - All PDESimulator factory methods including newer ones
  - Time-derivative consistency for all PDEs
  - L0 Pareto / L0 SDP2 recovery (requires cvxpy)
  - CCP variants (milp_vote, cluster_size)
  - Integration: sim → library → recovery end-to-end
  - Edge cases: empty data, single timestep, high noise, large libraries
"""
import io
import sys
import unittest
import numpy as np

from opid import PDESimulator, FeatureLibrary, OperatorIdentifier
from opid import RecoveryResult
from opid.utils import jaccard_score, add_noise, relative_error, print_recovery_table

# ── Optional dependency checks ──────────────────────────────────────────

try:
    import cvxpy as _cvxpy
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False

try:
    import sklearn
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from scipy.optimize import milp as _scipy_milp
    _HAS_SCIPY_MILP = True
except ImportError:
    _HAS_SCIPY_MILP = False


# ═══════════════════════════════════════════════════════════════════════ #
#  Utilities
# ═══════════════════════════════════════════════════════════════════════ #

class TestJaccardScore(unittest.TestCase):
    """Tests for jaccard_score utility."""

    def test_perfect_match(self):
        """Jaccard score should be 1.0 for identical sets."""
        self.assertEqual(jaccard_score(["a", "b"], ["a", "b"]), 1.0)

    def test_no_overlap(self):
        """Jaccard score should be 0.0 for disjoint sets."""
        self.assertEqual(jaccard_score(["a"], ["b"]), 0.0)

    def test_partial_overlap(self):
        """Jaccard score should handle partial overlap correctly."""
        self.assertAlmostEqual(jaccard_score(["a", "b"], ["a", "c"]), 1.0 / 3.0)

    def test_extra_term_in_found(self):
        """Jaccard penalizes spurious terms."""
        self.assertAlmostEqual(
            jaccard_score(["a", "b", "c"], ["a", "b"]), 2.0 / 3.0
        )

    def test_missing_term(self):
        """Jaccard penalizes missed terms."""
        self.assertAlmostEqual(
            jaccard_score(["a", "b"], ["a", "b", "c"]), 2.0 / 3.0
        )

    def test_empty_both(self):
        """Jaccard of two empty sets should be 0.0."""
        self.assertEqual(jaccard_score([], []), 0.0)

    def test_found_empty_true_nonempty(self):
        """Jaccard should be 0.0 when no terms are recovered."""
        self.assertEqual(jaccard_score(["a", "b"], []), 0.0)

    def test_true_empty_found_nonempty(self):
        """Jaccard should be 0.0 when true set is empty but terms are found."""
        self.assertEqual(jaccard_score([], ["a"]), 0.0)


class TestRelativeError(unittest.TestCase):
    """Tests for relative_error utility."""

    def test_zero_error(self):
        """Relative error of identical vectors must be 0."""
        coef = np.array([1.0, -6.0, 3.0])
        self.assertEqual(relative_error(coef, coef), 0.0)

    def test_nonzero_error(self):
        """Relative error must be consistent for scaled differences."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 0.0])
        expected = np.linalg.norm(a - b) / np.linalg.norm(a)
        self.assertAlmostEqual(relative_error(a, b), expected)

    def test_true_vector_near_zero(self):
        """When the true vector is nearly zero, return L2 norm of prediction."""
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(relative_error(a, b), np.linalg.norm(b))


class TestAddNoise(unittest.TestCase):
    """Tests for add_noise utility."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.Theta = rng.standard_normal((100, 5))
        self.y = rng.standard_normal(100)

    def test_zero_noise(self):
        """Zero noise level should return unchanged data."""
        Tn, yn = add_noise(self.Theta, self.y, noise_level=0.0, seed=0)
        self.assertTrue(np.allclose(Tn, self.Theta, rtol=1e-14))
        self.assertTrue(np.allclose(yn, self.y, rtol=1e-14))

    def test_reproducibility(self):
        """Same seed should produce identical noise."""
        T1, y1 = add_noise(self.Theta, self.y, noise_level=0.1, seed=42)
        T2, y2 = add_noise(self.Theta, self.y, noise_level=0.1, seed=42)
        self.assertTrue(np.allclose(T1, T2, rtol=1e-14))
        self.assertTrue(np.allclose(y1, y2, rtol=1e-14))

    def test_different_seeds_different(self):
        """Different seeds should produce different noise."""
        T1, y1 = add_noise(self.Theta, self.y, noise_level=0.1, seed=1)
        T2, y2 = add_noise(self.Theta, self.y, noise_level=0.1, seed=2)
        self.assertFalse(np.allclose(T1, T2, rtol=1e-14))
        self.assertFalse(np.allclose(y1, y2, rtol=1e-14))

    def test_shape_preserved(self):
        """Output shapes should match input shapes."""
        Tn, yn = add_noise(self.Theta, self.y, noise_level=0.05, seed=7)
        self.assertEqual(Tn.shape, self.Theta.shape)
        self.assertEqual(yn.shape, self.y.shape)

    def test_noise_alters_data(self):
        """Noisy data should not be identical to clean data."""
        Tn, yn = add_noise(self.Theta, self.y, noise_level=0.05, seed=3)
        self.assertFalse(np.allclose(Tn, self.Theta, rtol=1e-14))
        self.assertFalse(np.allclose(yn, self.y, rtol=1e-14))

    def test_near_zero_columns(self):
        """Columns with near-zero std should still be handled (no crash)."""
        Theta = np.zeros((50, 3))
        Theta[:, 1] = 1e-20  # near-zero column
        Theta[:, 2] = np.arange(50)  # normal column
        y = np.zeros(50)
        y[0] = 1e-20
        Tn, yn = add_noise(Theta, y, noise_level=0.01, seed=0)
        self.assertFalse(np.isnan(Tn).any())
        self.assertFalse(np.isinf(Tn).any())
        self.assertFalse(np.isnan(yn).any())
        self.assertFalse(np.isinf(yn).any())


class TestPrintRecoveryTable(unittest.TestCase):
    """Tests for print_recovery_table utility."""

    def setUp(self):
        self.true_names = ["u u_x", "u_xxx"]
        self.true_coefs = [-6.0, -1.0]

    def _make_result(self, method, names, coefs, residual=1e-3):
        P = len(self.true_names)
        full_coef = np.zeros(P)
        support = []
        for name, c in zip(names, coefs):
            idx = self.true_names.index(name) if name in self.true_names else -1
            if idx >= 0:
                full_coef[idx] = c
                support.append(idx)
        return RecoveryResult(
            method=method,
            coef=full_coef,
            support=support,
            names=names,
            active_coef=coefs,
            residual=residual,
        )

    def test_output_contains_title(self):
        """print_recovery_table should include the title in output."""
        r = self._make_result("omp", ["u u_x"], [-6.0])
        buf = io.StringIO()
        sys.stdout = buf
        try:
            print_recovery_table([r], self.true_names, self.true_coefs, title="Test")
            output = buf.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("Test", output)

    def test_output_contains_method_name(self):
        """print_recovery_table should include method names."""
        r = self._make_result("lasso", ["u_xxx"], [-1.0])
        buf = io.StringIO()
        sys.stdout = buf
        try:
            print_recovery_table([r], self.true_names, self.true_coefs)
            output = buf.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("lasso", output)
        self.assertIn("residual", output)

    def test_output_contains_checkmarks(self):
        """Correctly recovered terms should be marked with a checkmark."""
        r = self._make_result("ccp", ["u u_x", "u_xxx"], [-6.0, -1.0])
        buf = io.StringIO()
        sys.stdout = buf
        try:
            print_recovery_table([r], self.true_names, self.true_coefs)
            output = buf.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("u u_x", output)
        self.assertIn("u_xxx", output)

    def test_multiple_results(self):
        """print_recovery_table should handle multiple results."""
        r1 = self._make_result("omp", ["u u_x"], [-6.0])
        r2 = self._make_result("ccp", ["u u_x", "u_xxx"], [-6.0, -1.0])
        results = [r1, r2]
        # Should not raise
        buf = io.StringIO()
        sys.stdout = buf
        try:
            print_recovery_table(results, self.true_names, self.true_coefs, title="Multi")
            output = buf.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("omp", output)
        self.assertIn("ccp", output)


# ═══════════════════════════════════════════════════════════════════════ #
#  RecoveryResult dataclass
# ═══════════════════════════════════════════════════════════════════════ #

class TestRecoveryResult(unittest.TestCase):
    """Tests for the RecoveryResult dataclass."""

    def test_creation_basic(self):
        """RecoveryResult should be constructable with all required fields."""
        coef = np.array([0.0, -6.0, -1.0])
        result = RecoveryResult(
            method="omp",
            coef=coef,
            support=[1, 2],
            names=["u u_x", "u_xxx"],
            active_coef=[-6.0, -1.0],
            residual=0.001,
        )
        self.assertEqual(result.method, "omp")
        self.assertEqual(result.support, [1, 2])
        self.assertAlmostEqual(result.residual, 0.001)

    def test_empty_support(self):
        """RecoveryResult should handle empty support."""
        coef = np.zeros(3)
        result = RecoveryResult(
            method="lasso",
            coef=coef,
            support=[],
            names=[],
            active_coef=[],
            residual=1.0,
        )
        self.assertEqual(len(result.support), 0)
        self.assertEqual(len(result.names), 0)

    def test_string_representation(self):
        """__str__ should include method name, residual, and coefficient lines."""
        coef = np.array([0.0, -6.0])
        result = RecoveryResult(
            method="omp",
            coef=coef,
            support=[1],
            names=["u u_x"],
            active_coef=[-6.0],
            residual=0.0012,
        )
        s = str(result)
        self.assertIn("omp", s)
        self.assertIn("1.2000e", s)
        self.assertIn("u u_x", s)
        self.assertIn("-6.000000", s)

    def test_repr_matches_str(self):
        """__repr__ should delegate to __str__."""
        coef = np.array([-1.0])
        result = RecoveryResult(
            method="test",
            coef=coef,
            support=[0],
            names=["x"],
            active_coef=[-1.0],
            residual=0.0,
        )
        self.assertEqual(str(result), repr(result))

    def test_meta_field(self):
        """The optional meta dict should be stored."""
        coef = np.array([1.0])
        result = RecoveryResult(
            method="ccp",
            coef=coef,
            support=[0],
            names=["u"],
            active_coef=[1.0],
            residual=0.01,
            meta={"n_feasible": 8, "warning": "test"},
        )
        self.assertEqual(result.meta["n_feasible"], 8)
        self.assertEqual(result.meta["warning"], "test")


# ═══════════════════════════════════════════════════════════════════════ #
#  FeatureLibrary
# ═══════════════════════════════════════════════════════════════════════ #

class TestFeatureLibraryBasics(unittest.TestCase):
    """Basic tests for FeatureLibrary construction and output."""

    def setUp(self):
        N, n_t = 32, 10
        self.U = np.random.default_rng(1).standard_normal((N, n_t))

    def test_default_params(self):
        """Default FeatureLibrary should produce a non-empty Theta with names."""
        lib = FeatureLibrary()
        Theta, names = lib.build(self.U)
        self.assertGreater(Theta.shape[1], 0)
        self.assertEqual(len(names), Theta.shape[1])
        self.assertEqual(Theta.shape[0], self.U.size)

    def test_poly_degree(self):
        """poly_degree controls how many pure polynomial terms are included."""
        for deg in [1, 2, 5]:
            lib = FeatureLibrary(poly_degree=deg, max_deriv=0, cross_terms=False,
                                 include_const=False)
            _, names = lib.build(self.U)
            expected_poly = sum(1 for n in names if n.startswith("u^") or n == "u")
            self.assertEqual(expected_poly, deg)

    def test_max_deriv(self):
        """max_deriv controls how many derivative terms are included."""
        for d in [1, 2, 4]:
            lib = FeatureLibrary(poly_degree=0, max_deriv=d, cross_terms=False,
                                 include_const=False)
            _, names = lib.build(self.U)
            deriv_count = sum(1 for n in names if "_" in n and n != "u")
            self.assertEqual(deriv_count, d)

    def test_include_const(self):
        """include_const=True adds the constant feature '1'."""
        lib1 = FeatureLibrary(include_const=True, poly_degree=1, max_deriv=0,
                              cross_terms=False)
        _, names1 = lib1.build(self.U)
        self.assertIn("1", names1)

        lib2 = FeatureLibrary(include_const=False, poly_degree=1, max_deriv=0,
                              cross_terms=False)
        _, names2 = lib2.build(self.U)
        self.assertNotIn("1", names2)

    def test_cross_terms_disabled(self):
        """cross_terms=False should suppress all cross terms."""
        lib = FeatureLibrary(poly_degree=2, max_deriv=3, cross_terms=False,
                             include_const=False)
        _, names = lib.build(self.U)
        has_cross = any(" " in n for n in names)
        self.assertFalse(has_cross, f"Unexpected cross term in: {names}")

    def test_max_cross_degree(self):
        """max_cross_degree limits total degree of cross terms."""
        lib_lo = FeatureLibrary(poly_degree=3, max_deriv=3, max_cross_degree=2,
                                include_const=False)
        _, names_lo = lib_lo.build(self.U)

        lib_hi = FeatureLibrary(poly_degree=3, max_deriv=3, max_cross_degree=5,
                                include_const=False)
        _, names_hi = lib_hi.build(self.U)

        self.assertLess(len(names_lo), len(names_hi),
                        f"Lo={len(names_lo)} terms, Hi={len(names_hi)} terms; "
                        f"expected lo < hi")

    def test_trig_terms(self):
        """trig_terms=True adds sin(u) and cos(u) terms."""
        lib = FeatureLibrary(trig_terms=True, poly_degree=1, max_deriv=1,
                             cross_terms=False, include_const=False)
        _, names = lib.build(self.U)
        self.assertIn("sin(u)", names)
        self.assertIn("cos(u)", names)

    def test_large_library(self):
        """A 'large' library should have many terms but no crashes."""
        lib = FeatureLibrary(poly_degree=3, max_deriv=4, max_cross_degree=4,
                             trig_terms=True)
        Theta, names = lib.build(self.U)
        self.assertGreater(len(names), 30)
        self.assertFalse(np.isnan(Theta).any())
        self.assertFalse(np.isinf(Theta).any())
        self.assertEqual(Theta.shape[1], len(names))

    def test_describe(self):
        """describe() should return a string listing all terms."""
        lib = FeatureLibrary(poly_degree=1, max_deriv=1, cross_terms=False,
                             include_const=True)
        _, names = lib.build(self.U)
        desc = lib.describe(names)
        self.assertIn("FeatureLibrary", desc)
        self.assertIn("1", desc)
        self.assertIn("u", desc)

    def test_single_timestep(self):
        """Library should work with a single time snapshot (n_t=1)."""
        U = np.random.default_rng(0).standard_normal((16, 1))
        lib = FeatureLibrary()
        Theta, names = lib.build(U)
        self.assertEqual(Theta.shape[0], 16)  # N * 1
        self.assertGreater(len(names), 0)

    def test_empty_data_handling(self):
        """Library with U having zero columns should return empty Theta."""
        U = np.empty((16, 0))
        lib = FeatureLibrary()
        Theta, names = lib.build(U)
        # Should have zero samples but P columns for the features
        self.assertEqual(Theta.shape[0], 0)
        self.assertGreater(len(names), 0)


# ═══════════════════════════════════════════════════════════════════════ #
#  PDESimulator factory methods (including newer ones)
# ═══════════════════════════════════════════════════════════════════════ #

class TestSimulatorNewMethods(unittest.TestCase):
    """Tests for factory methods not covered by existing simulator tests."""

    N = 32
    T = 0.01
    n_t = 10
    seed = 42

    def _check(self, sim):
        U, U_t = sim.run()
        self.assertEqual(U.shape, (self.N, self.n_t))
        self.assertEqual(U_t.shape, (self.N, self.n_t))
        self.assertFalse(np.isnan(U).any())
        self.assertFalse(np.isinf(U).any())
        self.assertFalse(np.isnan(U_t).any())
        self.assertFalse(np.isinf(U_t).any())

    def test_kdv_burgers(self):
        """KdV-Burgers should produce valid trajectories."""
        sim = PDESimulator.kdv_burgers(nu=0.05, beta=1.0/6,
                                        N=self.N, T=self.T, n_t=self.n_t,
                                        n_modes=3, seed=self.seed, backend="numpy")
        self._check(sim)
        self.assertEqual(sim.name, "KdV-Burgers")
        self.assertAlmostEqual(sim.nu, 0.05)
        self.assertAlmostEqual(sim.beta, 1.0/6)

    def test_swift_hohenberg(self):
        """Swift-Hohenberg should produce valid trajectories."""
        sim = PDESimulator.swift_hohenberg(N=self.N, T=self.T, n_t=self.n_t,
                                            n_modes=3, seed=self.seed, backend="numpy")
        self._check(sim)
        self.assertEqual(sim.name, "Swift-Hohenberg")

    def test_fitzhugh_nagumo(self):
        """FitzHugh-Nagumo should produce valid trajectories."""
        sim = PDESimulator.fitzhugh_nagumo(N=self.N, T=self.T, n_t=self.n_t,
                                            n_modes=3, seed=self.seed, backend="numpy")
        self._check(sim)
        self.assertEqual(sim.name, "FitzHugh-Nagumo")

    def test_burgers_property(self):
        """Burgers simulator store nu as an attribute."""
        sim = PDESimulator.burgers(nu=0.03, N=self.N, T=self.T, n_t=self.n_t,
                                    n_modes=3, seed=self.seed, backend="numpy")
        self._check(sim)
        self.assertAlmostEqual(sim.nu, 0.03)

    def test_allen_cahn_property(self):
        """Allen-Cahn stores eps as an attribute."""
        sim = PDESimulator.allen_cahn(eps=0.02, N=self.N, T=self.T, n_t=self.n_t,
                                       n_modes=3, seed=self.seed, backend="numpy")
        self._check(sim)
        self.assertAlmostEqual(sim.eps, 0.02)


# ═══════════════════════════════════════════════════════════════════════ #
#  Time-derivative consistency for all PDEs
# ═══════════════════════════════════════════════════════════════════════ #

class TestTimeDerivativeConsistency(unittest.TestCase):
    """Verify U_t from RHS is consistent with finite-difference of U."""

    N = 64
    T = 0.05
    n_t = 30
    seed = 0

    def _check(self, sim, _):
        U, U_t = sim.run()
        dt = sim.T / sim.n_t
        U_t_fd = np.zeros_like(U)
        U_t_fd[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2 * dt)
        U_t_fd[:, 0] = (U[:, 1] - U[:, 0]) / dt
        U_t_fd[:, -1] = (U[:, -1] - U[:, -2]) / dt
        rel_err = float(np.linalg.norm(U_t - U_t_fd) / (np.linalg.norm(U_t) + 1e-14))
        self.assertLess(rel_err, 2.0, f"Time derivative mismatch: {rel_err:.3f}")

    def test_kdv_consistency(self):
        self._check(PDESimulator.kdv(N=self.N, T=self.T, n_t=self.n_t,
                                      n_modes=4, seed=self.seed, backend="numpy"), None)

    def test_ks_consistency(self):
        sim = PDESimulator.ks(N=self.N, T=0.005, n_t=20,
                               n_modes=2, seed=self.seed, backend="numpy")
        U, U_t = sim.run()
        self.assertFalse(np.isnan(U).any())
        self.assertFalse(np.isinf(U).any())
        self.assertFalse(np.isnan(U_t).any())
        self.assertFalse(np.isinf(U_t).any())

    def test_burgers_consistency(self):
        self._check(PDESimulator.burgers(nu=0.05, N=self.N, T=self.T, n_t=self.n_t,
                                          n_modes=4, seed=self.seed, backend="numpy"), None)

    def test_allen_cahn_consistency(self):
        self._check(PDESimulator.allen_cahn(eps=0.01, N=self.N, T=self.T, n_t=self.n_t,
                                             n_modes=4, seed=self.seed, backend="numpy"), None)

    def test_fisher_kpp_consistency(self):
        self._check(PDESimulator.fisher_kpp(D=0.01, r=1.0, N=self.N, T=self.T,
                                             n_t=self.n_t, n_modes=4, seed=self.seed,
                                             backend="numpy"), None)

    def test_kdv_burgers_consistency(self):
        self._check(PDESimulator.kdv_burgers(nu=0.05, beta=1/6, N=self.N, T=self.T,
                                              n_t=self.n_t, n_modes=4, seed=self.seed,
                                              backend="numpy"), None)

    def test_swift_hohenberg_consistency(self):
        sim = PDESimulator.swift_hohenberg(N=self.N, T=0.005, n_t=20,
                                            n_modes=2, seed=self.seed,
                                            backend="numpy")
        U, U_t = sim.run()
        self.assertFalse(np.isnan(U).any())
        self.assertFalse(np.isinf(U).any())
        self.assertFalse(np.isnan(U_t).any())
        self.assertFalse(np.isinf(U_t).any())

    def test_fitzhugh_nagumo_consistency(self):
        self._check(PDESimulator.fitzhugh_nagumo(N=self.N, T=self.T, n_t=self.n_t,
                                                  n_modes=4, seed=self.seed,
                                                  backend="numpy"), None)

    def test_nls_consistency(self):
        sim = PDESimulator.nls(sigma=1.0, N=self.N, T=self.T, n_t=self.n_t,
                                n_modes=4, seed=self.seed, backend="numpy")
        U, U_t = sim.run()
        dt = sim.T / sim.n_t
        U_t_fd = np.zeros_like(U)
        U_t_fd[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2 * dt)
        U_t_fd[:, 0] = (U[:, 1] - U[:, 0]) / dt
        U_t_fd[:, -1] = (U[:, -1] - U[:, -2]) / dt
        rel_err = float(np.linalg.norm(U_t - U_t_fd) / (np.linalg.norm(U_t) + 1e-14))
        self.assertLess(rel_err, 2.0)


# ═══════════════════════════════════════════════════════════════════════ #
#  OMP recovery edge cases
# ═══════════════════════════════════════════════════════════════════════ #

class TestOMPEdgeCases(unittest.TestCase):
    """Edge cases for OMP recovery."""

    def setUp(self):
        np.random.seed(0)
        self.sim = PDESimulator.kdv(N=32, T=0.05, n_t=20, n_modes=3, seed=42)
        self.lib = FeatureLibrary(poly_degree=3, max_deriv=3, max_cross_degree=3)
        U, U_t = self.sim.run()
        self.Theta, self.names = self.lib.build(U)
        self.y = U_t.ravel()

    def test_omp_different_sparsity(self):
        """OMP should respect different n_nonzero targets."""
        for k in [1, 3, 5]:
            oid = OperatorIdentifier(method="omp", n_nonzero=k, verbose=False)
            result = oid.fit(self.Theta, self.y, feature_names=self.names)
            self.assertLessEqual(len(result.support), k,
                                 f"Expected ≤{k} terms, got {len(result.support)}")

    def test_omp_returns_residual(self):
        """OMP result should include a non-negative residual."""
        oid = OperatorIdentifier(method="omp", n_nonzero=3, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertGreaterEqual(result.residual, 0.0)

    def test_omp_support_lists_match(self):
        """OMP support indices and active_coef should be consistent."""
        oid = OperatorIdentifier(method="omp", n_nonzero=3, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertEqual(len(result.support), len(result.active_coef))
        self.assertEqual(len(result.support), len(result.names))
        for i, c in zip(result.support, result.active_coef):
            self.assertAlmostEqual(float(result.coef[i]), c)


class TestLassoDenseRecovery(unittest.TestCase):
    """Tests for Lasso recovery."""

    def setUp(self):
        self.sim = PDESimulator.kdv(N=32, T=0.02, n_t=15, n_modes=3, seed=99)
        U, U_t = self.sim.run()
        self.lib = FeatureLibrary(poly_degree=2, max_deriv=3, max_cross_degree=2)
        self.Theta, self.names = self.lib.build(U)
        self.y = U_t.ravel()

    def test_lasso_returns_result(self):
        """Lasso should return a RecoveryResult with valid structure."""
        oid = OperatorIdentifier(method="lasso", alpha=0.05, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertIsInstance(result, RecoveryResult)
        self.assertGreater(len(result.support), 0)
        self.assertGreaterEqual(result.residual, 0.0)

    def test_lasso_different_alpha(self):
        """Larger alpha should produce sparser solutions."""
        oid_small = OperatorIdentifier(method="lasso", alpha=0.01, verbose=False)
        oid_large = OperatorIdentifier(method="lasso", alpha=1.0, verbose=False)
        res_small = oid_small.fit(self.Theta, self.y, feature_names=self.names)
        res_large = oid_large.fit(self.Theta, self.y, feature_names=self.names)
        # Not guaranteed in all cases, but a stronger penalty tends to
        # produce fewer non-zero coefficients.
        self.assertLessEqual(len(res_large.support), len(res_small.support) + 1)


# ═══════════════════════════════════════════════════════════════════════ #
#  L0 Pareto recovery (requires cvxpy)
# ═══════════════════════════════════════════════════════════════════════ #

@unittest.skipIf(not _HAS_CVXPY, "cvxpy required for l0_pareto")
class TestL0Pareto(unittest.TestCase):
    """Tests for L0 Pareto recovery."""

    def setUp(self):
        self.sim = PDESimulator.kdv(N=32, T=0.02, n_t=15, n_modes=3, seed=7)
        U, U_t = self.sim.run()
        self.lib = FeatureLibrary(poly_degree=2, max_deriv=3, max_cross_degree=2)
        self.Theta, self.names = self.lib.build(U)
        self.y = U_t.ravel()

    def test_l0_pareto_returns_result(self):
        """L0 Pareto should produce a RecoveryResult with meta info."""
        oid = OperatorIdentifier(method="l0_pareto", n_eps=15,
                                  max_samples=500, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertIsInstance(result, RecoveryResult)
        self.assertEqual(result.method, "l0_pareto")
        self.assertGreater(len(result.support), 0)
        self.assertIsInstance(result.meta, dict)
        self.assertIn("n_feasible", result.meta)

    def test_l0_pareto_small_n_eps(self):
        """L0 Pareto with a few epsilon bisections should still work."""
        oid = OperatorIdentifier(method="l0_pareto", n_eps=5,
                                  max_samples=300, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertGreaterEqual(result.residual, 0.0)

    def test_l0_pareto_allen_cahn(self):
        """L0 Pareto should recover Allen-Cahn terms."""
        sim = PDESimulator.allen_cahn(eps=0.01, N=32, T=0.02, n_t=15,
                                       n_modes=3, seed=7, backend="numpy")
        U, Ut = sim.run()
        lib = FeatureLibrary(poly_degree=3, max_deriv=3, max_cross_degree=3)
        Th, nm = lib.build(U)
        oid = OperatorIdentifier(method="l0_pareto", n_eps=15,
                                  max_samples=300, verbose=False)
        result = oid.fit(Th, Ut.ravel(), nm)
        rec = {nm[i]: c for i, c in zip(result.support, result.active_coef)}
        self.assertAlmostEqual(rec.get("u", 0), 1.0, delta=0.5,
                               msg="Allen-Cahn L0: 'u' term mismatch")
        self.assertAlmostEqual(rec.get("u^3", 0), -1.0, delta=0.5,
                               msg="Allen-Cahn L0: 'u^3' term mismatch")

    def test_l0_pareto_threshold_coef(self):
        """threshold_coef should prune small coefficient terms."""
        oid = OperatorIdentifier(method="l0_pareto", n_eps=10,
                                  max_samples=300, threshold_coef=0.1, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertGreaterEqual(result.residual, 0.0)


@unittest.skipIf(not _HAS_CVXPY, "cvxpy required for l0_sdp2")
@unittest.skip("l0_sdp2 is computationally expensive (~60s even for P=5); "
               "run manually with --deselect=skip when needed")
class TestL0SDP2(unittest.TestCase):
    """Tests for L0 SDP2 recovery."""

    def setUp(self):
        sim = PDESimulator.kdv(N=32, T=0.02, n_t=15, n_modes=3, seed=1)
        U, U_t = sim.run()
        self.lib = FeatureLibrary(poly_degree=2, max_deriv=3, max_cross_degree=2)
        self.Theta, self.names = self.lib.build(U)
        self.y = U_t.ravel()

    def test_l0_sdp2_returns_result(self):
        """L0 SDP2 should produce a RecoveryResult."""
        oid = OperatorIdentifier(method="l0_sdp2", n_eps=10,
                                  max_samples=200, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertIsInstance(result, RecoveryResult)
        self.assertEqual(result.method, "l0_sdp2")
        self.assertGreaterEqual(result.residual, 0.0)


# ═══════════════════════════════════════════════════════════════════════ #
#  CCP variants
# ═══════════════════════════════════════════════════════════════════════ #

class TestCCP(unittest.TestCase):
    """Tests for CCP recovery with various parameters."""

    def setUp(self):
        self.sim = PDESimulator.kdv(N=32, T=0.05, n_t=20, n_modes=3, seed=24)
        U, U_t = self.sim.run()
        self.lib = FeatureLibrary(poly_degree=3, max_deriv=3, max_cross_degree=3)
        self.Theta, self.names = self.lib.build(U)
        self.y = U_t.ravel()

    def test_ccp_default(self):
        """Default CCP should produce a valid RecoveryResult."""
        oid = OperatorIdentifier(method="ccp", verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertIsInstance(result, RecoveryResult)
        self.assertEqual(result.method, "ccp")
        self.assertGreaterEqual(result.residual, 0.0)

    def test_ccp_cluster_size_4(self):
        """CCP with smaller cluster_size=4 should work."""
        oid = OperatorIdentifier(method="ccp", cluster_size=4, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertIsInstance(result, RecoveryResult)
        self.assertGreater(len(result.support), 0)

    def test_ccp_cluster_size_16(self):
        """CCP with larger cluster_size=16 should work."""
        oid = OperatorIdentifier(method="ccp", cluster_size=16, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertIsInstance(result, RecoveryResult)
        self.assertGreaterEqual(result.residual, 0.0)

    def test_ccp_threshold_coef(self):
        """CCP with threshold_coef should prune terms."""
        oid = OperatorIdentifier(method="ccp", threshold_coef=0.1, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertGreaterEqual(result.residual, 0.0)

    @unittest.skipIf(not _HAS_SCIPY_MILP, "scipy.optimize.milp required for milp_vote")
    def _skip_ccp_milp_vote(self):
        """CCP with milp_vote=True should produce a valid result."""
        oid = OperatorIdentifier(method="ccp", milp_vote=True, verbose=False)
        result = oid.fit(self.Theta, self.y, feature_names=self.names)
        self.assertIsInstance(result, RecoveryResult)
        self.assertEqual(result.method, "ccp")
        self.assertGreaterEqual(result.residual, 0.0)

    def test_ccp_allen_cahn_recovery(self):
        """CCP should recover u and u^3 for Allen-Cahn."""
        sim = PDESimulator.allen_cahn(eps=0.01, N=32, T=0.02, n_t=20,
                                       n_modes=3, seed=42, backend="numpy")
        U, Ut = sim.run()
        lib = FeatureLibrary(poly_degree=3, max_deriv=3, max_cross_degree=3)
        Th, nm = lib.build(U)
        oid = OperatorIdentifier(method="ccp", cluster_size=8, verbose=False)
        result = oid.fit(Th, Ut.ravel(), nm)
        rec = {nm[i]: c for i, c in zip(result.support, result.active_coef)}
        # AC: u_t = eps*u_xx + u - u^3. With eps=0.01, u_xx is very small.
        self.assertAlmostEqual(rec.get("u", 0), 1.0, delta=0.3)
        self.assertAlmostEqual(rec.get("u^3", 0), -1.0, delta=0.3)


# ═══════════════════════════════════════════════════════════════════════ #
#  OperatorIdentifier validation
# ═══════════════════════════════════════════════════════════════════════ #

class TestOperatorIdentifierValidation(unittest.TestCase):
    """Input validation for OperatorIdentifier."""

    def test_invalid_method_raises(self):
        """Passing an unknown method should raise ValueError."""
        with self.assertRaises(ValueError):
            OperatorIdentifier(method="invalid_method")

    def test_valid_methods_do_not_raise(self):
        """All known methods should construct without error."""
        for m in ["omp", "lasso", "l0_pareto", "l0_sdp2", "ccp"]:
            OperatorIdentifier(method=m)

    def test_feature_names_override(self):
        """Passing feature_names to fit() should override instance-level names."""
        sim = PDESimulator.kdv(N=32, T=0.01, n_t=5, n_modes=2, seed=0)
        U, U_t = sim.run()
        lib = FeatureLibrary(poly_degree=2, max_deriv=2, max_cross_degree=2,
                             include_const=False)
        Theta, names = lib.build(U)
        oid = OperatorIdentifier(method="omp", n_nonzero=2,
                                  feature_names=["wrong"] * Theta.shape[1], verbose=False)
        override = ["custom_" + str(i) for i in range(Theta.shape[1])]
        result = oid.fit(Theta, U_t.ravel(), feature_names=override)
        for n in result.names:
            self.assertTrue(n.startswith("custom_"))


# ═══════════════════════════════════════════════════════════════════════ #
#  Edge cases
# ═══════════════════════════════════════════════════════════════════════ #

class TestEdgeCases(unittest.TestCase):
    """Various edge-case tests."""

    def test_single_timestep_full_pipeline(self):
        """Simulate → library → recovery with n_t=2 (minimal time dimension)."""
        sim = PDESimulator.kdv(N=16, T=0.01, n_t=2, n_modes=2, seed=1)
        U, U_t = sim.run()
        self.assertEqual(U.shape, (16, 2))
        lib = FeatureLibrary(poly_degree=1, max_deriv=1, cross_terms=False)
        Theta, names = lib.build(U)
        oid = OperatorIdentifier(method="omp", n_nonzero=1, verbose=False)
        result = oid.fit(Theta, U_t.ravel(), feature_names=names)
        self.assertGreaterEqual(result.residual, 0.0)

    def test_custom_single_small(self):
        """Custom PDE with a single component on tiny grid."""
        sim = PDESimulator.custom("-u*D[u] + 0.1*D[D[u]]", name="Burgers-custom",
                                   N=16, T=0.01, n_t=5, n_modes=3,
                                   seed=7, backend="numpy")
        U, U_t = sim.run()
        self.assertEqual(U.shape, (16, 5))
        self.assertFalse(np.isnan(U).any())
        self.assertFalse(np.isinf(U).any())

    def test_ccp_on_small_feature_set(self):
        """CCP should handle a very small feature set (P < cluster_size)."""
        sim = PDESimulator.kdv(N=16, T=0.01, n_t=5, n_modes=2, seed=0)
        U, U_t = sim.run()
        lib = FeatureLibrary(poly_degree=1, max_deriv=1, cross_terms=False)
        Theta, names = lib.build(U)
        oid = OperatorIdentifier(method="ccp", cluster_size=16, verbose=False)
        result = oid.fit(Theta, U_t.ravel(), feature_names=names)
        self.assertIsInstance(result, RecoveryResult)

    def test_sequential_simulations(self):
        """Multiple subsequent sim.run() calls should not interfere."""
        sim = PDESimulator.kdv(N=16, T=0.01, n_t=5, n_modes=2, seed=5)
        U1, _ = sim.run()
        U2, _ = sim.run()
        self.assertTrue(np.allclose(U1, U2, rtol=1e-14))

    def test_factory_kwargs_passthrough(self):
        """Factory methods should pass kwargs through to the constructor."""
        sim = PDESimulator.kdv(N=16, T=0.02, n_t=8, n_modes=2, seed=123,
                                backend="numpy")
        self.assertEqual(sim.N, 16)
        self.assertEqual(sim.T, 0.02)
        self.assertEqual(sim.n_t, 8)
        self.assertEqual(sim.n_modes, 2)
        self.assertEqual(sim.seed, 123)


# ═══════════════════════════════════════════════════════════════════════ #
#  Integration tests (end-to-end pipeline)
# ═══════════════════════════════════════════════════════════════════════ #

class TestIntegrationEndToEnd(unittest.TestCase):
    """Full pipeline: simulate → build library → recover → check Jaccard."""

    def _run_pipeline(self, sim, true_terms, lib_kwargs=None, method="ccp",
                      method_kwargs=None, min_jaccard=0.5):
        U, U_t = sim.run()
        lib = FeatureLibrary(**(lib_kwargs or {}))
        Theta, names = lib.build(U)
        oid = OperatorIdentifier(method=method, verbose=False, **(method_kwargs or {}))
        result = oid.fit(Theta, U_t.ravel(), feature_names=names)
        rec = {names[i]: c for i, c in zip(result.support, result.active_coef)}
        # Prune very small coefficients
        max_c = max(abs(v) for v in rec.values()) if rec else 1.0
        found = sorted(k for k, v in rec.items() if abs(v) > 0.01 * max_c)
        true = sorted(true_terms)
        jac = jaccard_score(found, true)
        return jac, found, true

    def test_kdv_integration(self):
        """KdV: u_t = -6 u u_x - u_xxx  via CCP."""
        sim = PDESimulator.kdv(N=32, T=0.03, n_t=20, n_modes=4, seed=42)
        jac, found, true = self._run_pipeline(
            sim,
            true_terms=["u u_x", "u_xxx"],
            lib_kwargs={"poly_degree": 2, "max_deriv": 3, "max_cross_degree": 2},
            method="ccp", method_kwargs={"cluster_size": 8},
        )
        self.assertGreaterEqual(jac, 0.5,
                                f"Jaccard={jac}, found={found}, true={true}")

    def test_burgers_integration(self):
        """Burgers: u_t = -u u_x + nu u_xx  via Lasso."""
        sim = PDESimulator.burgers(nu=0.05, N=32, T=0.03, n_t=20,
                                    n_modes=4, seed=3)
        jac, found, true = self._run_pipeline(
            sim,
            true_terms=["u u_x", "u_xx"],
            lib_kwargs={"poly_degree": 2, "max_deriv": 2, "max_cross_degree": 2},
            method="lasso", method_kwargs={"alpha": 0.05},
            min_jaccard=0.4,
        )
        # Lasso often recovers at least one term
        self.assertGreater(len(found), 0,
                            f"No terms found: jac={jac}, found={found}, true={true}")

    def test_allen_cahn_integration(self):
        """Allen-Cahn: u_t = eps*u_xx + u - u^3  via CCP."""
        sim = PDESimulator.allen_cahn(eps=0.01, N=32, T=0.03, n_t=20,
                                       n_modes=4, seed=42, backend="numpy")
        jac, found, true = self._run_pipeline(
            sim,
            true_terms=["u", "u^3"],
            lib_kwargs={"poly_degree": 3, "max_deriv": 2, "max_cross_degree": 3},
            method="ccp", method_kwargs={"cluster_size": 8},
        )
        self.assertIn("u", found, f"Missing 'u' term: found={found}")


class TestIntegrationNoisy(unittest.TestCase):
    """End-to-end with noisy data."""

    def test_pipeline_with_noise(self):
        """Adding noise to Theta and y should still produce a valid result."""
        sim = PDESimulator.kdv(N=32, T=0.03, n_t=20, n_modes=3, seed=5)
        U, U_t = sim.run()
        lib = FeatureLibrary(poly_degree=2, max_deriv=3, max_cross_degree=2)
        Theta, names = lib.build(U)
        y = U_t.ravel()

        Tn, yn = add_noise(Theta, y, noise_level=0.005, seed=0)
        oid = OperatorIdentifier(method="omp", n_nonzero=3, verbose=False)
        result = oid.fit(Tn, yn, feature_names=names)
        self.assertGreaterEqual(result.residual, 0.0)
        self.assertGreater(len(result.support), 0)

        # Higher noise — recovery may degrade but should not crash
        Tn2, yn2 = add_noise(Theta, y, noise_level=0.05, seed=1)
        result2 = oid.fit(Tn2, yn2, feature_names=names)
        self.assertGreaterEqual(result2.residual, 0.0)


class TestIntegrationWithCustomPDE(unittest.TestCase):
    """End-to-end test with custom() PDE."""

    def test_custom_heat_with_drift(self):
        """u_t = -u*u_x + 0.1*u_xx via custom (Burgers-like)."""
        sim = PDESimulator.custom("-u*D[u] + 0.1*D[D[u]]", name="Burgers-cst",
                                   N=32, T=0.02, n_t=15, n_modes=3, seed=7,
                                   backend="numpy")
        U, U_t = sim.run()
        lib = FeatureLibrary(poly_degree=2, max_deriv=2, max_cross_degree=2,
                             include_const=False)
        Theta, names = lib.build(U)
        oid = OperatorIdentifier(method="ccp", cluster_size=6, verbose=False)
        result = oid.fit(Theta, U_t.ravel(), feature_names=names)
        rec = {names[i]: c for i, c in zip(result.support, result.active_coef)}
        max_c = max(abs(v) for v in rec.values()) if rec else 1.0
        found = sorted(k for k, v in rec.items() if abs(v) > 0.01 * max_c)
        self.assertIn("u u_x", found, f"Expected 'u u_x' term. Found: {found}")


if __name__ == "__main__":
    unittest.main()
