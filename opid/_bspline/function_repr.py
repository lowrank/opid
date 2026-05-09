import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_array, kron
from scipy.sparse.linalg import lsqr
from opid._bspline import design_matrix

"""
Function Representation class.
"""

# b: b-spline
# w: wavelet
# n: neural network

__supported__ = {'b': 'linear', 'w': 'linear', 'n': 'nonlinear'}


class FunctionRepr(object):
    def __init__(self, basis_type='b'):
        self.type = basis_type
        self.linear = __supported__[basis_type] == 'linear'

    @classmethod
    def b_construct_1d_design_matrix(cls, eval_pts, knots, degree, derivative=0, extrapolate=False, periodic=False):
        """
        Create the design matrix for B splines.

        Args:
            eval_pts: point coordinates for evaluation
            knots: knot points coordinates
            degree: degree of B spline
            derivative: derivative order of B spline
            extrapolate: whether the knots form a periodic domain for extrapolation
            periodic: whether the b-splines are periodic
        Returns:
            sparse matrix of shape (num of eval_pts, knot number - (degree + 1))
        """
        b = make_design_matrix(eval_pts, knots, degree, derivative, extrapolate)
        if not periodic:
            return b
        else:
            b_full = b.tolil()
            for n in range(degree):
                b_full[:, n] = (b_full[:, n] + b_full[:, n - degree]).toarray()
            return b_full[:, :-degree].tocsr()

    @classmethod
    def b_construct_null_space(cls, design_matrix=None):
        """

        Args:
            design_matrix: design matrix of size (num of eval_pts, knot number - (degree + 1))

            QR decomposition may be slow. This is a one-time cost.

        Returns:
            null space matrix through QR decomposition.
        """
        if design_matrix.shape[0] < design_matrix.shape[1]:
            """
            The matrix is rank-deficient.
            """
            [q_mat, r_mat] = la.qr(design_matrix.transpose(), mode='full')
            return q_mat[:, r_mat.shape[1]:]
        else:
            return None

    @classmethod
    def b_1d_solve(cls, x_val, y_val, knots, degree, periodic, regularization=False, colloq=None, atol=1e-6, btol=1e-6):
        """

        Args:
            x_val: evaluation points coordinates
            y_val: values at evaluation points coordinates
            knots: B spline knots coordinates
            degree: degree of B spline
            periodic: whether the b-splines are periodic
            regularization: whether it uses regularization
            colloq: collocation points coordinates
            atol: stopping tolerance
            btol: stopping tolerance

            The tolerance definition are found at
            [scipy.sparse.linalg.lsqr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html)

        Returns:
            coefficient array of B spline basis.

        """
        design_matrix = cls.b_construct_1d_design_matrix(x_val, knots, degree, 0, False, periodic)
        c1 = lsqr(design_matrix, y_val, atol=atol, btol=btol)[0]
        # regularization with derivative (degree + 1) // 2, the representer theorem implies the minimizer should be
        # a combination of splines exactly.
        if regularization:
            null_space = cls.b_construct_null_space(design_matrix.toarray())
            smooth_matrix = cls.b_construct_1d_design_matrix(colloq, knots, degree, (degree + 1) // 2,
                                                             False, periodic)
            c2 = lsqr(smooth_matrix @ null_space, -smooth_matrix @ c1, atol=atol, btol=btol)[0]
            c1 += null_space @ c2
        return c1

    @classmethod
    def b_construct_nd_design_matrix(cls, eval_pts, knots, degree, derivative, extrapolate, periodic):
        """

        Args:

            eval_pts: ndarray(dim, n), multidimensional point coordinates for evaluation
            knots: List[List] knot points coordinates in each dimension
            degree: List[int],  degree of B spline in each dimension
            derivative: List[int],  derivative order of B spline in each dimension
            extrapolate: List[bool], whether the knots form a periodic domain for extrapolation
            periodic: List[bool], whether the b-splines are periodic

            Use tensor product of the basis as the multidimensional basis. The design matrix is the kron product in 2D.

        Returns:

        """
        dim = eval_pts.shape[0]
        if dim != 2:
            raise NotImplementedError('Only (1 + 1) dimension is supported in current version.')

        design_matrix = cls.b_construct_1d_design_matrix(eval_pts[0], knots[0], degree[0],
                                                         derivative[0], extrapolate[0], periodic[0])
        # the shape of design matrix is (num of eval, num of basis)
        # now do the tensor product with kron.
        for _dim in range(1, dim):
            tmp_matrix = cls.b_construct_1d_design_matrix(eval_pts[_dim], knots[_dim], degree[_dim],
                                                          derivative[_dim], extrapolate[_dim], periodic[_dim])
            design_matrix = kron(design_matrix, tmp_matrix)

        return design_matrix


# modified scipy's source code.
def make_design_matrix(x, t, k, nu, extrapolate):
    """

    Args:
        x: evaluation points
        t: knots
        k: degree
        nu: derivative
        extrapolate: bool

    Returns:
        design matrix [ B^{(nu)}_i(x_k) ]_{k, i}
    """
    x = np.ascontiguousarray(x).astype(np.float64, copy=False)
    t = np.ascontiguousarray(t).astype(np.float64, copy=False)

    if extrapolate != 'periodic':
        extrapolate = bool(extrapolate)

    if k < 0:
        raise ValueError("Spline order cannot be negative.")
    if t.ndim != 1 or np.any(t[1:] < t[:-1]):
        raise ValueError(f"Expect t to be a 1-D sorted array_like, but "
                         f"got t={t}.")
    # There are `nt - k - 1` basis elements in a BSpline built on the
    # vector of knots with length `nt`, so to have at least `k + 1` basis
    # elements we need to have at least `2 * k + 2` elements in the vector
    # of knots.
    if len(t) < 2 * k + 2:
        raise ValueError(f"Length t is not enough for k={k}.")

    if extrapolate == 'periodic':
        # With periodic extrapolation we map x to the segment
        # [t[k], t[n]].
        n = t.size - k - 1
        x = t[k] + (x - t[k]) % (t[n] - t[k])
        extrapolate = False
    elif not extrapolate and (
            (min(x) < t[k]) or (max(x) > t[t.shape[0] - k - 1])
    ):
        # Checks from `find_interval` function
        raise ValueError(f'Out of bounds w/ x = {x}.')

    # Compute number of non-zeros of final CSR array in order to determine
    # the dtype of indices and indptr of the CSR array.

    n = x.shape[0]
    nnz = n * (k + 1)
    if nnz < np.iinfo(np.int32).max:
        int_dtype = np.int32
    else:
        int_dtype = np.int64
    # Preallocate indptr and indices
    indices = np.empty(n * (k + 1), dtype=int_dtype)
    indptr = np.arange(0, (n + 1) * (k + 1), k + 1, dtype=int_dtype)

    # indptr is not passed to Cython as it is already fully computed
    data, indices = design_matrix.make_design_matrix(
        x, t, k, nu, extrapolate, indices
    )

    """
    @possible improvement: use dense array for first/last k columns if the b splines are periodic.
    """
    return csr_array(
        (data, indices, indptr),
        shape=(x.shape[0], t.shape[0] - k - 1)
    )
