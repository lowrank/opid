"""
Routines for evaluating and manipulating B-splines.

Modified from scipy.

"""

cimport numpy as cnp

cimport cython

cimport numpy as cnp
cimport cython
import numpy as np

cnp.import_array()

cdef extern from "../src/__fitpack.h":
    void _deBoor_D(const double *t, double x, int k, int ell, int m, double *result) nogil

ctypedef double complex double_complex

ctypedef fused double_or_complex:
    double
    double complex

ctypedef fused int32_or_int64:
    cnp.npy_int32
    cnp.npy_int64

#------------------------------------------------------------------------------
# B-splines
#------------------------------------------------------------------------------

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int find_interval(const double[::1] t,
                       int k,
                       double xval,
                       int prev_l,
                       bint extrapolate) noexcept nogil:

    """
    Find an interval such that t[interval] <= xval < t[interval+1].
    
    Uses a linear search with locality, see fitpack's splev.
    
    Parameters
    ----------
    t : ndarray, shape (nt,)
        Knots
    k : int
        B-spline degree
    xval : double
        value to find the interval for
    prev_l : int
        interval where the previous value was located.
        if unknown, use any value < k to start the search.
    extrapolate : int
        whether to return the last or the first interval if xval
        is out of bounds.
    
    Returns
    -------
    interval : int
        Suitable interval or -1 if xval was nan.
    
    """
    cdef:
        int l
        int n = t.shape[0] - k - 1
        double tb = t[k]
        double te = t[n]

    if xval != xval:
        # nan
        return -1

    if ((xval < tb) or (xval > te)) and not extrapolate:
        return -1

    l = prev_l if k < prev_l < n else k

    # xval is in support, search for interval s.t. t[interval] <= xval < t[l+1]
    while(xval < t[l] and l != k):
        l -= 1

    l += 1
    while(xval >= t[l] and l != n):
        l += 1

    return l-1

@cython.wraparound(False)
@cython.boundscheck(False)
def make_design_matrix(const double[::1] x,
                        const double[::1] t,
                        int k,
                        int nu,
                        bint extrapolate,
                        int32_or_int64[::1] indices):
    """
    Returns a design matrix (at derivative nu) in CSR format.

    Note that only indices is passed, but not indptr because indptr is already
    precomputed in the calling Python function design_matrix.

    Parameters
    ----------
    x : array_like, shape (n,)
        Points to evaluate the spline at.
    t : array_like, shape (nt,)
        Sorted 1D array of knots.
    k : int
        B-spline degree.
    nu: int
        derivative order
    extrapolate : bool, optional
        Whether to extrapolate to ouf-of-bounds points.
    indices : ndarray, shape (n * (k + 1),)
        Preallocated indices of the final CSR array.

    Returns
    -------
    data
        The data array of a CSR array of the b-spline design matrix.
        In each row all the basis elements are evaluated at the certain point
        (first row - x[0], ..., last row - x[-1]).

    indices
        The indices array of a CSR array of the b-spline design matrix.
    """
    cdef:
        cnp.npy_intp i, j, m, ind
        cnp.npy_intp n = x.shape[0]
        double[::1] work = np.empty(2*k+2, dtype=float)
        double[::1] data = np.zeros(n * (k + 1), dtype=float)
        double xval
    ind = k

    for i in range(n):
        xval = x[i]

        # Find correct interval. Note that interval >= 0 always as
        # extrapolate=False and out of bound values are already dealt with in
        # design_matrix
        ind = find_interval(t, k, xval, ind, extrapolate)
        _deBoor_D(&t[0], xval, k, ind, nu, &work[0])

        # data[(k + 1) * i : (k + 1) * (i + 1)] = work[:k + 1]
        # indices[(k + 1) * i : (k + 1) * (i + 1)] = np.arange(ind - k, ind + 1)
        for j in range(k + 1):
            m = (k + 1) * i + j
            data[m] = work[j]
            indices[m] = ind - k + j

    return np.asarray(data), np.asarray(indices)