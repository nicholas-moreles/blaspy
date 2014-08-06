"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

# noinspection PyProtectedMember
from ..config import _libblas
from ..helpers import find_length, ROW_MAJOR, NO_TRANS, TRANS
from numpy import zeros, matrix, asmatrix
from ctypes import c_int, c_double, c_float, POINTER


# noinspection PyUnresolvedReferences
def gemv(A, x, alpha=1, trans_a='no_trans', y=None, beta=1, lda=None, inc_x=1, inc_y=1):
    """Perform a general matrix-vector multiplication operation.

    y := beta * y + alpha * A * x

    where alpha and beta are scalars, A is a general matrix, and x and y are general column vectors.

    The trans argument allows the computation to proceed as if A were transposed.

    Vectors x and y can be row or column vectors. If necessary, an implicit transposition occurs.

    Vector y defaults to the zero vector of the appropriate size and type if vector y is not
    provided; however, the strides of x and y must be one if vector y is not provided.

    Args:
        A:        a 2D numpy matrix or ndarray representing matrix A
        x:        a 2D numpy matrix or ndarray representing vector x
        alpha:    scalar alpha
        trans_a:  'no_trans'  if the multiplication is to proceed normally
                  'trans'     if the multiplication is to proceed as if A is transposed
        y:        a 2D numpy matrix or ndarray representing vector y (default is zero vector)
        lda:      leading dimension of a (must be >= # of cols in A)
        inc_x:    stride of x (increment for the elements of x)
        inc_y:    stride of y (increment for the elements of y)

    Returns:
        Vector y, for use in case no vector y was passed into this function.
    """

    try:
        # get the dimensions of the parameters
        m_A, n_A = A.shape
        m_x, n_x = x.shape

        # if no vector y is given, create a zero vector of appropriate size with the same dtype and
        # orientation as x; requires increment of x and y are 1
        if y is None:
            if inc_x != 1 or inc_y != 1:
                raise ValueError("vector y must be provided if the increment of vectors x or y "
                                 "are not equal to one")
            if m_x == 1:  # x is a row vector
                y = zeros((1, n_A if trans_a == 'trans' else m_A), dtype=x.dtype)
            else:
                y = zeros((n_A if trans_a == 'trans' else m_A, 1), dtype=x.dtype)
            if type(x) is matrix:
                y = asmatrix(y)

        # continue getting dimensions of the parameters
        m_y, n_y = y.shape
        x_length = find_length(m_x, n_x, inc_x)
        y_length = find_length(m_y, n_y, inc_y)

        # if no lda is given, set it equal to n_A (row-major order is assumed):
        if lda is None:
            lda = n_A

        # ensure the parameters are appropriate for the operation
        if trans_a == 'no_trans':
            trans_a = NO_TRANS  # enum for CBLAS
            if x_length != n_A:
                raise ValueError("size mismatch between A and x")
            if y_length != m_A:
                raise ValueError("size mismatch between A and y")
        elif trans_a == 'trans':
            trans_a = TRANS  # enum for CBLAS
            if x_length != m_A:
                raise ValueError("size mismatch between A and x")
            if y_length != n_A:
                raise ValueError("size mismatch between A and y")
        else:
            raise ValueError("trans_a must equal either 'trans' or 'no_trans'")
        if not (m_x == 1 or n_x == 1):
            raise ValueError("x must be a vector")
        if not (m_y == 1 or n_y == 1):
            raise ValueError("y must be a vector")

        # determine which BLAS routine to call based on data type
        if A.dtype == 'float64' and x.dtype == 'float64' and y.dtype == 'float64':
            blas_func = _libblas.cblas_dgemv
            data_type = c_double
        elif A.dtype == 'float32' and x.dtype == 'float32' and y.dtype == 'float32':
            blas_func = _libblas.cblas_sgemv
            data_type = c_float
        else:
            raise ValueError("A, x, and y must have the same dtype, either float64 or float32")

        # call BLAS using ctypes
        ctype_A = POINTER(data_type * n_A * m_A)
        ctype_x = POINTER(data_type * n_x * m_x)
        ctype_y = POINTER(data_type * n_y * m_y)
        blas_func.argtypes = [c_int, c_int, c_int, c_int, data_type, ctype_A, c_int,
                              ctype_x, c_int, data_type, ctype_y, c_int]
        blas_func.restype = None
        blas_func(ROW_MAJOR, trans_a, m_A, n_A, alpha, A.ctypes.data_as(ctype_A), lda,
                  x.ctypes.data_as(ctype_x), inc_x, beta, y.ctypes.data_as(ctype_y), inc_y)

        return y  # in case no vector y was provided

    except AttributeError:
        raise ValueError("A, x, and y must be of type numpy.ndarray or numpy.matrix")