"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

# noinspection PyProtectedMember
from ..config import _libblas
from ..helpers import find_length, ROW_MAJOR
from numpy import zeros, matrix, asmatrix
from ctypes import c_int, c_double, c_float, POINTER


# noinspection PyUnresolvedReferences
def ger(x, y, alpha=1, A=None, lda=None, inc_x=1, inc_y=1):
    """Perform a general rank-1 update operation.

    A := A + alpha * x * y

    where alpha is a scalar, A is a general matrix, x is a general column vector, and y is a
    general row vector.

    Vectors x and y can be row or column vectors. If necessary, an implicit transposition occurs.

    Matrix A defaults to the zero vector of the appropriate size and type if matrix A is not
    provided.

    Args:
        x:        a 2D numpy matrix or ndarray representing vector x
        y:        a 2D numpy matrix or ndarray representing vector y (default is zero vector)
        alpha:    scalar alpha
        A:        a 2D numpy matrix or ndarray representing matrix A\
        lda:      leading dimension of a (must be >= # of cols in A)
        inc_x:    stride of x (increment for the elements of x)
        inc_y:    stride of y (increment for the elements of y)

    Constraints:
        - the number of rows in matrix A must equal the length of vector x
        - the number of columns in matrix A must equal the length of vector y

    Returns:
        Matrix A, for use in case no vector y was passed into this function.
    """

    try:
        # get the dimensions of the parameters
        m_x, n_x = x.shape
        m_y, n_y = y.shape
        x_length = find_length(m_x, n_x, inc_x)
        y_length = find_length(m_y, n_y, inc_y)

        # if no matrix A is given, create a zero vector of appropriate size with the same dtype as x
        if A is None:
            A = zeros((x_length, y_length), dtype=x.dtype)
            lda = y_length
            if type(x) is matrix:
                A = asmatrix(A)
        m_A, n_A = A.shape

        # if no lda is given, set it equal to n_A (row-major order is assumed):
        if lda is None:
            lda = n_A

        # ensure the parameters are appropriate for the operation
        if x_length != m_A:
            raise ValueError("size mismatch between A and x")
        if y_length != n_A:
            raise ValueError("size mismatch between A and y")
        if not (m_x == 1 or n_x == 1):
            raise ValueError("x must be a vector")
        if not (m_y == 1 or n_y == 1):
            raise ValueError("y must be a vector")

        # determine which BLAS routine to call based on data type
        if A.dtype == 'float64' and x.dtype == 'float64' and y.dtype == 'float64':
            blas_func = _libblas.cblas_dger
            data_type = c_double
        elif A.dtype == 'float32' and x.dtype == 'float32' and y.dtype == 'float32':
            blas_func = _libblas.cblas_sger
            data_type = c_float
        else:
            raise ValueError("A, x, and y must have the same dtype, either float64 or float32")

        # call BLAS using ctypes
        ctype_A = POINTER(data_type * n_A * m_A)
        ctype_x = POINTER(data_type * n_x * m_x)
        ctype_y = POINTER(data_type * n_y * m_y)
        blas_func.argtypes = [c_int, c_int, c_int, data_type, ctype_x, c_int, ctype_y, c_int,
                              ctype_A, c_int]
        blas_func.restype = None
        blas_func(ROW_MAJOR, m_A, n_A, alpha, x.ctypes.data_as(ctype_x), inc_x,
                  y.ctypes.data_as(ctype_y), inc_y, A.ctypes.data_as(ctype_A), lda)

        return A  # in case no matrix A was provided

    except AttributeError:
        raise ValueError("A, x, and y must be of type numpy.ndarray or numpy.matrix")