"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import get_vector_dimensions, get_matrix_dimensions, get_cblas_info, \
    check_equal_sizes, create_zero_matrix, ROW_MAJOR
from ctypes import c_int, POINTER


def ger(x, y, alpha=1, A=None, lda=None, inc_x=1, inc_y=1):
    """
    Perform a general rank-1 update operation.

    A := A + alpha * x * y

    where alpha is a scalar, A is a general matrix, x is a general column vector, and y is a
    general row vector.

    Vectors x and y can be row or column vectors. If necessary, an implicit transposition occurs.

    If matrix A is not provided, a zero vector of the appropriate size and type is created
    and returned. In such a case, lda is automatically set to the number of columns in the
    newly created matrix A.

    Args:
        x:        a 2D numpy matrix or ndarray representing vector x
        y:        a 2D numpy matrix or ndarray representing vector y (default is zero vector)
        alpha:    scalar alpha
        A:        a 2D numpy matrix or ndarray representing matrix A\
        lda:      leading dimension of A (must be >= # of cols in A)
        inc_x:    stride of x (increment for the elements of x)
        inc_y:    stride of y (increment for the elements of y)

    Returns:
        Matrix A, for use in case no matrix A was passed into this function.

     Raises:
        ValueError: if any of the following conditions occur:

                    - A, x, or y is not a 2D NumPy ndarray or NumPy matrix
                    - A, x, and y do not have the same dtype or that dtype is not supported
                    - x or y is not a vector
                    - the effective length of x or y does not conform to the dimensions of A
    """

    try:
        # get the dimensions of the parameters
        m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)
        m_y, n_y, y_length = get_vector_dimensions('y', y, inc_y)

        # if no matrix A is given, create a zero vector of appropriate size with the same dtype as x
        if A is None:
            A = create_zero_matrix(x_length, y_length, x.dtype, type(x))
            lda = None

        # continue getting dimensions of the parameters
        m_A, n_A = get_matrix_dimensions('A', A)

        if lda is None:
            lda = n_A  # row-major order is assumed

        # ensure the parameters are appropriate for the operation
        check_equal_sizes('A', m_A, 'x', x_length)
        check_equal_sizes('A', n_A, 'y', y_length)

        # determine which CBLAS subroutine to call based on matrix dtypes
        cblas_func, data_type = get_cblas_info('ger', A.dtype, x.dtype, y.dtype)

        # create ctypes POINTER for each matrix
        ctype_A = POINTER(data_type * n_A * m_A)
        ctype_x = POINTER(data_type * n_x * m_x)
        ctype_y = POINTER(data_type * n_y * m_y)

        # call CBLAS using ctypes
        cblas_func.argtypes = [c_int, c_int, c_int, data_type, ctype_x, c_int, ctype_y, c_int,
                              ctype_A, c_int]
        cblas_func.restype = None
        cblas_func(ROW_MAJOR, m_A, n_A, alpha, x.ctypes.data_as(ctype_x), inc_x,
                  y.ctypes.data_as(ctype_y), inc_y, A.ctypes.data_as(ctype_A), lda)

        return A  # A is also overwritten, so only useful if no A was provided

    except AttributeError:
        raise ValueError("A, x, and y must be of type numpy.ndarray or numpy.matrix")