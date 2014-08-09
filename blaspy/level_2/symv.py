"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import get_vector_dimensions, get_square_matrix_dimension, get_func_and_data_type, \
    check_equal_sizes, convert_uplo, ROW_MAJOR
from numpy import zeros, matrix, asmatrix
from ctypes import c_int, POINTER


def symv(A, x, y=None, uplo='u', alpha=1, beta=1, lda=None, inc_x=1, inc_y=1):
    """Perform a symmetric matrix-vector multiplication operation.

    y := beta * y + alpha * A * x

    where alpha and beta are scalars, A is a symmetric matrix, and x and y are general column
    vectors.

    The uplo argument indicates whether the lower or upper triangle of A is to be referenced by
    the operation.

    Vectors x and y can be row or column vectors. If necessary, an implicit transposition occurs.

    Vector y defaults to the zero vector of the appropriate size and type if vector y is not
    provided; however, the strides of x and y must be one if vector y is not provided.

    Args:
        A:        a 2D numpy matrix or ndarray representing matrix A
        x:        a 2D numpy matrix or ndarray representing vector x
        y:        a 2D numpy matrix or ndarray representing vector y
        uplo:     'u'   if the upper triangular part of A is to be used
                  'l'   if the lower triangular part of A is to be used
        alpha:    scalar alpha
        beta:     scalar beta
        lda:      leading dimension of a (must be >= # of cols in A)
        inc_x:    stride of x (increment for the elements of x)
        inc_y:    stride of y (increment for the elements of y)

    Constraints:
        - A must be a square matrix (number of rows equal to number of columns).
        - The length of vectors x and y must equal the dimension of A.

    Returns:
        Vector y, for use in case no vector y was passed into this function.
    """

    try:
        # get the dimensions of the parameters
        dim_A = get_square_matrix_dimension('A', A)
        m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)

        # if y is not given, create zero vector with same orientation and type as x
        if y is None:
            if inc_x != 1 or inc_y != 1:
                raise ValueError("vector y must be provided if the increment of vectors x or y "
                                 "are not equal to one")
            else:
                if m_x == 1:
                    y = zeros((1, dim_A), dtype=x.dtype)
                else:
                    y = zeros((dim_A, 1), dtype=x.dtype)

                if type(x) is matrix:
                    y = asmatrix(y)

        # continue getting dimensions of the parameters
        m_y, n_y, y_length = get_vector_dimensions('y', y, inc_y)

        # if no lda is given, set it equal to n_A (row-major order is assumed):
        if lda is None:
            lda = dim_A

        # ensure the parameters are appropriate for the operation
        check_equal_sizes('A', dim_A, 'x', x_length)
        check_equal_sizes('A', dim_A, 'y', y_length)

        # convert to appropriate CBLAS enum
        uplo = convert_uplo(uplo)

        # determine which BLAS routine to call based on data type
        blas_func, data_type = get_func_and_data_type('symv', A.dtype, x.dtype, y.dtype)

        # call BLAS using ctypes
        ctype_A = POINTER(data_type * dim_A * dim_A)
        ctype_x = POINTER(data_type * n_x * m_x)
        ctype_y = POINTER(data_type * n_y * m_y)
        blas_func.argtypes = [c_int, c_int, c_int, data_type, ctype_A, c_int, ctype_x, c_int,
                              data_type, ctype_y, c_int]
        blas_func.restype = None
        blas_func(ROW_MAJOR, uplo, dim_A, alpha, A.ctypes.data_as(ctype_A), lda,
                  x.ctypes.data_as(ctype_x), inc_x, beta, y.ctypes.data_as(ctype_y), inc_y)

        return y  # y is also overwritten, so only useful if no y was provided

    except AttributeError:
        raise ValueError("A, x, and y must be of type numpy.ndarray or numpy.matrix")