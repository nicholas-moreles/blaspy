"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import (get_square_matrix_dimension, get_vector_dimensions, check_strides_equal_one,
                       create_similar_zero_vector, check_equal_sizes, convert_uplo, get_cblas_info,
                       ROW_MAJOR)
from ctypes import c_int, POINTER


def symv(A, x, y=None, uplo='u', alpha=1, beta=1, lda=None, inc_x=1, inc_y=1):
    """
    Perform a symmetric matrix-vector multiplication operation.

    y := beta * y + alpha * A * x

    where alpha and beta are scalars, A is a symmetric matrix, and x and y are general column
    vectors.

    The uplo argument indicates whether the lower or upper triangle of A is to be referenced by
    the operation.

    Vectors x and y can be passed in as either row or column vectors. If necessary, an implicit
    transposition occurs.

    Vector y defaults to the zero vector of the appropriate size and type if vector y is not
    provided; however, the strides of x and y must be one if vector y is not provided.

    Args:
        A:        a 2D numpy matrix or ndarray representing matrix A
        x:        a 2D numpy matrix or ndarray representing vector x
        y:        a 2D numpy matrix or ndarray representing vector y
        uplo:     'u'  if the upper triangular part of A is to be used
                  'l'  if the lower triangular part of A is to be used
        alpha:    scalar alpha
        beta:     scalar beta
        lda:      leading dimension of A (must be >= # of cols in A)
        inc_x:    stride of x (increment for the elements of x)
        inc_y:    stride of y (increment for the elements of y)

    Returns:
        Vector y, for use in case no vector y was passed into this function.

    Raises:
        ValueError: if any of the following conditions occur:

                    - A, x, or y is not a 2D NumPy ndarray or NumPy matrix
                    - A, x, and y do not have the same dtype or that dtype is not supported
                    - A is not a square matrix
                    - x or y is not a vector
                    - the effective length of x or y do not equal the dimension of A
                    - y is not provided and the stride of either x or y does not equal one
                    - uplo is not equal to one of the following: 'u', 'U', 'l', 'L'
    """

    try:
        # get the dimensions of the parameters
        dim_A = get_square_matrix_dimension('A', A)
        m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)

        # if y is not given, create a zero vector with same orientation and type as x
        if y is None:
            check_strides_equal_one(inc_x, inc_y)
            y = create_similar_zero_vector(x, dim_A)

        # continue getting dimensions of the parameters
        m_y, n_y, y_length = get_vector_dimensions('y', y, inc_y)

        # assign a default value to lda if necessary (assumes row-major order)
        if lda is None:
            lda = dim_A

        # ensure the parameters are appropriate for the desired operation
        check_equal_sizes('A', dim_A, 'x', x_length)
        check_equal_sizes('A', dim_A, 'y', y_length)

        # convert to appropriate CBLAS value
        cblas_uplo = convert_uplo(uplo)

        # determine which CBLAS subroutine to call and which ctypes data type to use
        cblas_func, ctype_dtype = get_cblas_info('symv', A.dtype, x.dtype, y.dtype)

        # create a ctypes POINTER for each matrix
        ctype_A = POINTER(ctype_dtype * dim_A * dim_A)
        ctype_x = POINTER(ctype_dtype * n_x * m_x)
        ctype_y = POINTER(ctype_dtype * n_y * m_y)

        # call CBLAS using ctypes
        cblas_func.argtypes = [c_int, c_int, c_int, ctype_dtype, ctype_A, c_int, ctype_x, c_int,
                               ctype_dtype, ctype_y, c_int]
        cblas_func.restype = None
        cblas_func(ROW_MAJOR, cblas_uplo, dim_A, alpha, A.ctypes.data_as(ctype_A), lda,
                   x.ctypes.data_as(ctype_x), inc_x, beta, y.ctypes.data_as(ctype_y), inc_y)

        return y  # y is also overwritten, so only useful if no y was provided

    except AttributeError:
        raise ValueError("Either A, x, or y is not a 2D NumPy ndarray or matrix.")