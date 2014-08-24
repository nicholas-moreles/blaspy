"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import (get_matrix_dimensions, get_vector_dimensions, check_strides_equal_one,
                       create_similar_zero_vector, check_equal_sizes, convert_trans,
                       get_cblas_info, ROW_MAJOR)
from ..errors import raise_generic_type_error
from ctypes import c_int, POINTER


def gemv(A, x, y=None, trans_a='n', alpha=1, beta=1, lda=None, inc_x=1, inc_y=1):
    """
    Perform a general matrix-vector multiplication operation.

    y := beta * y + alpha * A * x

    where alpha and beta are scalars, A is a general matrix, and x and y are general column vectors.

    The trans argument allows the operation to proceed as if A is transposed.

    Vectors x and y can be passed in as either row or column vectors. If necessary, an implicit
    transposition occurs.

    Vector y defaults to the zero vector of the appropriate size and type if vector y is not
    provided; however, the strides of x and y must be one if vector y is not provided.

    Args:
        A:          2D numpy matrix or ndarray representing matrix A
        x:          2D numpy matrix or ndarray representing vector x
        y:          2D numpy matrix or ndarray representing vector y (default is zero vector)
        trans_a:    'n'  if the operation is to proceed normally
                    't'  if the operation is to proceed as if A is transposed
        alpha:      scalar alpha
        beta:       scalar beta
        lda:        leading dimension of a (must be >= # of cols in A)
        inc_x:      stride of x (increment for the elements of x)
        inc_y:      stride of y (increment for the elements of y)

    Returns:
        Vector y, for use in case no vector y was passed into this function.

    Raises:
        ValueError: if any of the following conditions occur:

                    - A, x, or y is not a 2D NumPy ndarray or NumPy matrix
                    - A, x, and y do not have the same dtype or that dtype is not supported
                    - x or y is not a vector
                    - the effective length of either x or y does not conform to the dimensions of A
                    - y is not provided and the stride of either x or y does not equal one
                    - trans_a is not equal to one of the following: 'n', 'N', 't', 'T'
    """

    try:
        # convert to appropriate CBLAS value
        cblas_trans_a = convert_trans(trans_a)
        transpose_A = trans_a == 't' or trans_a == 'T'

        # get the dimensions of the parameters
        m_A, n_A = get_matrix_dimensions('A', A)
        m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)

        # if y is not given, create zero vector with same orientation and type as x
        if y is None:
            check_strides_equal_one(inc_x, inc_y)
            length = n_A if transpose_A else m_A
            y = create_similar_zero_vector(x, length)

        # continue getting dimensions of the parameters
        m_y, n_y, y_length = get_vector_dimensions('y', y, inc_y)

        # assign a default value to lda if necessary (assumes row-major order)
        if lda is None:
            lda = n_A

        # ensure the parameters are appropriate for the desired operation
        x_check, y_check = (n_A, m_A) if not transpose_A else (m_A, n_A)
        check_equal_sizes('A', x_check, 'x', x_length)
        check_equal_sizes('A', y_check, 'y', y_length)

        # determine which CBLAS subroutine to call and which ctypes data type to use
        cblas_func, ctype_dtype = get_cblas_info('gemv', (A.dtype, x.dtype, y.dtype))

        # create a ctypes POINTER for each vector and matrix
        ctype_x = POINTER(ctype_dtype * n_x * m_x)
        ctype_y = POINTER(ctype_dtype * n_y * m_y)
        ctype_A = POINTER(ctype_dtype * n_A * m_A)

        # call CBLAS using ctypes
        cblas_func.argtypes = [c_int, c_int, c_int, c_int, ctype_dtype, ctype_A, c_int,
                              ctype_x, c_int, ctype_dtype, ctype_y, c_int]
        cblas_func.restype = None
        cblas_func(ROW_MAJOR, cblas_trans_a, m_A, n_A, alpha, A.ctypes.data_as(ctype_A), lda,
                  x.ctypes.data_as(ctype_x), inc_x, beta, y.ctypes.data_as(ctype_y), inc_y)

        return y  # y is also overwritten, so only useful if no y was provided

    except (AttributeError, TypeError):
        raise_generic_type_error()