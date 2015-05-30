"""

    Copyright (c) 2014-2015-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import (get_matrix_dimensions, get_vector_dimensions, check_strides_equal_one,
                       create_similar_zero_vector, check_equal_sizes, convert_trans,
                       get_cblas_info, ROW_MAJOR, TRANS)
from ctypes import c_int, POINTER


def gemv(A, x, y=None, trans_a='n', alpha=1.0, beta=1.0, lda=None, inc_x=1, inc_y=1):
    """
    Perform a general matrix-vector multiplication operation.

    y := beta * y + alpha * A * x

    where alpha and beta are scalars, A is a general matrix, and x and y are general column vectors.

    The 'trans' argument allows the operation to proceed as if A is transposed.

    Vectors x and y can be passed in as either row or column vectors. If necessary, an implicit
    transposition occurs.

    Vector y defaults to the zero vector of the appropriate size, orientation, and type if vector y is not
    provided; however, the stride of y becomes fixed at 1 and the parameter inc_y is ignored.

    Args:
        A:          2D NumPy matrix or ndarray representing matrix A
        x:          2D NumPy matrix or ndarray representing vector x

        --optional arguments--

        y:          2D NumPy matrix or ndarray representing vector y
                        < default is the zero vector >
        trans_a:    'n'  if the operation is to proceed normally
                    't'  if the operation is to proceed as if A is transposed
                        < default is 'n' >
        alpha:      scalar alpha
                        < default is 1.0 >
        beta:       scalar beta
                        < default is 1.0 >
        lda:        leading dimension of a (must be >= # of columns in A)
                        < default is the number of columns in A >
        inc_x:      stride of x (increment for the elements of x)
                        < default is 1 >
        inc_y:      stride of y (increment for the elements of y)
                        < default is 1 >

    Returns:
        Vector y (which is also overwritten)

    Raises:
        ValueError: if any of the following conditions occur:
                    - A, x, or y is not a 2D NumPy ndarray or NumPy matrix
                    - A, x, and y do not have the same dtype or that dtype is not supported
                    - x or y is not a vector
                    - the effective length of either x or y does not conform to the dimensions of A
                    - trans_a is not equal to one of the following: 'n', 'N', 't', 'T'
    """

    # convert to appropriate CBLAS value
    cblas_trans_a = convert_trans(trans_a)
    transpose_A = cblas_trans_a == TRANS

    # get the dimensions of the parameters
    m_A, n_A = get_matrix_dimensions('A', A)
    m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)

    # if y is not given, create zero vector with same orientation as x that conforms to matrix A
    if y is None:
        inc_y = 1  # stride of unprovided vector y is set to 1, there is currently no option for the user to provide this
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

    return y  # y is also overwritten