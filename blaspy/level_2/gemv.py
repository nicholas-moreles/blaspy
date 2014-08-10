"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

# noinspection PyProtectedMember
from ..config import _libblas
from ..helpers import ROW_MAJOR, get_matrix_dimensions, get_vector_dimensions, check_equal_sizes, \
    create_similar_zero_vector, convert_trans, get_func_and_data_type
from ctypes import c_int, c_double, c_float, POINTER


def gemv(A, x, alpha=1, trans_a='no_trans', y=None, beta=1, lda=None, inc_x=1, inc_y=1):
    """Perform a general matrix-vector multiplication operation.

    y := beta * y + alpha * A * x

    where alpha and beta are scalars, A is a general matrix, and x and y are general column vectors.

    The trans argument allows the operation to proceed as if A were transposed.

    Vectors x and y can be row or column vectors. If necessary, an implicit transposition occurs.

    Vector y defaults to the zero vector of the appropriate size and type if vector y is not
    provided; however, the strides of x and y must be one if vector y is not provided.

    Args:
        A:        a 2D numpy matrix or ndarray representing matrix A
        x:        a 2D numpy matrix or ndarray representing vector x
        alpha:    scalar alpha
        trans_a:  'n'  if the multiplication is to proceed normally
                  't'     if the multiplication is to proceed as if A is transposed
        y:        a 2D numpy matrix or ndarray representing vector y (default is zero vector)
        lda:      leading dimension of a (must be >= # of cols in A)
        inc_x:    stride of x (increment for the elements of x)
        inc_y:    stride of y (increment for the elements of y)

    Returns:
        Vector y, for use in case no vector y was passed into this function.
    """

    try:
        # get the dimensions of the parameters
        m_A, n_A = get_matrix_dimensions('A', A)
        m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)

        # if no vector y is given, create a zero vector of appropriate size with the same dtype and
        # orientation as x; requires increment of x and y are 1

        if y is None:
            if inc_x != 1 or inc_y != 1:
                raise ValueError("vector y must be provided if the increment of vectors x or y "
                                 "are not equal to one")
            else:
                length = n_A if trans_a == 't' else m_A
                y = create_similar_zero_vector(x, length)

        # continue getting dimensions of the parameters
        m_y, n_y, y_length = get_vector_dimensions('y', y, inc_y)

        # if no lda is given, set it equal to n_A (row-major order is assumed):
        if lda is None:
            lda = n_A

        # convert to appropriate CBLAS enum
        cblas_trans_a = convert_trans(trans_a)

        # ensure the parameters are appropriate for the operation
        if cblas_trans_a & 1:  # no transpose
            x_check, y_check = n_A, m_A
        else:  # transpose
            x_check, y_check = m_A, n_A
        check_equal_sizes('A', x_check, 'x', x_length)
        check_equal_sizes('A', y_check, 'y', y_length)

        # determine which CBLAS routine to call based on parameter dtypes
        cblas_func, data_type = get_func_and_data_type('gemv', A.dtype, x.dtype, y.dtype)

        # create ctype POINTER for each matrix
        ctype_A = POINTER(data_type * n_A * m_A)
        ctype_x = POINTER(data_type * n_x * m_x)
        ctype_y = POINTER(data_type * n_y * m_y)

        # call BLAS using ctypes
        cblas_func.argtypes = [c_int, c_int, c_int, c_int, data_type, ctype_A, c_int,
                              ctype_x, c_int, data_type, ctype_y, c_int]
        cblas_func.restype = None
        cblas_func(ROW_MAJOR, cblas_trans_a, m_A, n_A, alpha, A.ctypes.data_as(ctype_A), lda,
                  x.ctypes.data_as(ctype_x), inc_x, beta, y.ctypes.data_as(ctype_y), inc_y)

        return y  # y is also overwritten, so only useful if no y was provided

    except AttributeError:
        raise ValueError("A, x, and y must be of type numpy.ndarray or numpy.matrix")