"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import (get_vector_dimensions, get_square_matrix_dimension, get_cblas_info,
                       check_equal_sizes, create_zero_matrix, convert_uplo, ROW_MAJOR)
from ..errors import raise_not_2d_numpy
from ctypes import c_int, POINTER


def syr(x, A=None, uplo='u', alpha=1, lda=None, inc_x=1):
    """
    Perform a symmetric rank-1 update operation.

    A := A + alpha * x * x_T

    where alpha is a scalar, A is a symmetric matrix, and x is a general column vector.

    The uplo argument indicates whether the lower or upper triangle of A is to be referenced and
    updated by the operation.

    Vector x can be passed in as either row or column vector. If necessary, an implicit
    transposition occurs.

    If matrix A is not provided, a zero matrix of the appropriate size and type is created
    and returned. In such a case, lda is automatically set to the number of columns in the
    newly created matrix A.

    Args:
        x:        2D numpy matrix or ndarray representing vector x
        A:        2D numpy matrix or ndarray representing matrix A
        alpha:    scalar alpha
        lda:      leading dimension of A (must be >= # of cols in A)
        inc_x:    stride of x (increment for the elements of x)
        inc_y:    stride of y (increment for the elements of y)

    Returns:
        Matrix A, for use in case no matrix A was passed into this function.

     Raises:
        ValueError: if any of the following conditions occur:

                    - A or x is not a 2D NumPy ndarray or NumPy matrix
                    - A and x do not have the same dtype or that dtype is not supported
                    - x is not a vector
                    - the effective length of x does not equal the dimension of A
    """

    try:
        # get the dimensions of the parameters
        m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)

        # if no matrix A is given, create a zero matrix of appropriate size with the same dtype as x
        if A is None:
            A = create_zero_matrix(x_length, x_length, x.dtype, type(x))
            lda = None

        # continue getting dimensions of the parameters
        dim_A = get_square_matrix_dimension('A', A)

        # assign a default value to lda if necessary (assumes row-major order)
        if lda is None:
            lda = dim_A

        # ensure the parameters are appropriate for the operation
        check_equal_sizes('A', dim_A, 'x', x_length)

        # convert to appropriate CBLAS value
        cblas_uplo = convert_uplo(uplo)

        # determine which CBLAS subroutine to call and which ctypes data type to use
        cblas_func, data_type = get_cblas_info('syr', (A.dtype, x.dtype))

        # create a ctypes POINTER for each vector and matrix
        ctype_x = POINTER(data_type * n_x * m_x)
        ctype_A = POINTER(data_type * dim_A * dim_A)

        # call CBLAS using ctypes
        cblas_func.argtypes = [c_int, c_int, c_int, data_type, ctype_x, c_int, ctype_A, c_int]
        cblas_func.restype = None
        cblas_func(ROW_MAJOR, cblas_uplo, dim_A, alpha, x.ctypes.data_as(ctype_x), inc_x,
                   A.ctypes.data_as(ctype_A), lda)

        return A  # A is also overwritten, so only useful if no A was provided

    except AttributeError:
        raise_not_2d_numpy()