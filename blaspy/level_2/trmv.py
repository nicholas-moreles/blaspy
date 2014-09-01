"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import (get_vector_dimensions, get_square_matrix_dimension, get_cblas_info,
                       check_equal_sizes, convert_uplo, convert_trans, convert_diag, ROW_MAJOR)
from ..errors import raise_generic_type_error
from ctypes import c_int, POINTER


def trmv(A, x, uplo='u', trans_a='n', diag='n', lda=None, inc_x=1):
    """
    Perform a triangular matrix-vector multiplication operation.

    x := A * x

    where alpha is a scalar, A is a symmetric matrix, and x is a general column vector.

    The 'uplo' argument indicates whether the lower or upper triangle of A is to be referenced and
    updated by the operation. The 'trans_a' argument allows the computation to proceed as if A is
    transposed. The 'diag' argument indicates whether the diagonal of A is unit or non-unit.

    Vector x can be passed in as either row or column vector. If necessary, an implicit
    transposition occurs.

    IMPORTANT: This function does not test for singularity or near-singularity.
    Such tests should be performed prior to calling this function.

    Args:
        x:          2D numpy matrix or ndarray representing vector x
        A:          2D numpy matrix or ndarray representing matrix A
        uplo:       'u'  if the upper triangular part of A is to be used
                    'l'  if the lower triangular part of A is to be used
        trans_a:    'n'  if the operation is to proceed normally
                    't'  if the operation is to proceed as if A is transposed
        diag:       'n'  if the diagonal of A is non-unit
                    'u'  if the diagonal of A is unit
        lda:        leading dimension of A (must be >= # of cols in A)
        inc_x:      stride of x (increment for the elements of x)

    Raises:
        TypeError:  if either A or x is not a 2D NumPy ndarray or NumPy matrix

        ValueError: if any of the following conditions occur:
                    - A and x do not have the same dtype or that dtype is not supported
                    - A is not a square matrix
                    - x is not a vector
                    - the effective length of x does not equal the dimension of A
                    - uplo is not equal to one of the following: 'u', 'U', 'l', 'L'
                    - trans_a is not equal to one of the following: 'n', 'N', 't', 'T'
                    - diag is not equal to one fo the following: 'n', 'N', 'u', 'U'
    """

    try:
        # get the dimensions of the parameters
        m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)
        dim_A = get_square_matrix_dimension('A', A)

        # assign a default value to lda if necessary (assumes row-major order)
        if lda is None:
            lda = dim_A

        # ensure the parameters are appropriate for the operation
        check_equal_sizes('A', dim_A, 'x', x_length)

        # convert to appropriate CBLAS values
        cblas_uplo = convert_uplo(uplo)
        cblas_trans_a = convert_trans(trans_a)
        cblas_diag = convert_diag(diag)

        # determine which CBLAS subroutine to call and which ctypes data type to use
        cblas_func, data_type = get_cblas_info('trmv', (A.dtype, x.dtype))

        # create a ctypes POINTER for each vector and matrix
        ctype_x = POINTER(data_type * n_x * m_x)
        ctype_A = POINTER(data_type * dim_A * dim_A)

        # call CBLAS using ctypes
        cblas_func.argtypes = [c_int, c_int, c_int, c_int, c_int, ctype_A, c_int, ctype_x, c_int]
        cblas_func.restype = None
        cblas_func(ROW_MAJOR, cblas_uplo, cblas_trans_a, cblas_diag, dim_A,
                   A.ctypes.data_as(ctype_A), lda, x.ctypes.data_as(ctype_x), inc_x)

    except (AttributeError, TypeError):
        raise_generic_type_error()