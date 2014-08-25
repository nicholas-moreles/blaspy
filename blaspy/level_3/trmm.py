"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import (get_matrix_dimensions, get_square_matrix_dimension, check_equal_sizes,
                       convert_uplo, convert_side, convert_trans, convert_diag, get_cblas_info,
                       ROW_MAJOR)
from ..errors import raise_generic_type_error
from ctypes import c_int, POINTER


def trmm(A, B, side='l', uplo='u', trans_a='n', diag='n', alpha=1, lda=None, ldb=None):
    """
    Perform a triangular matrix-matrix multiplication operation.

    B := alpha * A * B
    or
    B :=  alpha * B * A

    where alpha is a scalar, A is a triangular matrix, and B is a general matrix.

    The 'side' argument indicates whether the symmetric matrix A is multiplied on the left or the
    right side of B. The 'uplo' argument indicates whether the lower or upper triangle of A is to be
    referenced by the operation. The 'trans_a' argument allows the computation to proceed as if A is
    transposed.

    Args:
        A:          2D numpy matrix or ndarray representing matrix A
        B:          2D numpy matrix or ndarray representing matrix A
        uplo:       'u'  if the upper triangular part of A is to be used
                    'l'  if the lower triangular part of A is to be used
        side:       'l'  if the operation is to proceed as if A is to the left of B
                    'r'  if the operation is to proceed as if A is to the right of B
        trans_a:    'n'  if the operation is to proceed as if A is not transposed
                    't'  if the operation is to proceed as if A is transposed
        diag:       'n'  if the diagonal of A is non-unit
                    'u'  if the diagonal of A is unit
        alpha:      scalar alpha
        lda:        leading dimension of A (must be >= # of cols in A)
        ldb:        leading dimension of B (must be >= # of cols in B)

    Raises:
        TypeError:  if A or B is not a 2D NumPy ndarray or NumPy matrix

        ValueError: if any of the following conditions occur:
                    - A or B do not have the same dtype or that dtype is not supported
                    - A is not a square matrix
                    - The dimensions of A and B do not conform
                    - 'uplo' is not equal to one of the following: 'u', 'U', 'l', 'L'
                    - 'side' is not equal to one of the following: 'l', 'L', 'r', 'R'
                    - 'trans_a' is not equal to one of the following: 'n', 'N', 't', 'T'
                    - 'diag' is not equal to one of the following: 'n', 'N', 'u', 'U'
    """

    try:
        # convert to appropriate CBLAS value
        cblas_uplo = convert_uplo(uplo)
        cblas_side = convert_side(side)
        cblas_trans_a = convert_trans(trans_a)
        cblas_diag = convert_diag(diag)
        side_is_left = side == 'l' or side == 'L'

        # get the dimensions of the parameters
        dim_A = get_square_matrix_dimension('A', A)
        m_B, n_B = get_matrix_dimensions('B', B)
        m, n, k = (dim_A, n_B, m_B) if side_is_left else (m_B, dim_A, n_B)

        # assign a default value to lda and ldb if necessary (assumes row-major order)
        if lda is None:
            lda = dim_A
        if ldb is None:
            ldb = n_B

        # ensure the matrix dimensions conform for the desired operation
        check_equal_sizes('A', dim_A, 'B', k)

        # determine which CBLAS subroutine to call and which ctypes data type to use
        cblas_func, ctype_dtype = get_cblas_info('trmm', (A.dtype, B.dtype))

        # create a ctypes POINTER for each matrix
        ctype_A = POINTER(ctype_dtype * dim_A * dim_A)
        ctype_B = POINTER(ctype_dtype * n_B * m_B)

        # call CBLAS using ctypes
        cblas_func.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, c_int, ctype_dtype,
                               ctype_A, c_int, ctype_B, c_int]
        cblas_func.restype = None
        cblas_func(ROW_MAJOR, cblas_side, cblas_uplo, cblas_trans_a, cblas_diag, m, n, alpha,
                   A.ctypes.data_as(ctype_A), lda, B.ctypes.data_as(ctype_B), ldb)

    except (AttributeError, TypeError):
        raise_generic_type_error()