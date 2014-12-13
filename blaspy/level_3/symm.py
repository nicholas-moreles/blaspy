"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import (get_matrix_dimensions, get_square_matrix_dimension, create_zero_matrix,
                       check_equal_sizes, convert_uplo, convert_side, get_cblas_info, ROW_MAJOR,
                       LEFT)
from ctypes import c_int, POINTER


def symm(A, B, C=None, side='l', uplo='u', alpha=1.0, beta=1.0, lda=None, ldb=None, ldc=None):
    """
    Perform a symmertric matrix-matrix multiplication operation.

    C := beta * C + alpha * A * B
    or
    C := beta * C + alpha * B * A

    where alpha and beta are scalars, A is a symmetric matrix, and B and C are general matrices.

    The 'side' argument indicates whether the symmetric matrix A is multiplied on the left or the
    right side of B. The 'uplo' argument indicates whether the lower or upper triangle of A is to be
    referenced by the operation.

    If matrix C is not provided, a zero matrix of the appropriate size and type is created
    and returned. In such a case, 'ldc' is automatically set to the number of columns in the
    newly created matrix C.

    Args:
        A:        2D NumPy matrix or ndarray representing matrix A
        B:        2D NumPy matrix or ndarray representing matrix A

        --optional arguments--

        C:        2D NumPy matrix or ndarray representing matrix C
                      < default is the zero matrix >
        side:     'l'  if the operation is to proceed as if A is to the left of B
                  'r'  if the operation is to proceed as if A is to the right of B
                      < default is 'l' >
        uplo:     'u'  if the upper triangular part of A is to be used
                  'l'  if the lower triangular part of A is to be used
                      < default is 'u' >
        alpha:    scalar alpha
                      < default is 1.0 >
        beta:     scalar beta
                      < default is 1.0 >
        lda:      leading dimension of A (must be >= # of cols in A)
                      < default is the number of columns in A >
        ldb:      leading dimension of B (must be >= # of cols in B)
                      < default is the number of columns in B >
        ldc:      leading dimension of C (must be >= # of cols in C)
                      < default is the number of columns in C >

    Returns:
        Matrix C (which is also overwritten)

    Raises:
        ValueError: if any of the following conditions occur:
                    - A, B, or C is not a 2D NumPy ndarray or NumPy matrix
                    - A, B, and C do not have the same dtype or that dtype is not supported
                    - A is not a square matrix
                    - the dimensions of A, B, and C do not conform
                    - 'uplo' is not equal to one of the following: 'u', 'U', 'l', 'L'
                    - 'side' is not equal to one of the following: 'l', 'L', 'r', 'R'
    """

    # convert to appropriate CBLAS value
    cblas_uplo = convert_uplo(uplo)
    cblas_side = convert_side(side)
    side_is_left = cblas_side == LEFT

    # get the dimensions of the parameters
    dim_A = get_square_matrix_dimension('A', A)
    m_B, n_B = get_matrix_dimensions('B', B)
    m, n, k = (dim_A, n_B, m_B) if side_is_left else (m_B, dim_A, n_B)

    # if C is not given, create zero matrix with same type as A
    if C is None:
        C = create_zero_matrix(m, n, A.dtype, type(A))
        ldc = None

    # continue getting dimensions of the parameters
    m_C, n_C = get_matrix_dimensions('C', C)

    # assign a default value to lda, ldb, and ldc if necessary (assumes row-major order)
    if lda is None:
        lda = dim_A
    if ldb is None:
        ldb = n_B
    if ldc is None:
        ldc = n_C

    # ensure the matrix dimensions conform for the desired operation
    check_equal_sizes('A', dim_A, 'B', k)
    check_equal_sizes('A' if side_is_left else 'B', m, 'C', m_C)
    check_equal_sizes('B' if side_is_left else 'A', n, 'C', n_C)

    # determine which CBLAS subroutine to call and which ctypes data type to use
    cblas_func, ctype_dtype = get_cblas_info('symm', (A.dtype, B.dtype, C.dtype))

    # create a ctypes POINTER for each matrix
    ctype_A = POINTER(ctype_dtype * dim_A * dim_A)
    ctype_B = POINTER(ctype_dtype * n_B * m_B)
    ctype_C = POINTER(ctype_dtype * n_C * m_C)

    # call CBLAS using ctypes
    cblas_func.argtypes = [c_int, c_int, c_int, c_int, c_int, ctype_dtype,
                           ctype_A, c_int, ctype_B, c_int, ctype_dtype, ctype_C, c_int]
    cblas_func.restype = None
    cblas_func(ROW_MAJOR, cblas_side, cblas_uplo, m, n, alpha,
               A.ctypes.data_as(ctype_A), lda, B.ctypes.data_as(ctype_B), ldb, beta,
               C.ctypes.data_as(ctype_C), ldc)

    return C  # C is also overwritten