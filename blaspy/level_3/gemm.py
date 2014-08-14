"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import (get_matrix_dimensions, create_zero_matrix, check_equal_sizes, convert_trans,
                       get_cblas_info, ROW_MAJOR)
from ..errors import raise_not_2d_numpy
from ctypes import c_int, POINTER


def gemm(A, B, C=None, trans_a='n', trans_b='n', alpha=1, beta=1, lda=None, ldb=None, ldc=None):
    """
    Perform a general matrix-vector multiplication operation.

    C := beta * C + alpha * A * B

    where alpha is a scalar and A, B, and C are general matrices.

    The trans_a and trans_b arguments allow the computation to proceed as if A and/or B is
    transposed.

    If matrix A is not provided, a zero matrix of the appropriate size and type is created
    and returned. In such a case, lda is automatically set to the number of columns in the
    newly created matrix A.

    Args:
        A:          2D numpy matrix or ndarray representing matrix A
        B:          2D numpy matrix or ndarray representing matrix A
        C:          2D numpy matrix or ndarray representing matrix C (default is zero matrix)
        trans_a:    'n'  if the operation is to proceed as if A is not transposed
                    't'  if the operation is to proceed as if A is transposed
        trans_b:    'n'  if the operation is to proceed as if A is not transposed
                    't'  if the operation is to proceed as if b is transposed
        alpha:      scalar alpha
        beta:       scalar beta
        lda:        leading dimension of A (must be >= # of cols in A)
        ldb:        leading dimension of B (must be >= # of cols in B)
        ldc:        leading dimension of C (must be >= # of cols in C)

    Returns:
        Matrix C, for use in case no matrix C was passed into this function.

    Raises:
        ValueError: if any of the following conditions occur:

                    - A, B, or C is not a 2D NumPy ndarray or NumPy matrix
                    - A, B, and C do not have the same dtype or that dtype is not supported
                    - The dimensions of A, B, and C do not conform
                    - Either trans_a or trans_b is not equal to one of the following: 'n', 'N',
                      't', 'T'
    """

    try:
        # get the dimensions of the parameters
        m_A, n_A = get_matrix_dimensions('A', A)
        m_B, n_B = get_matrix_dimensions('B', B)
        m, k_A = (m_A, n_A) if trans_a == 'n' else (n_A, m_A)
        n, k_B = (n_B, n_A) if trans_b == 'n' else (m_A, n_A)

        # if C is not given, create zero matrix with same type as A
        if C is None:
            C = create_zero_matrix(m, n, A.dtype, type(A))
            ldc = None

        # continue getting dimensions of the parameters
        m_C, n_C = get_matrix_dimensions('C', C)

        # assign a default value to lda, ldb, and ldc if necessary (assumes row-major order)
        if lda is None:
            lda = n_A
        if ldb is None:
            ldb = n_B
        if lda is None:
            ldc = n_C

        # convert to appropriate CBLAS value
        cblas_trans_a = convert_trans(trans_a)
        cblas_trans_b = convert_trans(trans_b)

        # ensure the matrix dimensions conform for the desired operation
        check_equal_sizes('A', k_A, 'B', k_B)
        check_equal_sizes('A', m, 'C', m_C)
        check_equal_sizes('B', n, 'C', n_C)

        # determine which CBLAS subroutine to call and which ctypes data type to use
        cblas_func, ctype_dtype = get_cblas_info('gemm', (A.dtype, B.dtype, C.dtype))

        # create a ctypes POINTER for each matrix
        ctype_A = POINTER(ctype_dtype * n_A * m_A)
        ctype_B = POINTER(ctype_dtype * n_B * m_B)
        ctype_C = POINTER(ctype_dtype * n_C * m_C)

        # call CBLAS using ctypes
        cblas_func.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, ctype_dtype,
                               ctype_A, c_int, ctype_B, c_int, ctype_dtype, ctype_C, c_int]
        cblas_func.restype = None
        cblas_func(ROW_MAJOR, cblas_trans_a, cblas_trans_b, m, n, k_A, alpha,
                   A.ctypes.data_as(ctype_A), lda, B.ctypes.data_as(ctype_B), ldb, beta,
                   C.ctypes.data_as(ctype_C), ldc)

        return C  # C is also overwritten, so only useful if no C was provided

    except AttributeError:
        raise_not_2d_numpy()