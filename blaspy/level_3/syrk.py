"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import (get_matrix_dimensions, get_square_matrix_dimension, create_zero_matrix,
                       check_equal_sizes, convert_uplo, convert_trans, get_cblas_info, ROW_MAJOR,
                       TRANS)
from ctypes import c_int, POINTER


def syrk(A, C=None, uplo='u', trans='n', alpha=1.0, beta=1.0, lda=None, ldc=None):
    """
    Perform a symmetric rank-k update operation.

    C := beta * C + alpha * A * A_T  [trans='n']
    or
    C := beta * C + alpha * A_T * A  [trans='t']

    where alpha and beta are scalars, A is a general matrix, and C is a symmetric matrix.

    The 'trans' argument allows the computation to proceed as if A is transposed, resulting in the
    alternate rank-k product (second computation listed above). The 'uplo' argument indicates
    whether the lower or upper triangle of C is to be referenced and updated by the operation.

    If matrix C is not provided, a zero matrix of the appropriate size and type is created
    and returned. In such a case, 'ldc' is automatically set to the number of columns in the
    newly created matrix C.

    Args:
        A:       2D numpy matrix or ndarray representing matrix A

        --optional arguments--

        C:        2D numpy matrix or ndarray representing matrix C
                      < default is the zero matrix >
        uplo:     'u'  if the upper triangular part of C is to be used/overwritten
                  'l'  if the lower triangular part of C is to be used/overwritten
                      < default is 'u' >
        trans:    'n'  if the operation is to proceed as if A is not transposed
                  't'  if the operation is to proceed as if A is transposed
                      < default is 'n' >
        alpha:    scalar alpha
                      < default is 1.0 >
        beta:     scalar beta
                      < default is 1.0 >
        lda:      leading dimension of A (must be >= # of cols in A)
                      < default is the number of columns in A >
        ldc:      leading dimension of C (must be >= # of cols in C)
                      < default is the number of columns in C >

    Returns:
        matrix C (which is also overwritten)

    Raises:
        ValueError: if any of the following conditions occur:
                    - A or C is not a 2D NumPy ndarray or NumPy matrix
                    - A and C do not have the same dtype or that dtype is not supported
                    - C is not a square matrix
                    - The dimensions of A and C do not conform
                    - 'uplo' is not equal to one of the following: 'u', 'U', 'l', 'L'
                    - 'trans' is not equal to one of the following: 'n', 'N', 't', 'T'
    """

    # convert to appropriate CBLAS value
    cblas_uplo = convert_uplo(uplo)
    cblas_trans = convert_trans(trans)
    transpose_a = cblas_trans == TRANS

    # get the dimensions of the parameters
    m_A, n_A = get_matrix_dimensions('A', A)
    n, k = (m_A, n_A) if not transpose_a else (n_A, m_A)

    # if C is not given, create zero matrix with same type as A
    if C is None:
        C = create_zero_matrix(n, n, A.dtype, type(A))
        ldc = None

    # continue getting dimensions of the parameters
    dim_C = get_square_matrix_dimension('C', C)

    # assign a default value to lda and ldc if necessary (assumes row-major order)
    if lda is None:
        lda = n_A
    if ldc is None:
        ldc = dim_C

    # ensure the matrix dimensions conform for the desired operation
    check_equal_sizes('A', n, 'C', dim_C)

    # determine which CBLAS subroutine to call and which ctypes data type to use
    cblas_func, ctype_dtype = get_cblas_info('syrk', (A.dtype, C.dtype))

    # create a ctypes POINTER for each matrix
    ctype_A = POINTER(ctype_dtype * n_A * m_A)
    ctype_C = POINTER(ctype_dtype * dim_C * dim_C)

    # call CBLAS using ctypes
    cblas_func.argtypes = [c_int, c_int, c_int, c_int, c_int, ctype_dtype,
                           ctype_A, c_int, ctype_dtype, ctype_C, c_int]
    cblas_func.restype = None
    cblas_func(ROW_MAJOR, cblas_uplo, cblas_trans, n, k, alpha,
               A.ctypes.data_as(ctype_A), lda, beta, C.ctypes.data_as(ctype_C), ldc)

    return C  # C is also overwritten