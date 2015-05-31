"""

    Copyright (c) 2014-2015-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import (get_vector_dimensions, get_square_matrix_dimension, get_cblas_info,
                       check_equal_sizes, convert_uplo, convert_trans, convert_diag, ROW_MAJOR)
from ctypes import c_int, POINTER


def trsv(A, b, uplo='u', trans_a='n', diag='n', lda=None, inc_b=1):
    """
    Perform a triangular solve operation.

    A * x = b

    which is solved by overwriting b with the contents of the solution vector x as follows:

    b := A_inv * b

    where A is a triangular matrix and x and b are general vectors.

    The 'uplo' argument indicates whether the lower or upper triangle of A is to be referenced and
    updated by the operation. The 'trans_a' argument allows the computation to proceed as if A is
    transposed. The 'diag' argument indicates whether the diagonal of A is unit or non-unit.

    Vector b can be passed in as either row or column vector. If necessary, an implicit
    transposition occurs.

    WARNING: This function does not test for singularity or near-singularity. Such tests should
             be performed prior to calling this function.

    Args:
        A:          2D NumPy matrix or ndarray representing matrix A
        b:          2D NumPy matrix or ndarray representing vector b

        --optional arguments--

        uplo:       'u'  if the upper triangle of A is to be used
                    'l'  if the lower triangle A is to be used
                        < default is 'u' >
        trans_a:    'n'  if the operation is to proceed normally
                    't'  if the operation is to proceed as if A is transposed
                        < default is 'n' >
        diag:       'n'  if the diagonal of A is non-unit
                    'u'  if the diagonal of A is unit
                        < default is 'n' >
        lda:        leading dimension of A (must be >= # of cols in A)
                        < default is the number of columns in A >
        inc_b:      stride of b (increment for the elements of b)
                        < default is 1 >

    Raises:
        ValueError: if any of the following conditions occur:
                    - A or b is not a 2D NumPy ndarray or NumPy matrix
                    - A and b do not have the same dtype or that dtype is not supported
                    - A is not a square matrix
                    - b is not a vector
                    - the effective length of b does not equal the dimension of A
                    - uplo is not equal to one of the following: 'u', 'U', 'l', 'L'
                    - trans_a is not equal to one of the following: 'n', 'N', 't', 'T'
                    - diag is not equal to one fo the following: 'n', 'N', 'u', 'U'

    Returns:
        Vector x (which is also written to vector b)
    """

    # convert to appropriate CBLAS values
    cblas_uplo = convert_uplo(uplo)
    cblas_trans_a = convert_trans(trans_a)
    cblas_diag = convert_diag(diag)

    # get the dimensions of the parameters
    m_b, n_b, b_length = get_vector_dimensions('b', b, inc_b)
    dim_A = get_square_matrix_dimension('A', A)

    # assign a default value to lda if necessary (assumes row-major order)
    if lda is None:
        lda = dim_A

    # ensure the parameters are appropriate for the operation
    check_equal_sizes('A', dim_A, 'b', b_length)

    # determine which CBLAS subroutine to call and which ctypes data type to use
    cblas_func, data_type = get_cblas_info('trsv', (A.dtype, b.dtype))

    # create a ctypes POINTER for each vector and matrix
    ctype_x = POINTER(data_type * n_b * m_b)
    ctype_A = POINTER(data_type * dim_A * dim_A)

    # call CBLAS using ctypes
    cblas_func.argtypes = [c_int, c_int, c_int, c_int, c_int, ctype_A, c_int, ctype_x, c_int]
    cblas_func.restype = None
    cblas_func(ROW_MAJOR, cblas_uplo, cblas_trans_a, cblas_diag, dim_A,
               A.ctypes.data_as(ctype_A), lda, b.ctypes.data_as(ctype_x), inc_b)

    return b  # contains the value of x (also written to b)