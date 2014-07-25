"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
from ctypes import byref, c_int, c_double, POINTER

def dgemm(order, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Wrapper for BLAS dgemm.
    Perform a general matrix-matrix multiplication operation.

    C := beta * C + alpha * A * B

    where alpha and beta are scalars and A, B, and C are general matrices. The transA and transB
    arguments allow the computation to proceed as if A and/or B were transposed.

    Args:
        order:          Order.ROW_MAJOR  if A, B, and C are stored in row-major order
                        Order.COL_MAJOR  if A, B, and C are stored in column-major order
        transA:         Trans.NO_TRANS   if A is not to be transposed before the multiply
                        Trans.TRANS      if A is to be transposed before the multiply
        transB:         Trans.NO_TRANS   if B is not to be transposed before the multiply
                        Trans.TRANS      if B is to be transposed before the multiply
        m:              number of rows in matrix A if Trans.NO_TRANS, number of columns in matrix
                        A if TRANS; also, number of rows in matrix C
        n:              number of columns in matrix B if Trans.NO_TRANS, number of rows in matrix
                        B if TRANS; also, number of columns in matrix C
        k:              number of columns in matrix A if Trans.NO_TRANS, number of rows in matrix
                        A if TRANS; also, number of rows in matrix B if Trans.NO_TRANS, number of
                        columns in matrix B if TRANS
        alpha:          double representing scalar alpha
        A:              a 2-dimensional array of doubles representing general matrix A
        lda:            leading dimension of A (must be >= number of rows in A if A is stored in
                        column-major order or must be >= number of columns in A if A is stored in
                        row-major order)
        B:              a 2-dimensional array of doubles representing general matrix AB
        ldb:            leading dimension of B (must be >= number of rows in B if B is stored in
                        column-major order or must be >= number of columns in B if B is stored in
                        row-major order)
        beta:           double representing scalar beta
        C:              a 2-dimensional array of doubles representing general matrix C
        ldc:            leading dimension of C (must be >= number of rows in C if C is stored in
                        column-major order or must be >= number of columns in C if C is stored in
                        row-major order)
    """

    _libblas.cblas_dgemm.argtypes = [c_int, c_int, c_int, c_int, c_int, c_int, c_double,
                                     POINTER((c_double * k * m) if transA & 1
                                     else (c_double * m * k)), c_int,
                                     POINTER((c_double * n * k) if transB & 1
                                     else (c_double * k * n)), c_int, c_double,
                                     POINTER(c_double * n * m), c_int]
    _libblas.cblas_dgemm.restype = None

    return _libblas.cblas_dgemm(order, transA, transB, m, n, k, alpha, byref(A), lda, byref(B),
                                 ldb, beta, C, ldc)