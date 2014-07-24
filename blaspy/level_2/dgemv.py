"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
from ctypes import byref, c_int, c_double, POINTER

def dgemv(order, trans, m, n, alpha, A, lda, x, inc_x, beta, y, inc_y, vec_orient):
    """Wrapper for BLAS dgemv.
    Perform a general matrix-vector multiplication operation.

    y := beta * y + alpha * A * x

    where alpha and beta are scalars, A is a general matrix, and x and y are treated as general
    column vectors. The trans argument allows the computation to proceed as if A were transposed.

    Args:
        order:          Order.ROW_MAJOR  if A is stored in row-major order
                        Order.COL_MAJOR  if A is stored in column-major order
        trans:          Trans.NO_TRANS   if the multiply is to proceed as normal
                        Trans.TRANS      if the multiply is to proceed as if A is transposed
        m:              number of rows in matrix A
        n:              number of columns in matrix A
        alpha:          double representing scalar alpha
        A:              a 2-dimensional array of doubles representing general matrix A
        lda:            leading dimension of A (must be >= m if A is stored in column-major order
                        or must be >= n if A is stored in row-major order)
        x:              array of doubles representing vector x
        inc_x:          stride of x (increment for the elements of x)
        beta:           double representing scalar beta
        y:              array of doubles representing vector y
        inc_y:          stride of y (increment for the elements of y)
        vec_orient:     Vec.ROW_ROW  if x is a row vector and y is a row vector
                        Vec.ROW_COL  if x is a row vector and y is a column vector
                        Vec.COL_COL  if x is a column vector and y is a column vector
                        Vec.COL_ROW  if x is a column vector and y is a row vector
    """

    _libblas.cblas_dgemv.argtypes = [c_int, c_int, c_int, c_int, c_double,
                                     POINTER(c_double * n * m), c_int,
                                     POINTER((c_double * (n if trans & 1 else m) * 1)
                                     if vec_orient & 1
                                     else (c_double * 1 * (n if trans & 1 else m))), c_int,
                                     c_double, POINTER((c_double * (m if trans & 1 else n) * 1)
                                     if vec_orient >> 1 & 1
                                     else (c_double * 1 * (m if trans & 1 else n))), c_int]
    _libblas.cblas_dgemv.restype = None

    return _libblas.cblas_dgemv(order, trans, m, n, alpha, byref(A), lda, byref(x), inc_x, beta,
                                byref(y), inc_y)