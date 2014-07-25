"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
from ctypes import byref, c_int, c_double, POINTER

def dger(order, m, n, alpha, x, inc_x, y, inc_y, A, lda, vec_orient):
    """Wrapper for BLAS dger.
    Perform a general rank-1 update operation.

    A := A + alpha * x * y

    where alpha is a scalar, A is a general matrix, x is treated as a general column vector,
    and y is treated as a general row vector.

    Args:
        order:          Order.ROW_MAJOR  if A is stored in row-major order
                        Order.COL_MAJOR  if A is stored in column-major order
        m:              number of rows in matrix A
        n:              number of columns in matrix A
        alpha:          double representing scalar alpha
        x:              array of doubles representing vector x
        inc_x:          stride of x (increment for the elements of x)
        y:              array of doubles representing vector y
        inc_y:          stride of y (increment for the elements of y)
        A:              a 2-dimensional array of doubles representing general matrix A
        lda:            leading dimension of A (must be >= m if A is stored in column-major order
                        or must be >= n if A is stored in row-major order)
        vec_orient:     Vec.ROW_ROW  if x is a row vector and y is a row vector
                        Vec.ROW_COL  if x is a row vector and y is a column vector
                        Vec.COL_COL  if x is a column vector and y is a column vector
                        Vec.COL_ROW  if x is a column vector and y is a row vector
    """

    _libblas.cblas_dger.argtypes = [c_int, c_int, c_int, c_double,
                                    POINTER((c_double * m * 1) if vec_orient & 1
                                    else (c_double * 1 * m)), c_int,
                                    POINTER((c_double * n * 1) if vec_orient >> 1 & 1
                                    else (c_double * 1 * n)), c_int,
                                    POINTER(c_double * n * m), c_int]
    _libblas.cblas_dger.restype = None

    return _libblas.cblas_dger(order, m, n, alpha, byref(x), inc_x, byref(y), inc_y, byref(A), lda)