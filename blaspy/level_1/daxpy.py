"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
import ctypes as c

def daxpy(n, alpha, x, inc_x, y, inc_y, orientation):
    """ Wrapper for BLAS daxpy.
    Perform an axpy operation between two vectors.

    y := alpha * x + y

    where alpha is a scalar, and x and y are either both column vectors or both row vectors.

    Args:
        n:              the number of elements in the vectors x and y
        alpha:          a double representing scalar alpha
        x:              an array of doubles representing vector x
        inc_x:          stride of x (increment for the elements of x)
        y:              an array of doubles representing vector y
        inc_y:          stride of y (increment for the elements of y)
        orientation:    blaspy.ROW_ROW  if x is a row vector and y is a row vector
                        blaspy.ROW_COL  if x is a row vector and y is a column vector
                        blaspy.COL_COL  if x is a column vector and y is a column vector
                        blaspy.COL_ROW  if x is a column vector and y is a row vector
    """

    _libblas.cblas_daxpy.argtypes = [c.c_int, c.c_double, c.POINTER((c.c_double * n * 1)
                                     if orientation & 1 else (c.c_double * 1 * n)), c.c_int,
                                     c.POINTER((c.c_double * n * 1) if orientation >> 1 & 1
                                     else (c.c_double * 1 * n)), c.c_int]
    _libblas.cblas_daxpy.restype = None

    _libblas.cblas_daxpy(n, alpha, c.byref(x), inc_x, c.byref(y), inc_y)