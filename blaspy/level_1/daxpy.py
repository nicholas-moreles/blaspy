"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy.config import _libblas
import ctypes as c

def daxpy(n, alpha, x, x_is_col, inc_x, y, y_is_col, inc_y):
    """ Wrapper for BLAS daxpy.
    Perform an axpy operation between two vectors.

    y := alpha * x + y

    where alpha is a scalar, and x and y are either both column vectors or both row vectors.

    Args:
        n:          the number of elements in the vectors x and y
        alpha:      a double representing scalar alpha
        x:          an array of doubles representing vector x
        x_is_col:   True if x is a column vector, False if x is a row vector
        inc_x:      stride of x (increment for the elements of x)
        y:          an array of doubles representing vector y
        y_is_col:   True if y is a column vector, False if y is a row vector
        inc_y:      stride of y (increment for the elements of y)
    """

    _libblas.cblas_daxpy.argtypes = [c.c_int, c.c_double, c.POINTER((c.c_double * 1 * n) if x_is_col
                                            else (c.c_double * n * 1)), c.c_int,
                                            c.POINTER((c.c_double * 1 * n) if y_is_col
                                            else (c.c_double * n * 1)), c.c_int]
    _libblas.cblas_daxpy.restype = None

    _libblas.cblas_daxpy(n, alpha, c.byref(x), inc_x, c.byref(y), inc_y)