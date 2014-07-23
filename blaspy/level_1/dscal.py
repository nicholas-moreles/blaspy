"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
import ctypes as c

def dscal(n, alpha, x, inc_x, orientation):
    """Wrapper for BLAS dscal.
    Perform a scaling operation on a vector.

    x := alpha * x

    where alpha is a scalar and x is a row or column vector.

    Args:
        n:              the number of elements in the vector x
        alpha:          a double representing scalar alpha
        x:              an array of doubles representing vector x
        inc_x           stride of x (increment for the elements of x)
        orientation:    blaspy.ROW if x is a row vector, blaspy.COL if x is a column vector
    """

    _libblas.cblas_dscal.argtypes = [c.c_int, c.c_double,
                                     c.POINTER((c.c_double * n * 1) if orientation & 1
                                     else (c.c_double * 1 * n)), c.c_int]
    _libblas.cblas_dscal.restype = None

    _libblas.cblas_dscal(n, alpha, c.byref(x), inc_x)