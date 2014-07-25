"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
from ctypes import byref, c_int, c_double, POINTER

def dscal(n, alpha, x, inc_x, vec_orient):
    """Wrapper for BLAS dscal.
    Perform a scaling operation on a vector.

    x := alpha * x

    where alpha is a scalar and x is a row or column vector.

    Args:
        n:              number of elements in the vector x
        alpha:          double representing scalar alpha
        x:              array of doubles representing vector x
        inc_x           stride of x (increment for the elements of x)
        vec_orient:     Vec.ROW if x is a row vector, Vec.COL if x is a column vector
    """

    _libblas.cblas_dscal.argtypes = [c_int, c_double,
                                     POINTER((c_double * n * 1) if vec_orient & 1
                                     else (c_double * 1 * n)), c_int]
    _libblas.cblas_dscal.restype = None

    _libblas.cblas_dscal(n, alpha, byref(x), inc_x)