"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
from ctypes import byref, c_int, c_double, POINTER

def dcopy(n, x, inc_x, y, inc_y, vec_orient):
    """ Wrapper for BLAS dcopy.
    Copy the numerical contents of one vector to another.

    y := x

    where x and y are both vectors of length n.

    Args:
        n:              number of elements in the vectors x and y
        x:              array of doubles representing vector x
        inc_x:          stride of x (increment for the elements of x)
        y:              array of doubles representing vector y
        inc_y:          stride of y (increment for the elements of y)
        vec_orient:     Vec.ROW_ROW  if x is a row vector and y is a row vector
                        Vec.ROW_COL  if x is a row vector and y is a column vector
                        Vec.COL_COL  if x is a column vector and y is a column vector
                        Vec.COL_ROW  if x is a column vector and y is a row vector
    """

    _libblas.cblas_ddot.argtypes = [c_int, POINTER((c_double * n * 1) if vec_orient & 1
                                    else (c_double * 1 * n)), c_int,
                                    POINTER((c_double * n * 1) if vec_orient >> 1 & 1
                                    else (c_double * 1 * n)), c_int]
    _libblas.cblas_dcopy.restype = None

    _libblas.cblas_dcopy(n, byref(x), inc_x, byref(y), inc_y)