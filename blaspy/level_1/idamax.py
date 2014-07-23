"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
import ctypes as c

def idamax(n, x, inc_x, orientation):
    """Wrapper for BLAS idamax.
    Find and return the index of the element which has the maximum absolute value in the vector x.
    If the maximum absolute value is shared by more than one element, then the element whose index
    is lowest is chosen.

    Args:
        n:              the number of elements in the vector x
        x:              an array of doubles representing vector x
        inc_x:          stride of x (increment for the elements of x)
        orientation:    blaspy.ROW if x is a row vector, blaspy.COL if x is a column vector

    Returns:
        An int representing the index of the element which has the maximum absolute value in the
        vector x.
    """

    _libblas.cblas_idamax.argtypes = [c.c_int, c.POINTER((c.c_double * n * 1) if orientation & 1
                                      else (c.c_double * 1 * n)), c.c_int]
    _libblas.cblas_idamax.restype = c.c_int

    return _libblas.cblas_idamax(n, c.byref(x), inc_x)