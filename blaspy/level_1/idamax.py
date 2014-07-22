"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy.config import _libblas
import ctypes as c

def idamax(n, x, x_is_col, inc_x):
    """Wrapper for BLAS idamax.
    Find and return the index of the element which has the maximum absolute value in the vector x.
    If the maximum absolute value is shared by more than one element, then the element whose index
    is lowest is chosen.

    Args:
        n:          the number of elements in the vector x
        x:          an array of doubles representing vector x
        x_is_col:   True if x is a column vector, False if x is a row vector
        inc_x:      stride of x (increment for the elements of x)

    Returns:
        An int representing the index of the element which has the maximum absolute value in the
        vector x.
    """

    _libblas.cblas_idamax.argtypes = [c.c_int, c.POINTER((c.c_double * 1 * n) if x_is_col
                                            else (c.c_double * n * 1)), c.c_int]
    _libblas.cblas_idamax.restype = c.c_int

    return _libblas.cblas_idamax(n, c.byref(x), inc_x)