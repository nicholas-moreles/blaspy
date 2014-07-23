"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
import ctypes as c

def ddot(n, x, inc_x, y, inc_y, orientation):
    """Wrapper for BLAS ddot.
    Perform a dot (inner) product operation between two vectors.

    rho := SUM(chi_i * psi_i) from i=0 to i=n-1

    where rho is a scalar, and chi_i and psi_i are the ith elements of vectors x and y,
    respectively, where both vectors are of length n. Upon completion, the dot product rho is
    returned.

    Args:
        n:              the number of elements in the vectors x and y
        x:              an array of doubles representing vector x
        inc_x:          stride of x (increment for the elements of x)
        y:              an array of doubles representing vector y
        inc_y:          stride of y (increment for the elements of y)
        orientation:    blaspy.ROW_ROW  if x is a row vector and y is a row vector
                        blaspy.ROW_COL  if x is a row vector and y is a column vector
                        blaspy.COL_COL  if x is a column vector and y is a column vector
                        blaspy.COL_ROW  if x is a column vector and y is a row vector

    Returns:
        A float representing rho, the result of the dot product between x and y.
    """

    _libblas.cblas_ddot.argtypes = [c.c_int, c.POINTER((c.c_double * n * 1) if orientation & 1
                                    else (c.c_double * 1 * n)), c.c_int,
                                    c.POINTER((c.c_double * n * 1) if orientation >> 1 & 1
                                    else (c.c_double * 1 * n)), c.c_int]
    _libblas.cblas_ddot.restype = c.c_double

    return _libblas.cblas_ddot(n, c.byref(x), inc_x, c.byref(y), inc_y)