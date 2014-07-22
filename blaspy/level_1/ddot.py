"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy.config import _libblas
import ctypes as c

def ddot(n, x, x_is_col, inc_x, y, y_is_col, inc_y):
    """Wrapper for BLAS ddot.
    Perform a dot (inner) product operation between two vectors.

    rho := SUM(chi_i * psi_i) from i=0 to i=n-1

    where rho is a scalar, and chi_i and psi_i are the ith elements of vectors x and y,
    respectively, where both vectors are of length n. Upon completion, the dot product rho is
    returned.

    Args:
        n:          the number of elements in the vectors x and y
        x:          an array of doubles representing vector x
        x_is_col:   True if x is a column vector, False if x is a row vector
        inc_x:      stride of x (increment for the elements of x)
        y:          an array of doubles representing vector y
        y_is_col:   True if y is a column vector, False if y is a row vector
        inc_y:      stride of y (increment for the elements of y)

    Returns:
        A float representing rho, the result of the dot product between x and y.
    """

    _libblas.cblas_ddot.argtypes = [c.c_int, c.POINTER((c.c_double * 1 * n) if x_is_col
                                            else (c.c_double * n * 1)), c.c_int,
                                            c.POINTER((c.c_double * 1 * n) if y_is_col
                                            else (c.c_double * n * 1)), c.c_int]
    _libblas.cblas_ddot.restype = c.c_double

    return _libblas.cblas_ddot(n, c.byref(x), inc_x, c.byref(y), inc_y)