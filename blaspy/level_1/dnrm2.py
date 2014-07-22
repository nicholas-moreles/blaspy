"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy.config import _libblas
import ctypes as c

def dnrm2(n, x, x_is_col, inc_x):
    """Wrapper for BLAS dnrm2.
    Compute the 2-norm (Euclidean norm) of a vector.

    ||x||_2 = [SUM(|chi_i|^2)]^(1/2) from i=0 to i=n-1

    where chi_i is the ith elements of vector x of length n and ||x||_2 is returned.

    Args:
        n:          the number of elements in the vector x
        x:          an array of doubles representing vector x
        x_is_col:   True if x is a column vector, False if x is a row vector
        inc_x:      stride of x (increment for the elements of x)

    Returns:
        A float representing the 2-norm of vector x.
    """

    _libblas.cblas_dnrm2.argtypes = [c.c_int, c.POINTER((c.c_double * 1 * n) if x_is_col
                                            else (c.c_double * n * 1)), c.c_int]
    _libblas.cblas_dnrm2.restype = c.c_double

    return _libblas.cblas_dnrm2(n, c.byref(x), inc_x)