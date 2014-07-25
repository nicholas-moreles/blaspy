"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
from ctypes import byref, c_int, c_double, POINTER

def dnrm2(n, x, inc_x, vec_orient):
    """Wrapper for BLAS dnrm2.
    Compute the 2-norm (Euclidean norm) of a vector.

    ||x||_2 = [SUM(|chi_i|^2)]^(1/2) from i=0 to i=n-1

    where chi_i is the ith elements of vector x of length n and ||x||_2 is returned.

    Args:
        n:              number of elements in the vector x
        x:              array of doubles representing vector x
        inc_x:          stride of x (increment for the elements of x)
        vec_orient:     Vec.ROW if x is a row vector, Vec.COL if x is a column vector

    Returns:
        A float representing the 2-norm of vector x.
    """

    _libblas.cblas_dnrm2.argtypes = [c_int, POINTER((c_double * n * 1) if vec_orient & 1
                                     else (c_double * 1 * n)), c_int]
    _libblas.cblas_dnrm2.restype = c_double

    return _libblas.cblas_dnrm2(n, byref(x), inc_x)