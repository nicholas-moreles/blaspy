"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ctypes import *

# has to be placed in /usr/lib
_libblas = cdll.LoadLibrary("libopenblasp-r0.2.9-64ref32threads.so")

"""

    Level 1 BLAS

"""

def daxpy(n, alpha, x, x_is_col, inc_x, y, y_is_col, inc_y):
    """ Wrapper for BLAS daxpy.
    Performs an axpy operation between two vectors:

    y := alpha * x + y

    where alpha is a scalar, and x and y are either both column vectors or both row vectors.

    Args:
        n:          the number of elements in the vectors x and y
        alpha:      an int representing scalar alpha
        x:          an array representing vector x
        x_is_col:   True if x is a column vector, False if x is a row vector
        inc_x:      stride of x (increment for the elements of x)
        y:          an array representing vector y
        y_is_col:   True if y is a column vector, False if y is a row vector
        inc_y:      stride of y (increment for the elements of y)
    """

    _libblas.cblas_daxpy.argtypes = [c_int, c_double, POINTER((c_double * 1 * n) if x_is_col
                                            else (c_double * n * 1)), c_int,
                                            POINTER((c_double * 1 * n) if y_is_col
                                            else (c_double * n * 1)), c_int]
    _libblas.cblas_daxpy.restype = None

    _libblas.cblas_daxpy(n, alpha, x, inc_x, y, inc_y)


def ddot(n, x, x_is_col, inc_x, y, y_is_col, inc_y):
    """Wrapper for BLAS ddot.
    Performs a dot (inner) product operation between two vectors.

    rho := SUM(chi_i * psi_i) from i=0 to i=n-1

    where rho is a scalar, and chi_i and psi_i are the ith elements of vectors x and y,
    respectively, where both vectors are of length n. Upon completion, the dot product rho is
    returned.

    Args:
        n:          the number of elements in the vectors x and y
        x:          an array representing vector x
        x_is_col:   True if x is a column vector, False if x is a row vector
        inc_x:      stride of x (increment for the elements of x)
        y:          an array representing vector y
        y_is_col:   True if y is a column vector, False if y is a row vector
        inc_y:      stride of y (increment for the elements of y)

    Returns:
        A float representing rho, the result of the dot product between x and y.
    """

    _libblas.cblas_ddot.argtypes = [c_int, POINTER((c_double * 1 * n) if x_is_col
                                            else (c_double * n * 1)), c_int,
                                            POINTER((c_double * 1 * n) if y_is_col
                                            else (c_double * n * 1)), c_int]
    _libblas.cblas_ddot.restype = c_double

    return _libblas.cblas_ddot(n, byref(x), inc_x, byref(y), inc_y)


def dscal(n, alpha, x, x_is_col, inc_x):
    """Wrapper fn, or BLAS dscal.
    Perform a scaling operation on a vector.

    x := alpha * x

    where alpha is a scalar and x is a row or column vector.

    Args:
        n:          the number of elements in the vector x
        alpha:      a float representing scalar alpha
        x:          an array representing vector x
        x_is_col:   True if x is a column vector, False if x is a row vector
        inc_x       stride of x (increment for the elements of x)
    """

    _libblas.cblas_dscal.argtypes = [c_int, c_double, POINTER((c_double * 1 * n) if x_is_col else
                                            (c_double * n * 1)), c_int]
    _libblas.cblas_dscal.restype = None

    _libblas.cblas_dscal(n, alpha, byref(x), inc_x)