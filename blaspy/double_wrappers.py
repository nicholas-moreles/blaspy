"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ctypes import *

# has to be located in /usr/lib
_libblas = cdll.LoadLibrary("libcblas.so")

"""

    Level 1 BLAS

"""


def dasum(n, x, x_is_col, inc_x):
    """ Wrapper for BLAS dasum.
    Compute the 1-norm of a vector (i.e. the sum of the magnitudes of the vector elements).

    SUM(|chi_i|) from i=0 to i=n-1

    where chi_i is the ith elements of vector x of length n.

    Args:
        n:          the number of elements in the vector x
        x:          an array representing vector x
        x_is_col:   True if x is a column vector, False if x is a row vector
        inc_x:      stride of x (increment for the elements of x)

    Returns:
        A float representing the 1-norm of vector x.
    """

    _libblas.cblas_dasum.argtypes = [c_int, POINTER((c_double * 1 * n) if x_is_col
                                            else (c_double * n * 1)), c_int]
    _libblas.cblas_dasum.restype = c_double

    return _libblas.cblas_dasum(n, byref(x), inc_x)


def daxpy(n, alpha, x, x_is_col, inc_x, y, y_is_col, inc_y):
    """ Wrapper for BLAS daxpy.
    Perform an axpy operation between two vectors.

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

    _libblas.cblas_daxpy(n, alpha, byref(x), inc_x, byref(y), inc_y)


def dcopy(n, x, x_is_col, inc_x, y, y_is_col, inc_y):
    """ Wrapper for BLAS dcopy.
    Copy the numerical contents of one vector to another.

    y := x

    where x and y are both vectors of length n.

    Args:
        n:          the number of elements in the vectors x and y
        x:          an array representing vector x
        x_is_col:   True if x is a column vector, False if x is a row vector
        inc_x:      stride of x (increment for the elements of x)
        y:          an array representing vector y
        y_is_col:   True if y is a column vector, False if y is a row vector
        inc_y:      stride of y (increment for the elements of y)
    """

    _libblas.cblas_dcopy.argtypes = [c_int, POINTER((c_double * 1 * n) if x_is_col
                                            else (c_double * n * 1)), c_int,
                                            POINTER((c_double * 1 * n) if y_is_col
                                            else (c_double * n * 1)), c_int]
    _libblas.cblas_dcopy.restype = None

    _libblas.cblas_dcopy(n, byref(x), inc_x, byref(y), inc_y)


def ddot(n, x, x_is_col, inc_x, y, y_is_col, inc_y):
    """Wrapper for BLAS ddot.
    Perform a dot (inner) product operation between two vectors.

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
    """Wrapper for BLAS dscal.
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


# def dsdot(n, x, x_is_col, inc_x, y, y_is_col, inc_y):
#     """Wrapper for BLAS dsdot. (DOES NOT FUNCTION PROPERLY)
#     Perform a dot (inner) product operation between two vectors with extended precision.
#
#     rho := SUM(chi_i * psi_i) from i=0 to i=n-1
#
#     where rho is a scalar, and chi_i and psi_i are the ith elements of vectors x and y,
#     respectively, where both vectors are of length n. Upon completion, the dot product rho is
#     returned.
#
#     Args:
#         n:          the number of elements in the vectors x and y
#         x:          an array representing vector x
#         x_is_col:   True if x is a column vector, False if x is a row vector
#         inc_x:      stride of x (increment for the elements of x)
#         y:          an array representing vector y
#         y_is_col:   True if y is a column vector, False if y is a row vector
#         inc_y:      stride of y (increment for the elements of y)
#
#     Returns:
#         A float representing rho, the result of the dot product between x and y.
#     """
#
#     _libblas.cblas_dsdot.argtypes = [c_int, POINTER((c_double * 1 * n) if x_is_col
#                                             else (c_double * n * 1)), c_int,
#                                             POINTER((c_double * 1 * n) if y_is_col
#                                             else (c_double * n * 1)), c_int]
#     _libblas.cblas_dsdot.restype = c_double
#
#     return _libblas.cblas_dsdot(n, byref(x), inc_x, byref(y), inc_y)


def idamax(n, x, x_is_col, inc_x):
    """Wrapper for BLAS idamax.
    Find and return the index of the element which has the maximum absolute value in the vector x.
    If the maximum absolute value is shared by more than one element, then element whose index is
    highest is chosen.

    Args:
        n:          the number of elements in the vector x
        x:          an array representing vector x
        x_is_col:   True if x is a column vector, False if x is a row vector
        inc_x:      stride of x (increment for the elements of x)

    Returns:
        An int representing the index of the element which has the maximum absolute value in the
        vector x.
    """

    _libblas.cblas_idamax.argtypes = [c_int, POINTER((c_double * 1 * n) if x_is_col
                                            else (c_double * n * 1)), c_int]
    _libblas.cblas_idamax.restype = c_int

    return _libblas.cblas_idamax(n, byref(x), inc_x)