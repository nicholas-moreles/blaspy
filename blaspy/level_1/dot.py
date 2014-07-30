"""

    Copyright (c) 2014, The University of Texas at Austin..
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

# noinspection PyProtectedMember
from ..config import _libblas
from ..helpers import find_length
from ctypes import c_int, c_double, c_float, POINTER


def dot(x, y, inc_x=1, inc_y=1):
    """Perform a dot (inner) product operation between two vectors.

    rho := SUM(chi_i * psi_i) from i=0 to i=n-1

    where rho is a scalar, and chi_i and psi_i are the ith elements of vectors x and y,
    respectively, where x and y are general vectors of length n.

    Vectors x and y can be row or column vectors. If necessary, an implicit transposition occurs.

    Args:
        x:      a 2D numpy matrix or ndarray representing vector x
        y:      a 2D numpy matrix or ndarray representing vector y
        inc_x:  stride of x (increment for the elements of x)
        inc_y:  stride of y (increment for the elements of y)

    Returns:
        rho, the result of the dot product between x and y.
    """

    try:
        # get the dimensions of the parameters
        m_x, n_x = x.shape
        m_y, n_y = y.shape
        x_length = find_length(m_x, n_x, inc_x)
        y_length = find_length(m_y, n_y, inc_y)

        # ensure the parameters are appropriate for the operation
        if not (m_x == 1 or n_x == 1):
            raise ValueError("x must be a vector")
        if not (m_y == 1 or n_y == 1):
            raise ValueError("y must be a vector")
        if x_length != y_length:
            raise ValueError("size mismatch between x and y")

        # determine which BLAS routine to call based on data type
        if x.dtype == 'float64' and y.dtype == 'float64':  # ddot
            blas_func = _libblas.cblas_ddot
            data_type = c_double
        elif x.dtype == 'float32' and y.dtype == 'float32':  # sdot
            blas_func = _libblas.cblas_sdot
            data_type = c_float
        else:
            raise ValueError("x and y must have the same dtype, either float64 or float32")

        # call BLAS using ctypes
        ctype_x = POINTER(data_type * n_x * m_x)
        ctype_y = POINTER(data_type * n_y * m_y)
        blas_func.argtypes = [c_int, ctype_x, c_int, ctype_y, c_int]
        blas_func.restype = data_type
        return blas_func(x_length, x.ctypes.data_as(ctype_x), inc_x, y.ctypes.data_as(ctype_y),
                         inc_y)

    except AttributeError:
        raise ValueError("x and y must be of type numpy.ndarray or numpy.matrix")