"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

# noinspection PyProtectedMember
from ..config import _libblas
from ..helpers import find_length
from ctypes import c_int, c_double, c_float, POINTER


# noinspection PyUnresolvedReferences
def amax(x, inc_x=1):
    """Find and return the index of the element which has the maximum absolute value in the vector x.
    If the maximum absolute value is shared by more than one element, then the element whose index
    is lowest is chosen.

    Args:
        x:              a 2D numpy matrix or ndarray representing vector x
        inc_x:          stride of x (increment for the elements of x)

    Returns:
        The index of the element which has the maximum absolute value in the vector x.
    """

    try:
        # get the dimensions of the parameters
        m_x, n_x = x.shape
        x_length = find_length(m_x, n_x, inc_x)

        # ensure the parameters are appropriate for the operation
        if not (m_x == 1 or n_x == 1):
            raise ValueError("x must be a vector")

        # determine which BLAS routine to call based on data type
        if x.dtype == 'float64':
            blas_func = _libblas.cblas_idamax
            data_type = c_double
        elif x.dtype == 'float32':
            blas_func = _libblas.cblas_isamax
            data_type = c_float
        else:
            raise ValueError("x must have dtype of either float64 or float32")

        # call BLAS using ctypes
        ctype_x = POINTER(data_type * n_x * m_x)
        blas_func.argtypes = [c_int, ctype_x, c_int]
        blas_func.restype = c_int
        return blas_func(x_length, x.ctypes.data_as(ctype_x), inc_x)

    except AttributeError:
        raise ValueError("x and y must be of type numpy.ndarray or numpy.matrix")