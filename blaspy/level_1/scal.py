"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import get_vector_dimensions, get_cblas_info
from ..errors import raise_generic_type_error
from ctypes import c_int, POINTER


def scal(alpha, x, inc_x=1):
    """
    Perform a scaling operation on a vector.

    x := alpha * x

    where alpha is a scalar and x is a general vector.

    Args:
        alpha:          scalar alpha
        x:              2D NumPy matrix or ndarray representing vector x

        --optional arguments--

        inc_x           stride of x (increment for the elements of x)
                            < default is 1 >

    Returns:
        Vector x

    Raises:
        TypeError:  if x is not a 2D NumPy matrix or ndarray
        ValueError: if any of the following conditions occur:
                        - if x has a dtype that is not supported
                        - x is not a vector
    """

    try:
        # get the dimensions of the parameters
        m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)

        # determine which CBLAS subroutine to call and which ctypes data type to use
        cblas_func, ctype_dtype = get_cblas_info('scal', (x.dtype,))

        # create a ctypes POINTER for vector x
        ctype_x = POINTER(ctype_dtype * n_x * m_x)

        # call CBLAS using ctypes
        cblas_func.argtypes = [c_int, ctype_dtype, ctype_x, c_int]
        cblas_func.restype = None
        cblas_func(x_length, alpha, x.ctypes.data_as(ctype_x), inc_x)

        return x

    except (AttributeError, TypeError):
        raise_generic_type_error()