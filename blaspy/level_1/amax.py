"""

    Copyright (c) 2014-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import get_vector_dimensions, get_cblas_info
from ctypes import c_int, POINTER


def amax(x, inc_x=1):
    """
    Find and return the index of the element which has the maximum absolute value in the vector x.
    If the maximum absolute value is shared by more than one element, then the element whose index
    is lowest is chosen.

    Args:
        x:              2D NumPy matrix or ndarray representing vector x

        --optional arguments--

        inc_x:          stride of x (increment for the elements of x)
                            < default is 1 >

    Returns:
        The index of the element which has the maximum absolute value in the vector x.

    Raises:
        ValueError: if any of the following conditions occur:
                        - x is not a 2D NumPy matrix or ndarray
                        - x has a dtype that is not supported
                        - x is not a vector
    """

    # get the dimensions of the parameters
    m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)

    # determine which CBLAS subroutine to call and which ctypes data type to use
    cblas_func, ctype_dtype = get_cblas_info('amax', (x.dtype,))

    # create a ctypes POINTER for vector x
    ctype_x = POINTER(ctype_dtype * n_x * m_x)

    # call CBLAS using ctypes
    cblas_func.argtypes = [c_int, ctype_x, c_int]
    cblas_func.restype = c_int
    return cblas_func(x_length, x.ctypes.data_as(ctype_x), inc_x)