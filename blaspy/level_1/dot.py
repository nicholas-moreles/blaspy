"""

    Copyright (c) 2014-2015-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import get_vector_dimensions, get_cblas_info, check_equal_sizes
from ctypes import c_int, POINTER


def dot(x, y, inc_x=1, inc_y=1):
    """Perform a dot (inner) product operation between two vectors.

    rho := SUM(chi_i * psi_i) from i=0 to i=n-1

    where rho is a scalar, and chi_i and psi_i are the ith elements of vectors x and y,
    respectively, where x and y are general vectors of length n.

    Vectors x and y can be passed in as either row or column vectors. If necessary, an implicit
    transposition occurs.

    Args:
        x:      2D NumPy matrix or ndarray representing vector x
        y:      2D NumPy matrix or ndarray representing vector y

        --optional arguments--

        inc_x:  stride of x (increment for the elements of x)
                    < default is 1 >
        inc_y:  stride of y (increment for the elements of y)
                    < default is 1 >

    Returns:
        rho, the result of the dot product between x and y.

    Raises:
        ValueError: if any of the following conditions occur:
                        - x or y is not a 2D NumPy matrix or ndarray
                        - x and y do not have the same dtype or that dtype is not supported
                        - x or y is not a vector
                        - x and y do not have the same length
    """

    # get the dimensions of the parameters
    m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)
    m_y, n_y, y_length = get_vector_dimensions('y', y, inc_y)

    # ensure the parameters are appropriate for the operation
    check_equal_sizes('x', x_length, 'y', y_length)

    # determine which CBLAS subroutine to call and which ctypes data type to use
    cblas_func, ctype_dtype = get_cblas_info('dot', (x.dtype, y.dtype))

    # create a ctypes POINTER for each vector
    ctype_x = POINTER(ctype_dtype * n_x * m_x)
    ctype_y = POINTER(ctype_dtype * n_y * m_y)

    # call CBLAS using ctypes
    cblas_func.argtypes = [c_int, ctype_x, c_int, ctype_y, c_int]
    cblas_func.restype = ctype_dtype
    return cblas_func(x_length, x.ctypes.data_as(ctype_x), inc_x, y.ctypes.data_as(ctype_y), inc_y)