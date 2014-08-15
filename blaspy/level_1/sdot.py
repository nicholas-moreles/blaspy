"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..config import _libblas
from ..helpers import get_vector_dimensions, check_equal_sizes
from ..errors import raise_generic_type_error, raise_invalid_dtypes, raise_invalid_parameter
from ctypes import c_int, c_double, c_float, POINTER


def sdot(x, y, inc_x=1, inc_y=1, output='float64'):
    """
    Perform a dot (inner) product operation with extended precision between two vectors.

    rho := SUM(chi_i * psi_i) from i=0 to i=n-1

    where rho is a scalar, and chi_i and psi_i are the ith elements of vectors x and y,
    respectively, where x and y are general vectors of length n.

    Vectors x and y can be passed in as either row or column vectors. If necessary, an implicit
    transposition occurs.

    Args:
        x:         2D numpy matrix or ndarray representing vector x
        y:         2D numpy matrix or ndarray representing vector y
        inc_x:     stride of x (increment for the elements of x)
        inc_y:     stride of y (increment for the elements of y)
        output:    precision of the returned value rho, either 'float64' or 'float32'

    Returns:
        rho, the result of the dot product between x and y.
    """

    try:
        # get the dimensions of the parameters
        m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)
        m_y, n_y, y_length = get_vector_dimensions('y', y, inc_y)

        # ensure the parameters are appropriate for the operation
        check_equal_sizes('x', x_length, 'y', y_length)

        # get_cblas_info cannot be used for sdot due to the way input/output data types don't
        # necessarily match. For now, this is handled in this function, but it may be moved into
        # helpers.py if it occurs again when implementing other CBLAS subroutines.

        # check type of x and y
        if not (x.dtype == 'float32' and y.dtype == 'float32'):
            raise_invalid_dtypes('float32')

        # determine which CBLAS routine to call based on data type, then call CBLAS using ctypes
        if output == 'float64':
            cblas_func = _libblas.cblas_dsdot
            ctype_x = POINTER(c_float * n_x * m_x)
            ctype_y = POINTER(c_float * n_y * m_y)
            cblas_func.argtypes = [c_int, ctype_x, c_int, ctype_y, c_int]
            cblas_func.restype = c_double
            return cblas_func(x_length, x.ctypes.data_as(ctype_x), inc_x,
                              y.ctypes.data_as(ctype_y), inc_y)
        elif output == 'float32':
            cblas_func = _libblas.cblas_sdsdot
            ctype_x = POINTER(c_float * n_x * m_x)
            ctype_y = POINTER(c_float * n_y * m_y)
            cblas_func.argtypes = [c_int, c_float, ctype_x, c_int, ctype_y, c_int]
            cblas_func.restype = c_float
            return cblas_func(x_length, 0, x.ctypes.data_as(ctype_x), inc_x,
                              y.ctypes.data_as(ctype_y), inc_y)
        else:
            raise_invalid_parameter('output', ('float32', 'float64'), output)

    except (AttributeError, TypeError):
        raise_generic_type_error()