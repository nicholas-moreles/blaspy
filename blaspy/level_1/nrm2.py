"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import get_vector_dimensions, get_cblas_info
from ..errors import raise_not_2d_numpy
from ctypes import c_int, POINTER


def nrm2(x, inc_x=1):
    """Compute the 2-norm (Euclidean norm) of a vector.

    ||x||_2 = [SUM(|chi_i|^2)]^(1/2) from i=0 to i=n-1

    where chi_i is the ith element of general vector x of length n and ||x||_2 is returned.

    Args:
        x:              a 2D numpy matrix or ndarray representing vector x
        inc_x:          stride of x (increment for the elements of x)

    Returns:
        The 2-norm of vector x.
    """

    try:
        # get the dimensions of the parameters
        m_x, n_x, x_length = get_vector_dimensions('x', x, inc_x)

        # determine which CBLAS subroutine to call and which ctypes data type to use
        cblas_func, ctype_dtype = get_cblas_info('nrm2', (x.dtype,))

        # call BLAS using ctypes
        ctype_x = POINTER(ctype_dtype * n_x * m_x)
        cblas_func.argtypes = [c_int, ctype_x, c_int]
        cblas_func.restype = ctype_dtype
        return cblas_func(x_length, x.ctypes.data_as(ctype_x), inc_x)

    except AttributeError:
        raise_not_2d_numpy()