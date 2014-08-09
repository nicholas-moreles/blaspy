"""

    Copyright (c) 2014, The University of Texas at Austin..
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from .config import _libblas as lib
from ctypes import c_double, c_float

# CBLAS enum values
ROW_MAJOR = 101
COL_MAJOR = 102
NO_TRANS = 111
TRANS = 112
CONJ_TRANS = 113
CONJ_NO_TRANS = 114
UPPER = 121
LOWER = 122

# dictionary of BLASpy functions mapping to CBLAS subroutines
# - first entry in value pair is for double precision reals
# - second entry in value pair is for single precision reals
# - sdot is omitted as a special case
FUNC_MAP = {'amax': (lib.cblas_idamax, lib.cblas_isamax),
            'asum': (lib.cblas_dasum,  lib.cblas_sasum),
            'axpy': (lib.cblas_daxpy,  lib.cblas_saxpy),
            'copy': (lib.cblas_dcopy,  lib.cblas_scopy),
            'dot':  (lib.cblas_ddot,   lib.cblas_sdot),
            'nrm2': (lib.cblas_dnrm2,  lib.cblas_snrm2),
            'scal': (lib.cblas_dscal,  lib.cblas_sscal),
            'swap': (lib.cblas_dswap,  lib.cblas_sswap),
            'gemv': (lib.cblas_dgemv,  lib.cblas_sgemv),
            'ger':  (lib.cblas_dger,   lib.cblas_sger),
            'symv': (lib.cblas_dsymv,  lib.cblas_ssymv)}

def get_func_and_data_type(calling_func, *args):
    """
    Return the appropriate CBLAS subroutine based on the calling function and dtypes of args.

    Args:
        calling_func:  a string representation of the calling function
        *args:         the dtypes of all of the matrices or vectors used by the calling function

    Returns:
        A tuple of two elements where each element is described in order below:

        - appropriate CBLAS function
        - appropriate ctypes data type

    Raises:
        ValueError  if all matrices and/or vectors do not have the same dtype
    """

    if all(dtype == 'float64' for dtype in args):
        return FUNC_MAP[calling_func][0], c_double

    elif all(dtype == 'float32' for dtype in args):
        return FUNC_MAP[calling_func][1], c_float

    else:
        raise ValueError("All matrices and/or vectors must have the same dtype: either float64 "
                         "or float32. Actual dtypes: %s" % (args,))

def get_vector_dimensions(name, vector, stride):
    """
    Return the number of rows, number of columns, and length of a vector taking into account
    the stride of the vector.

    Args:
        name:    string to print as the vector's name if an error occurs
        vector:  numpy 2D ndarray or matrix representing a vector
        stride:  stride of the vector (increment for the elements of the vector)

    Returns:
        A tuple of three elements where each element is described in order below:

        - number of rows in vector
        - number of columns in vector
        - length of vector after accounting for stride

    Raises:
        ValueError  if vector is not a vector represented by a 2D NumPy ndarray or matrix
    """

    try:
        # will cause AttributeError if vector is not a 2D NumPy ndarray or matrix
        rows, cols = vector.shape

        # ensure vector is actually a vector
        if not (rows == 1 or cols == 1):
            raise ValueError("%s is not a vector. Rows: %i. Columns: %i." % (name, rows, cols))

        # calculate length, accounting for strides > 1
        length = rows if rows > cols else cols
        if stride > 1:
            length = (length / stride) + (length % stride > 0)

        return rows, cols, length

    except AttributeError:
        raise ValueError("%s is not a 2D NumPy ndarray or matrix." % name)

def get_matrix_dimensions(name, matrix):
    """
    Return the number of rows and number of columns in a matrix.

    Args:
        name:    string to print as the matrix's name if an error occurs
        vector:  numpy 2D ndarray or matrix representing the matrix

    Returns:
        A tuple of two elements where each element is described in order below:

        - number of rows in matrix
        - number of columns in matrix

    Raises:
        ValueError  if matrix is not a 2D NumPy ndarray or matrix
    """

    try:
        # will cause AttributeError if matrix is not a 2D NumPy ndarray or matrix
        rows, cols = matrix.shape

        return rows, cols

    except AttributeError:
        raise ValueError("%s is not a 2D NumPy ndarray or matrix." % name)

def get_square_matrix_dimension(name, matrix):
    """
    Return the dimension of a square matrix.

    Args:
        name:    string to print as the matrix's name if an error occurs
        vector:  numpy 2D ndarray or matrix representing a square matrix

    Returns:
        An int representing both the number of rows and number of columns in matrix.

    Raises:
        ValueError  if matrix is not a square 2D NumPy ndarray or matrix
    """

    try:
        # will cause AttributeError if matrix is not a 2D NumPy ndarray or matrix
        rows, cols = matrix.shape

        if rows != cols:
            raise ValueError("%s is not a square matrix. Rows: %i. Cols: %i." % (name, rows, cols))

        return rows

    except AttributeError:
        raise ValueError("%s is not a 2D NumPy ndarray or matrix." % name)

def check_equal_sizes(name_1, size_1, name_2, size_2):
    """
    Check that size_1 and size_2 are equal.

    Args:
        name_1:  string to print as the name of the first element
        size_1:  size of the first element
        name_2:  string to print as the name of the second element
        size_2:  size of the second element
    """
    if size_1 != size_2:
        raise ValueError("Size mismatch between %s and %s." % (name_1, name_2))

def convert_uplo(uplo):
    if uplo == 'u' or uplo == 'U':
        return UPPER
    elif uplo == 'l' or uplo == 'L':
        return LOWER
    else:
        raise ValueError("Parameter 'uplo' must equal one of the following: 'u', 'U', 'l', "
                         "or 'L'. Actual value: %s" % uplo)


# TO BE REMOVED - ALL BELOW THIS LINE

def find_length(m, n, stride):
    """Find the length of a vector to be used for the BLAS parameter "n".
    Note: Adjusting the length by the value of stride prevents a segfault that would otherwise
    occur by ensuring BLAS does not attempt to operate on memory locations past the end
    of the vector.
    Args:
    m: the number of rows in the vector
    n: the number of columns in the vector
    stride: stride of the vector (increment for the elements of the vector)
    Returns:
    The appropriate length to be passed to BLAS for the vector.
    """

    # set length to the max of m and n
    length = m if m > n else n

    # return the ceiling of the result of float division of the length divided by the stride if
    # the stride is greater than 1, else return the length
    return (length / stride) + (length % stride > 0) if stride > 1 else length

def check_is_vector(name, m, n):
    if not (m == 1 or n == 1):
        raise ValueError(str(name) + " must be a vector")

def check_is_square(name, m, n):
    if m != n:
        raise ValueError(str(name) + " must be a square matrix")