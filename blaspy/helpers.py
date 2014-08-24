"""

    Copyright (c) 2014, The University of Texas at Austin..
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from .config import _libblas as lib
from .errors import (raise_invalid_dtypes, raise_not_vector, raise_not_2d_numpy, raise_not_square,
                     raise_size_mismatch, raise_strides_not_one, raise_invalid_parameter)
from ctypes import c_double, c_float
from numpy import asmatrix, zeros
from numpy import matrix as np_matrix

# CBLAS_ORDER
ROW_MAJOR = 101
COL_MAJOR = 102

# CBLAS_TRANSPOSE
NO_TRANS = 111
TRANS = 112
CONJ_TRANS = 113
CONJ_NO_TRANS = 114

# CBLAS_UPLO
UPPER = 121
LOWER = 122

# CBLAS_DIAG
NON_UNIT = 131
UNIT = 132

# CBLAS_SIDE
LEFT = 141
RIGHT = 142

# dictionary of BLASpy functions mapping to CBLAS subroutines
# - first entry in value pair is for double precision reals
# - second entry in value pair is for single precision reals
# - sdot is omitted as a special case
FUNC_DICT = {'amax': (lib.cblas_idamax, lib.cblas_isamax),  # level 1
             'asum': (lib.cblas_dasum,  lib.cblas_sasum),
             'axpy': (lib.cblas_daxpy,  lib.cblas_saxpy),
             'copy': (lib.cblas_dcopy,  lib.cblas_scopy),
             'dot':  (lib.cblas_ddot,   lib.cblas_sdot),
             'nrm2': (lib.cblas_dnrm2,  lib.cblas_snrm2),
             'scal': (lib.cblas_dscal,  lib.cblas_sscal),
             'swap': (lib.cblas_dswap,  lib.cblas_sswap),
             'gemv': (lib.cblas_dgemv,  lib.cblas_sgemv),   # level 2
             'ger':  (lib.cblas_dger,   lib.cblas_sger),
             'symv': (lib.cblas_dsymv,  lib.cblas_ssymv),
             'syr':  (lib.cblas_dsyr,   lib.cblas_ssyr),
             'syr2': (lib.cblas_dsyr2,  lib.cblas_ssyr2),
             'trmv': (lib.cblas_dtrmv,  lib.cblas_strmv),
             'gemm': (lib.cblas_dgemm,  lib.cblas_sgemm),   # level 3
             'symm': (lib.cblas_dsymm,  lib.cblas_ssymm)}


def get_cblas_info(calling_func, dtypes):
    """
    Return the appropriate CBLAS subroutine and ctype data type based on the calling function and
    dtypes of args.

    Args:
        calling_func:    a string representation of the calling function
        dtypes:          a tuple of the dtypes of all of the matrices or vectors passed into the
                         calling function

    Returns:
        A tuple of two elements where each element is described in order below:

        - appropriate CBLAS function
        - appropriate ctypes data type

    Raises:
        ValueError: if all matrices and/or vectors do not have the same dtype
    """

    if all(dtype == 'float64' for dtype in dtypes):
        return FUNC_DICT[calling_func][0], c_double

    elif all(dtype == 'float32' for dtype in dtypes):
        return FUNC_DICT[calling_func][1], c_float

    else:
        raise_invalid_dtypes(('float64', 'float32'))


def get_vector_dimensions(name, vector, stride):
    """
    Return the number of rows, number of columns, and length of a vector taking into account
    the stride of the vector.

    Args:
        name:      string to print as the vector's name if an error occurs
        vector:    numpy 2D ndarray or matrix representing a vector
        stride:    stride of the vector (increment for the elements of the vector)

    Returns:
        A tuple of three elements where each element is described in order below:

        - number of rows in vector
        - number of columns in vector
        - length of vector after accounting for stride

    Raises:
        TypeError: if vector is not a vector represented by a 2D NumPy ndarray or matrix
    """

    try:
        rows, cols = vector.shape

        if not (rows == 1 or cols == 1):
            raise_not_vector(name, rows, cols)

        length = max(rows, cols)
        if stride > 1:
            length = (length / stride) + (length % stride > 0)

        return rows, cols, length

    except (AttributeError, TypeError):
        raise_not_2d_numpy(name)


def get_matrix_dimensions(name, matrix):
    """
    Return the number of rows and number of columns in a matrix.

    Args:
        name:      string to print as the vector's name if an error occurs
        vector:    numpy 2D ndarray or matrix representing the matrix

    Returns:
        A tuple of two elements where each element is described in order below:

        - number of rows in matrix
        - number of columns in matrix

    Raises:
        TypeError: if matrix is not a 2D NumPy ndarray or matrix
    """

    try:
        rows, cols = matrix.shape

        return rows, cols

    except (AttributeError, TypeError):
        raise_not_2d_numpy(name)


def get_square_matrix_dimension(name, matrix):
    """
    Return the dimension of a square matrix.

    Args:
        name:      string to print as the matrix's name if an error occurs
        vector:    numpy 2D ndarray or matrix representing a square matrix

    Returns:
        An int representing both the number of rows and number of columns in matrix.

    Raises:
        TypeError:  if matrix is not a 2D NumPy ndarray or matrix
        ValueError: if matrix is not square
    """

    try:
        rows, cols = matrix.shape

        if rows != cols:
            raise_not_square(name, rows, cols)

        return rows

    except (AttributeError, TypeError):
        raise_not_2d_numpy(name)


def check_equal_sizes(name_1, size_1, name_2, size_2):
    """
    Check that size_1 and size_2 are equal.

    Args:
        name_1:    string to print as the name of the first element
        size_1:    size of the first element
        name_2:    string to print as the name of the second element
        size_2:    size of the second element

    Raises:
        ValueError: if size_1 != size_2
    """

    if size_1 != size_2:
        raise_size_mismatch(name_1, name_2)


def check_strides_equal_one(*args):
    if any(stride != 1 for stride in args):
        raise_strides_not_one()


def create_similar_zero_vector(other_vector, length):
    """
    Create and return a zero vector of the given length with the same dtype and orientation as
    other_vector.

    Args:
        other_vector:    vector whose dtype and orientation to copy
        length:          length of the new zero vector

    Returns:
        A new NumPy ndarray or matrix filled with zeros of specified length with the same
        characteristics as other_vector.
    """

    if other_vector.shape[0] == 1:
        new_vector = zeros((1, length), dtype=other_vector.dtype)
    else:
        new_vector = zeros((length, 1), dtype=other_vector.dtype)

    if type(other_vector) is np_matrix:
        new_vector = asmatrix(new_vector)

    return new_vector


def create_zero_matrix(rows, cols, dtype, matrix_type):
    """
    Create and return a zero matrix with the given number of rows and columns, and of the
    appropriate dtype and matrix type.

    Args:
        rows:         number of rows in the new matrix
        cols:         number of columns in the new matrix
        dtype:        NumPy dtype for the elements of the new matrix
        matrix_type:  either NumPy ndarray or matrix; the new matrix will be of the same type

    Returns:
        A new NumPy ndarray or matrix filled with zeros of specified dimensions and dtype.
    """

    new_matrix = zeros((rows, cols), dtype=dtype)

    if matrix_type == np_matrix:
        return asmatrix(new_matrix)
    else:
        return new_matrix


def convert_uplo(uplo):
    if uplo == 'u' or uplo == 'U':
        return UPPER
    elif uplo == 'l' or uplo == 'L':
        return LOWER
    else:
        raise_invalid_parameter('uplo', ('u', 'U', 'l', 'L'), uplo)


def convert_trans(trans):
    if trans == 'n' or trans == 'N':
        return NO_TRANS
    elif trans == 't' or trans == 'T':
        return TRANS
    else:
        raise_invalid_parameter('trans', ('n', 'N', 't', 'T'), trans)


def convert_diag(diag):
    if diag == 'n' or diag == 'N':
        return NON_UNIT
    elif diag == 'u' or diag == 'U':
        return UNIT
    else:
        raise_invalid_parameter('diag', ('n', 'N', 'u', 'U'), diag)


def convert_side(side):
    if side == 'l' or side == 'L':
        return LEFT
    elif side == 'r' or side == 'R':
        return RIGHT
    else:
        raise_invalid_parameter('side', ('l', 'L', 'r', 'R'), side)