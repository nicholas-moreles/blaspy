"""

    Copyright (c) 2014, The University of Texas at Austin..
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

# CBLAS enum values
ROW_MAJOR = 101
COL_MAJOR = 102
NO_TRANS = 111
TRANS = 112
CONJ_TRANS = 113
CONJ_NO_TRANS = 114
UPPER = 121
LOWER = 122

def find_length(m, n, stride):
    """Find the length of a vector to be used for the BLAS parameter "n".

    Note: Adjusting the length by the value of stride prevents a segfault that would otherwise
    occur by ensuring BLAS does not attempt to operate on memory locations past the end
    of the vector.

    Args:
        m:       the number of rows in the vector
        n:       the number of columns in the vector
        stride:  stride of the vector (increment for the elements of the vector)

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

def check_equal_sizes(name_1, size_1, name_2, size_2):
    if size_1 != size_2:
        raise ValueError("size mismatch between " + str(name_1) + " and " + str(name_2))

def convert_uplo(uplo):
    if uplo == 'u' or uplo == 'U':
        return UPPER
    elif uplo == 'l' or uplo == 'L':
        return LOWER
    else:
        raise ValueError("Parameter 'uplo' must equal one of the following: 'u', 'U', 'l', "
                         "or 'L'. Actual value: " + str(uplo))