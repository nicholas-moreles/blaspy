"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from numpy import asmatrix, random, ndarray
from numpy import matrix as np_matrix

# constants to clarify what is being tested in the test functions
COL, ROW = False, True
NDARRAY, MATRIX = False, True

def random_vector(length, is_row, dtype, as_matrix):
    assert dtype == 'float32' or dtype == 'float64'
    m, n = (1, length) if is_row else (length, 1)
    vector = random.uniform(-10, 10, (m, n)).astype(dtype)
    if as_matrix:
        vector = asmatrix(vector)
    assert vector.shape == (1, n) if is_row else (n, 1)
    assert vector.dtype == dtype
    assert type(vector) == np_matrix if as_matrix else type(vector) == ndarray
    return vector

def random_matrix(m, n, dtype, as_matrix):
    assert dtype == 'float32' or dtype == 'float64'
    matrix = random.uniform(-10, 10, (m, n)).astype(dtype)
    if as_matrix:
        matrix = asmatrix(matrix)
    assert matrix.shape == (m, n)
    assert matrix.dtype == dtype
    assert type(matrix) == np_matrix if as_matrix else type(matrix) == ndarray
    return matrix