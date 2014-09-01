"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from numpy import asmatrix, fill_diagonal, random, tril, triu

# min and max values for elements of random matrices and vectors
MIN = -10
MAX = 10

def random_vector(length, is_row, dtype, as_matrix):
    """ Generate a random vector """
    m, n = (1, length) if is_row else (length, 1)
    vector = random.uniform(MIN, MAX, (m, n)).astype(dtype)
    if as_matrix:
        vector = asmatrix(vector)
    return vector


def random_matrix(m, n, dtype, as_matrix):
    """ Generate a random matrix """
    matrix = random.uniform(MIN, MAX, (m, n)).astype(dtype)
    if as_matrix:
        matrix = asmatrix(matrix)
    return matrix


def random_symmetric_matrix(n, dtype, as_matrix):
    """ Generate a random symmetric matrix """
    rand_matrix = random_matrix(n, n, dtype, as_matrix)
    return (rand_matrix + rand_matrix.T) / 2


def random_triangular_matrix(n, dtype, as_matrix, uplo, diag, trans='n'):
    """ Generate a random triangular matrix """
    rand_matrix = random_matrix(n, n, dtype, as_matrix)
    if diag == 'u' or diag == 'U':
        fill_diagonal(rand_matrix, 1)
    rand_matrix = triu(rand_matrix) if (uplo == 'u' or uplo == 'U') else tril(rand_matrix)
    return (rand_matrix if (trans == 'n' or trans == 'N') else rand_matrix.T).astype(dtype)