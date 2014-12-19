"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_matrix, random_triangular_matrix
from blaspy import trmm
from numpy import allclose, copy, dot, zeros
from itertools import product
from random import randint, uniform

N_MIN, N_MAX = 2, 1e3           # matrix/vector sizes
SCAL_MIN, SCAL_MAX = -10, 10    # scalar values
STRIDE_MAX = 1e2                # max vector stride
RTOL, ATOL = 5e-01, 5e-02       # margin of error


def acceptance_test_trmm():
    """
    Test triangular matrix-matrix multiplication.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ('float64', 'float32')
    bools = (True, False)
    uplos = ('u', 'l')
    sides = ('l', 'r')
    trans_tuple = ('n', 't')
    diags = ('n', 'u')


    # test all combinations of all possible values
    for (dtype, as_matrix, uplo, side, trans_a, diag) \
            in product(dtypes, bools, uplos, sides, trans_tuple, diags):

        # if a test fails, create a string representation of its name and append it to the list
        # of failed tests
        if not passed_test(dtype, as_matrix, uplo, side, trans_a, diag):
            variables = (dtype,
                         "_matrix_" if as_matrix else "_ndarray_",
                         uplo, "_",
                         side, "_",
                         trans_a, "_",
                         diag)
            test_name = "".join(variables)
            tests_failed.append(test_name)

    return tests_failed

def passed_test(dtype, as_matrix, uplo, side, trans_a, diag):
    """
    Run one symmetric matrix-matrix multiplication test.

    Arguments:
        dtype:        either 'float64' or 'float32', the NumPy dtype to test
        as_matrix:    True to test a NumPy matrix, False to test a NumPy ndarray
        uplo:         BLASpy 'uplo' parameter to test
        side:         BLASpy 'side' parameter to test
        trans_a:      BLASpy 'trans_a' parameter to test
        diag:         BLASpy 'diag' parameter to test

    Returns:
        True if the expected result is within the margin of error of the actual result,
        False otherwise.
    """

    side_is_left = side == 'l' or side == 'L'
    transpose = trans_a == 't' or trans_a == 'T'

    # generate random sizes for matrix dimensions
    m = randint(N_MIN, N_MAX)
    n = randint(N_MIN, N_MAX)
    k = m if side_is_left else n

    # create random scalars and matrices to test
    alpha = uniform(SCAL_MIN, SCAL_MAX)
    A = random_triangular_matrix(k, dtype, as_matrix, uplo, diag, trans_a)
    B = random_matrix((k if side_is_left else m), (n if side_is_left else k), dtype, as_matrix)

    # compute the expected result
    B_2 = alpha * (dot(A, B) if side_is_left else dot(B, A))

    # get the actual result
    trmm(A, B, side, uplo, trans_a, diag, alpha)

    # compare the actual result to the expected result and return result of the test
    return allclose(B, B_2, RTOL, ATOL)