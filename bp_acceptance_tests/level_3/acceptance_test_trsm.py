"""

    Copyright (c) 2014-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_matrix, random_triangular_matrix
from blaspy import trsm, trmm
from numpy import allclose, fill_diagonal
from itertools import product
from random import randint, uniform

N_MIN, N_MAX = 2, 3000       # matrix/vector sizes
SCAL_MIN, SCAL_MAX = 1, 2    # scalar values
RTOL, ATOL = 5e-01, 5e-02    # margin of error


def acceptance_test_trsm():
    """
    Test triangular solve with multiple right-hand sides.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ('float64', 'float32')
    bools = (True, False)
    sides = ('l', 'r')
    uplos = ('u', 'l')
    trans_tuple = ('n', 't')
    diags = ('n', 'u')

    # test all combinations of all possible values
    for (dtype, as_matrix, side, uplo, trans_a, diag) \
            in product(dtypes, bools, sides, uplos, trans_tuple, diags):

        # if a test fails, create a string representation of its name and append it to the list
        # of failed tests
        if not passed_test(dtype, as_matrix, side, uplo, trans_a, diag):
            variables = (dtype,
                         "_matrix_" if as_matrix else "_ndarray_",
                         side, "_",
                         uplo, "_",
                         trans_a, "_",
                         diag)
            test_name = "".join(variables)
            tests_failed.append(test_name)

    return tests_failed

def passed_test(dtype, as_matrix, side, uplo, trans_a, diag):
    """
    Run one triangular solve with multiple right-hand sides test.

    Arguments:
        dtype:        either 'float64' or 'float32', the NumPy dtype to test
        as_matrix:    True to test a NumPy matrix, False to test a NumPy ndarray
        side:         BLASpy 'side' parameter to test
        uplo:         BLASpy 'uplo' parameter to test
        trans_a:      BLASpy 'trans_a' parameter to test
        diag:         BLASpy 'diag' parameter to test

    Returns:
        True if the expected result is within the margin of error of the actual result,
        False otherwise.
    """

    side_is_left = side == 'l' or side == 'L'
    a_is_unit = diag == 'u' or diag == 'U'

    # generate random sizes for matrix dimensions
    m = randint(N_MIN, N_MAX)
    n = randint(N_MIN, N_MAX)
    dim_A = m if side_is_left else n

    # create random scalars and matrices to test
    alpha = uniform(SCAL_MIN, SCAL_MAX)
    A = random_triangular_matrix(dim_A, dtype, as_matrix, uplo, diag='n')
    B = random_matrix(m, n, dtype, as_matrix)

    # scale off-diagonals to avoid numerical issues
    A /= dim_A
    B /= dim_A

    # fill diagonal with 1 if unit-triangular, else fill diagonal with values between 1 and 2
    if a_is_unit:
        fill_diagonal(A, 1)
    else:
        for i in range(dim_A):
            A[i, i] = uniform(1, 2)

    # compute the expected result
    expected = alpha * B

    #compute the actual result
    trsm(A, B, side, uplo, trans_a, diag, alpha)
    trmm(A, B, side, uplo, trans_a, diag)
    actual = B

    # compare the actual result to the expected result and return result of the test
    return allclose(actual, expected, RTOL, ATOL)