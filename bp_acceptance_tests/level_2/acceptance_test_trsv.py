"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_vector, random_triangular_matrix
from blaspy import trsv, gemv
from numpy import allclose, copy, fill_diagonal, zeros
from itertools import product
from random import randint, uniform

N_MIN, N_MAX = 2, 3000       # matrix/vector sizes
STRIDE_MAX = 1000            # max vector stride
RTOL, ATOL = 5e-01, 5e-02    # margin of error


def acceptance_test_trsv():
    """
    Test triangular solve.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ('float64', 'float32')
    bools = (True, False)
    strides = (1, None)
    uplos = ('l', 'u')
    trans_tuple = ('t', 'n')
    diags = ('n', 'u')

    # test all combinations of all possible values
    for (dtype, as_matrix, b_is_row, stride, uplo, trans, diag) \
            in product(dtypes, bools, bools, strides, uplos, trans_tuple, diags):

        # if a test fails, create a string representation of its name and append it to the list
        # of failed tests
        if not passed_test(dtype, as_matrix, b_is_row, stride, uplo, trans, diag):
            variables = (dtype,
                         "_matrix" if as_matrix else "_ndarray",
                         "_row" if b_is_row else "_col",
                         "_rand_stride_" if stride is None else "_",
                         uplo, "_",
                         trans, "_",
                         diag)
            test_name = "".join(variables)

            print(test_name)
            tests_failed.append(test_name)

    return tests_failed


def passed_test(dtype, as_matrix, b_is_row, stride, uplo, trans_a, diag):
    """
    Run one triangular solve test.

    Arguments:
        dtype:        either 'float64' or 'float32', the NumPy dtype to test
        as_matrix:    True to test a NumPy matrix, False to test a NumPy ndarray
        b_is_row:     True to test a row vector as parameter b, False to test a column vector
        stride:       stride of x and y to test; if None, a random stride is assigned
        uplo:         BLASpy uplo parameter to test
        trans_a:      BLASpy trans parameter to test
        diag:         BLASpy diag parameter to test

    Returns:
        True if the expected result is within the margin of error of the actual result,
        False otherwise.
    """

    # generate random sizes for matrix/vector dimensions and vector stride (if necessary)
    n = randint(N_MIN, N_MAX)
    stride = randint(N_MIN, STRIDE_MAX) if stride is None else stride
    n_A = n / stride + (n % stride > 0)

    # create random vectors and matrices to test
    b = random_vector(n, b_is_row, dtype, as_matrix)
    A = random_triangular_matrix(n_A, dtype, as_matrix, uplo, diag)
    A /= n_A  # scale off-diagonal to avoid numerical issues

    # fill diagonal with 1 if unit-triangular, else fill diagonal with values between 1 and 2
    if diag == 'u':
        fill_diagonal(A, 1)
    else:
        for i in range(n_A):
            A[i, i] = uniform(1, 2)

    # copy b for comparison later
    expected = copy(b.T) if b_is_row else copy(b)

    # solve for x
    trsv(A, b, uplo, trans_a, diag, inc_b=stride)
    b = b.T if b_is_row else b

    # compute actual result
    actual = gemv(A, b, zeros((n, 1), dtype=dtype), trans_a=trans_a, inc_x=stride, inc_y=stride)
    if stride != 1:
        for i in range(n):
            if i % stride != 0:  # zero out all non-relevant entries in expected vector
                expected[i] = 0

    # compare the actual result to the expected result and return result of the test
    return allclose(actual, expected, RTOL, ATOL)