"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_vector, random_triangular_matrix
from blaspy import trsv
from numpy import allclose, copy, dot, fill_diagonal
from itertools import product
from random import randint

N_MIN, N_MAX = 2, 100           # matrix/vector sizes
OFFSET = 100                    # avoid singularity
STRIDE_MAX = 10                 # max vector stride
RTOL, ATOL = 5e-01, 5e-02       # margin of error


def test_trsv():
    """
    Test triangular solve.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ('float64', 'float32')
    bools = (True, False)
    strides = (1,)  # TODO: Test random strides
    uplos = ('l', 'u')
    trans_tuple = ('t', 'n')
    diags = ('n',) # TODO: Discover why unit triangular matrices fail

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
        trans:        BLASpy trans parameter to test
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
    x = random_vector(n, False, dtype, as_matrix)
    x += OFFSET
    A = random_triangular_matrix(n_A, dtype, as_matrix, uplo, diag)
    if diag == 'n':
        fill_diagonal(A, A.diagonal() + OFFSET)
    A_2 = A.T if trans_a == 't' or trans_a == 'T' else A
    b = dot(A_2, x).T if b_is_row else dot(A_2, x)
    b_2 = copy(b.T) if b_is_row else copy(b)

    trsv(A, b, uplo, trans_a, diag, inc_b=stride)
    b = b.T if b_is_row else b

    # compare the actual result to the expected result and return result of the test
    return allclose(dot(A, b), b_2, RTOL, ATOL)