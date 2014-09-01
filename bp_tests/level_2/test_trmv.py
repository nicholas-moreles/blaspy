"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_vector, random_triangular_matrix
from blaspy import trmv
from numpy import allclose, copy, dot
from itertools import product
from random import randint

N_MIN, N_MAX = 2, 1e3           # matrix/vector sizes
SCAL_MIN, SCAL_MAX = -100, 100  # scalar values
STRIDE_MAX = 1e2                # max vector stride
RTOL, ATOL = 5e-01, 5e-02       # margin of error


def test_trmv():
    """
    Test triangular matrix-vector multiplication.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ('float64', 'float32')
    bools = (True, False)
    strides = (1, None)  # None indicates random stride
    uplos = ('u', 'l')
    trans_tuple = ('n', 't')
    diags = ('n', 'u')

    # test all combinations of all possible values
    for (dtype, as_matrix, x_is_row, stride, uplo, trans, diag) \
            in product(dtypes, bools, bools, strides, uplos, trans_tuple, diags):

        # if a test fails, create a string representation of its name and append it to the list
        # of failed tests
        if not passed_test(dtype, as_matrix, x_is_row, stride, uplo, trans, diag):
            variables = (dtype,
                         "_matrix" if as_matrix else "_ndarray",
                         "_row" if x_is_row else "_col",
                         "_rand_stride_" if stride is None else "_",
                         uplo, "_",
                         trans, "_",
                         diag)
            test_name = "".join(variables)
            tests_failed.append(test_name)

    return tests_failed

def passed_test(dtype, as_matrix, x_is_row, stride, uplo, trans_a, diag):
    """
    Run one triangular matrix-vector multiplication test.

    Arguments:
        dtype:        either 'float64' or 'float32', the NumPy dtype to test
        as_matrix:    True to test a NumPy matrix, False to test a NumPy ndarray
        x_is_row:     True to test a row vector as parameter x, False to test a column vector
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
    x = random_vector(n, x_is_row, dtype, as_matrix)
    A = random_triangular_matrix(n_A, dtype, as_matrix, uplo, diag)

    # create copies/views of A, x, and y that can be used to calculate the expected result
    A_2 = A if trans_a == 'n' else A.T
    x_2 = copy(x.T) if x_is_row else copy(x)
    x_3 = copy(x.T) if x_is_row else copy(x)

    # compute the expected result
    if stride == 1:
        x_2 = dot(A_2, x_2)
    else:
        for i in range(0, n, stride):
            A_partition = A_2[i / stride, :]
            x_partition = x_3[:: stride, :]
            x_2[i, 0] = dot(A_partition, x_partition)

    # get the actual result
    trmv(A, x, uplo, trans_a, diag, inc_x=stride,)

    # if x is a row vector, make x_2 a row vector as well
    if x_is_row:
        x_2 = x_2.T

    # compare the actual result to the expected result and return result of the test
    return allclose(x, x_2, RTOL, ATOL)