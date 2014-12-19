"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_vector, random_matrix
from blaspy import ger
from numpy import allclose, copy,  dot, zeros
from itertools import product
from random import randint, uniform

N_MIN, N_MAX = 2, 1e3           # matrix/vector sizes
SCAL_MIN, SCAL_MAX = -100, 100  # scalar values
STRIDE_MAX = 1e2                # max vector stride
RTOL, ATOL = 5e-03, 5e-05       # margin of error


def acceptance_test_ger():
    """
    Test general rank-1 update.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ('float64', 'float32')
    bools = (True, False)
    strides = (1, None)  # None indicates random stride

    # test all combinations of all possible values
    for (dtype, as_matrix, x_is_row, y_is_row, provide_A, stride) \
            in product(dtypes, bools, bools, bools, bools, strides):

        # if a test fails, create a string representation of its name and append it to the list
        # of failed tests
        if not passed_test(dtype, as_matrix, x_is_row, y_is_row, provide_A, stride):
            variables = (dtype,
                         "_matrix" if as_matrix else "_ndarray",
                         "_row" if x_is_row else "_col",
                         "_row" if y_is_row else "_col",
                         "_rand_stride" if stride is None else "",
                         "" if provide_A else "_no_A")
            test_name = "".join(variables)
            tests_failed.append(test_name)

    return tests_failed


def passed_test(dtype, as_matrix, x_is_row, y_is_row, provide_A, stride):
    """
    Run one general rank-1 update test.

    Arguments:
        dtype:        either 'float64' or 'float32', the NumPy dtype to test
        as_matrix:    True to test a NumPy matrix, False to test a NumPy ndarray
        x_is_row:     True to test a row vector as parameter x, False to test a column vector
        y_is_row:     True to test a row vector as parameter y, False to test a column vector
        provide_A:    True if A is to be provided to the BLASpy function, False otherwise
        stride:       stride of x and y to test; if None, a random stride is assigned

    Returns:
        True if the expected result is within the margin of error of the actual result,
        False otherwise.
    """

    # generate random sizes for matrix/vector dimensions and vector stride (if necessary)
    m = randint(N_MIN, N_MAX)
    n = randint(N_MIN, N_MAX)
    stride_x = randint(N_MIN, STRIDE_MAX) if stride is None else stride
    stride_y = randint(N_MIN, STRIDE_MAX) if stride is None else stride
    m_A = m / stride_x + (m % stride_x > 0)
    n_A = n / stride_y + (n % stride_y > 0)

    # create random scalars, vectors, and matrices to test
    alpha = uniform(SCAL_MIN, SCAL_MAX)
    x = random_vector(m, x_is_row, dtype, as_matrix)
    y = random_vector(n, y_is_row, dtype, as_matrix)
    A = random_matrix(m_A, n_A, dtype, as_matrix) if provide_A else None

    # create copies/views of A, x, and y that can be used to calculate the expected result
    x_2 = x.T if x_is_row else x
    y_2 = y if y_is_row else y.T
    A_2 = zeros((m_A, n_A)) if A is None else copy(A)

    # compute the expected result
    if stride == 1:
        A_2 += alpha * dot(x_2, y_2)
    else:
        for i in range(0, m_A):
                for j in range(0, n_A):
                    A_2[i, j] += alpha * x_2[i * stride_x, 0] * y_2[0, j * stride_y]

    # get the actual result
    A = ger(x, y, A, alpha, inc_x=stride_x, inc_y=stride_y)

    # compare the actual result to the expected result and return result of the test
    return allclose(A, A_2, RTOL, ATOL)