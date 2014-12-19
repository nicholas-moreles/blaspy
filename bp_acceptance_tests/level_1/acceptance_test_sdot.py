"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_vector
from blaspy import sdot
from numpy import dot
from itertools import product
from random import randint

N_MIN, N_MAX = 2, 1e6           # matrix/vector sizes
STRIDE_MAX = 1e5                # max vector stride
EPSILON = 0.001                 # margin of error


def acceptance_test_sdot():
    """
    Test extended-precision dot (inner) product.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    outputs = ('float64', 'float32')
    bools = (True, False)
    strides = (1, None)  # None indicates random stride

    # test all combinations of all possible values
    for (output, as_matrix, x_is_row, y_is_row, stride) \
            in product(outputs, bools, bools, bools, strides,):

        # if a test fails, create a string representation of its name and append it to the list
        # of failed tests
        if not passed_test(output, as_matrix, x_is_row, y_is_row, stride):
            variables = (output,
                         "_matrix" if as_matrix else "_ndarray",
                         "_row" if x_is_row else "_col",
                         "_row" if y_is_row else "_col",
                         "_rand_stride" if stride is None else "")
            test_name = "".join(variables)
            tests_failed.append(test_name)

    return tests_failed


def passed_test(output, as_matrix, x_is_row, y_is_row, stride):
    """
    Run one extended-precision dot (inner) product test.

    Arguments:
        output:       BLASpy 'output' parameter to test
        as_matrix:    True to test a NumPy matrix, False to test a NumPy ndarray
        x_is_row:     True to test a row vector as parameter x, False to test a column vector
        y_is_row:     True to test a row vector as parameter y, False to test a column vector
        stride:       stride of x and y to test; if None, a random stride is assigned

    Returns:
        True if the expected result is within the margin of error of the actual result,
        False otherwise.
    """

    # generate random sizes for vector dimensions and vector stride (if necessary)
    length = randint(N_MIN, N_MAX)
    stride = randint(N_MIN, STRIDE_MAX) if stride is None else stride

    # create random vectors to test
    x = random_vector(length, x_is_row, 'float32', as_matrix)
    y = random_vector(length, y_is_row, 'float32', as_matrix)

    # create views of x and y that can be used to calculate the expected result
    x_2 = x if x_is_row else x.T
    y_2 = y.T if y_is_row else y

    # compute the expected result
    if stride == 1:
        expected = dot(x_2, y_2)[0][0]
    else:
        expected = 0
        for i in range(0, length, stride):
            expected += x_2[0, i] * y_2[i, 0]

    # get the actual result
    actual = sdot(x, y, stride, stride, output)

    # compare the actual result to the expected result and return result of the test
    return abs(actual - expected) / expected < EPSILON