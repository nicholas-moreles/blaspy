"""

    Copyright (c) 2014-2015-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_vector
from blaspy import amax
from numpy import absolute, argmax
from itertools import product
from random import randint

N_MIN, N_MAX = 2, 1e6           # matrix/vector sizes
STRIDE_MAX = 1e5                # max vector stride


def acceptance_test_amax():
    """
    Test amax.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ('float64', 'float32')
    bools = (True, False)
    strides = (1, None)  # None indicates random stride

    # test all combinations of all possible values
    for (dtype, as_matrix, x_is_row, stride) in product(dtypes, bools, bools, strides):

        # if a test fails, create a string representation of its name and append it to the list
        # of failed tests
        if not passed_test(dtype, as_matrix, x_is_row, stride):
            variable_list = [dtype,
                             "_matrix" if as_matrix else "_ndarray",
                             "_row" if x_is_row else "_col",
                             "_rand_stride" if stride is None else ""]
            test_name = "".join(variable_list)
            tests_failed.append(test_name)

    return tests_failed


def passed_test(dtype, as_matrix, x_is_row, stride):
    """
    Run one amax test.

    Arguments:
        dtype:        either 'float64' or 'float32', the NumPy dtype to test
        as_matrix:    True to test a NumPy matrix, False to test a NumPy ndarray
        x_is_row:     True to test a row vector as parameter x, False to test a column vector
        stride:       stride of x and y to test; if None, a random stride is assigned

    Returns:
        True if the expected result is within the margin of error of the actual result,
        False otherwise.
    """

    # generate random sizes for vector dimensions and vector stride (if necessary)
    length = randint(N_MIN, N_MAX)
    stride = randint(N_MIN, STRIDE_MAX) if stride is None else stride

    # create random vector to test
    x = random_vector(length, x_is_row, dtype, as_matrix)

    # create view of x that can be used to calculate the expected result
    x_2 = x.T if x_is_row else x

    # compute the expected result
    if stride == 1:
        expected = argmax(absolute(x))
    else:
        expected = -1
        current_max = -1
        for i in range(0, length, stride):
            if abs(x_2[i, 0]) > current_max:
                current_max = abs(x_2[i, 0])
                expected = i / stride

    # get the actual result
    actual = amax(x, stride)

    # compare the actual result to the expected result and return result of the test
    return actual == expected