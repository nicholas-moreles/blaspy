"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_vector
from blaspy import scal
from numpy import allclose, copy
from itertools import product
from random import randint, uniform

N_MIN, N_MAX = 2, 1e6           # matrix/vector sizes
SCAL_MIN, SCAL_MAX = -100, 100  # scalar values
STRIDE_MAX = 1e5                # max vector stride


def acceptance_test_scal():
    """
    Test vector scaling.

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
            variables = (dtype,
                         "_matrix" if as_matrix else "_ndarray",
                         "_row" if x_is_row else "_col",
                         "_rand_stride" if stride is None else "")
            test_name = "".join(variables)
            tests_failed.append(test_name)

    return tests_failed


def passed_test(dtype, as_matrix, x_is_row, stride):
    """
    Run one vector scaling test.

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

    # create random scalar and vector to test
    alpha = uniform(SCAL_MIN, SCAL_MAX)
    x = random_vector(length, x_is_row, dtype, as_matrix)

    # create copy of x to hold the expected result
    x_2 = copy(x.T) if x_is_row else copy(x)

    # compute the expected result
    if stride == 1:
        x_2 *= alpha
    else:
        for i in range(0, length, stride):
            x_2[i, 0] *= alpha

    # get the actual result
    scal(alpha, x, stride)

    # if x is a row vector, make x_2 a row vector as well
    x_2 = x_2.T if x.shape[0] == 1 else x_2

    # compare the actual result to the expected result and return result of the test
    return allclose(x, x_2)