"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_vector, random_symmetric_matrix
from .expected_results import expected_symv
from blaspy import symv
from numpy import allclose, triu, tril
from itertools import product
from random import randint, uniform


def test_symv():
    """
    Test symmetric matrix-vector multiplication.

    Returns:
        A list of strings representing the failed tests.
    """
    tests_failed = []

    # values to test
    uplos = ['u', 'l']
    dtypes = ['float64', 'float32']
    bools = [True, False]
    strides = [1, None]

    # test all combinations of all possible values
    for (dtype, as_matrix, x_is_row, y_is_row, provide_y, stride, uplo) \
            in product(dtypes, bools, bools, bools, bools, strides, uplos):

        # avoid testing cases where y is not provided and stride != 1
        if provide_y or stride == 1:
            if not passed_test(dtype, as_matrix, x_is_row, y_is_row, provide_y, stride, uplo):

                # test has failed, create a string representing its name
                variable_list = [dtype,
                                 "_matrix" if as_matrix else "_ndarray",
                                 "_row" if x_is_row else "_col",
                                 "_row" if y_is_row else "_col",
                                 "_rand_stride" if stride is None else "",
                                 "_" if provide_y else "_no_y_",
                                 uplo]
                test_name = "".join(variable_list)

                # append that string to the list of failed tests
                tests_failed.append(test_name)

    return tests_failed


def passed_test(dtype, as_matrix, x_is_row, y_is_row, provide_y, stride, uplo):
    """
    Run one symmetric matrix-vector multiplication test.

    Arguments:
        dtype:      either 'float64' or 'float32', the NumPy dtype to test
        as_matrix:  True to test a NumPy matrix, False to test a NumPy ndarray
        x_is_row:   True to test a row vector as parameter x, False to test a column vector
        y_is_row:   True to test a row vector as parameter y, False to test a column vector
        provide_y:  True if y is to be provided to the BLASpy function, False otherwise
        stride:     stride of x and y to test; if None, a random stride is assigned
        uplo:       BLASpy uplo parameter to test

    Returns:
        True if the expected result was within the margin of error of the actual result.
        False otherwise.
    """

    n = randint(2, 1e3)
    alpha = uniform(-100, 100)
    beta = uniform(-100, 100)
    stride = randint(2, 1e2) if stride is None else stride

    # create the random matrix and vectors to test
    A = random_symmetric_matrix(n / stride + (n % stride > 0), dtype, as_matrix)
    x = random_vector(n, x_is_row, dtype, as_matrix)
    y = random_vector(n, y_is_row, dtype, as_matrix) if provide_y else None

    # get the expected result
    expected = expected_symv(A, x, x_is_row, y, y_is_row, n, alpha, beta, stride)

    # get the actual result
    A = triu(A) if uplo == 'u' else tril(A)
    y = symv(A, x, y, uplo, alpha, beta, inc_x=stride, inc_y=stride)

    # update the orientation of y as it may have changed
    y_is_row = y.shape[0] == 1

    # compare the actual result to the expected result
    if y_is_row:
        return allclose(y.T, expected, rtol=5e-02, atol=5e-04)
    else:
        return allclose(y, expected, rtol=5e-02, atol=5e-04)