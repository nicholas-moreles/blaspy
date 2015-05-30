"""

    Copyright (c) 2014-2015-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_vector, random_symmetric_matrix
from blaspy import symv
from numpy import allclose, copy, dot, triu, tril, zeros
from itertools import product
from random import randint, uniform

N_MIN, N_MAX = 2, 1e3           # matrix/vector sizes
SCAL_MIN, SCAL_MAX = -100, 100  # scalar values
STRIDE_MAX = 1e2                # max vector stride
RTOL, ATOL = 5e-01, 5e-02       # margin of error


def acceptance_test_symv():
    """
    Test symmetric matrix-vector multiplication.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ('float64', 'float32')
    bools = (True, False)
    strides = (1, None)  # None indicates random stride
    uplos = ('u', 'l')

    # test all combinations of all possible values
    for (dtype, as_matrix, x_is_row, y_is_row, provide_y, stride, uplo) \
            in product(dtypes, bools, bools, bools, bools, strides, uplos):

        # avoid testing cases where y is not provided and stride != 1
        if provide_y or stride == 1:

            # if a test fails, create a string representation of its name and append it to the list
            # of failed tests
            if not passed_test(dtype, as_matrix, x_is_row, y_is_row, provide_y, stride, uplo):
                variables = (dtype,
                             "_matrix" if as_matrix else "_ndarray",
                             "_row" if x_is_row else "_col",
                             "_row" if y_is_row else "_col",
                             "_rand_stride" if stride is None else "",
                             "_" if provide_y else "_no_y_",
                             uplo)
                test_name = "".join(variables)
                tests_failed.append(test_name)

    return tests_failed


def passed_test(dtype, as_matrix, x_is_row, y_is_row, provide_y, stride, uplo):
    """
    Run one symmetric matrix-vector multiplication test.

    Arguments:
        dtype:        either 'float64' or 'float32', the NumPy dtype to test
        as_matrix:    True to test a NumPy matrix, False to test a NumPy ndarray
        x_is_row:     True to test a row vector as parameter x, False to test a column vector
        y_is_row:     True to test a row vector as parameter y, False to test a column vector
        provide_y:    True if y is to be provided to the BLASpy function, False otherwise
        stride:       stride of x and y to test; if None, a random stride is assigned
        uplo:         BLASpy uplo parameter to test

    Returns:
        True if the expected result is within the margin of error of the actual result,
        False otherwise.
    """

    # generate random sizes for matrix/vector dimensions and vector stride (if necessary)
    n = randint(N_MIN, N_MAX)
    stride = randint(N_MIN, STRIDE_MAX) if stride is None else stride
    n_A = n / stride + (n % stride > 0)

    # create random scalars, vectors, and matrices to test
    alpha = uniform(SCAL_MIN, SCAL_MAX)
    beta = uniform(SCAL_MIN, SCAL_MAX)
    x = random_vector(n, x_is_row, dtype, as_matrix)
    y = random_vector(n, y_is_row, dtype, as_matrix) if provide_y else None
    A = random_symmetric_matrix(n_A, dtype, as_matrix)

    # create copies/views of x and y that can be used to calculate the expected result
    x_2 = x.T if x_is_row else x
    if y is None:
        y_2 = zeros((1, n))
    else:
        y_2 = copy(y.T) if y_is_row else copy(y)

    # compute the expected result
    if stride == 1:
        y_2 = beta * y_2 + alpha * dot(A, x_2)
    else:
        for i in range(0, y_2.shape[0], stride):
            A_partition = A[i / stride, :]
            x_partition = x_2[:: stride, :]
            y_2[i, 0] = (beta * y_2[i, 0]) + (alpha * dot(A_partition, x_partition))

    # get the actual result
    A = (triu(A) if uplo == 'u' else tril(A)).astype(dtype)
    y = symv(A, x, y, uplo, alpha, beta, inc_x=stride, inc_y=stride)

    # if y is a row vector, make y_2 a row vector as well
    if y.shape[0] == 1:
        y_2 = y_2.T

    # compare the actual result to the expected result and return result of the test
    return allclose(y, y_2, RTOL, ATOL)