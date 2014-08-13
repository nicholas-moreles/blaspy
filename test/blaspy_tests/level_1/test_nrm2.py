"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import nrm2


from ..helpers import random_vector
from blaspy import nrm2
from numpy.linalg import norm
from itertools import product
from random import randint

N_MIN, N_MAX = 2, 1e6           # matrix/vector sizes
STRIDE_MAX = 1e5                # max vector stride
EPSILON = 0.001                 # margin of error


def test_nrm2():
    """
    Test 2-norm computation.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ['float64', 'float32']
    bools = [True, False]
    strides = [1, None]  # None indicates random stride

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
    Run 2-norm computation test.

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

    # compute the expected result
    if stride == 1:
        expected = norm(x)
    else:
        expected = 0
        for i in range(0, length, stride):
            expected += abs(x[0, i] if x_is_row else x[i, 0]) ** 2
        expected **= 0.5

    # get the actual result
    actual = nrm2(x, stride)

    # compare the actual result to the expected result and return result of the test
    return abs(actual - expected) / expected < EPSILON
#
#
# def test_nrm2():
#     random.seed()
#     tests_failed = []
#     epsilon = 0.001  # account for round-off/precision error
#
#     # run one particular test
#     def passed_test(x_is_row, n=None, stride=None):
#
#         # set random values for n, alpha, and stride if none are passed in
#         if n is None:
#             n = random.randint(2, 1e6)
#         if stride is None:
#             stride = random.randint(2, 1e5)
#
#         # create the vector to test
#         x = random_vector(n, x_is_row, dtype, as_matrix)
#
#         # get the expected result
#         if stride == 1:
#             expected = norm(x)
#         else:
#             expected = 0
#             for i in range(0, n, stride):
#                 if x_is_row:
#                     expected += abs(x[0, i]) ** 2
#                 else:
#                     expected += abs(x[i, 0]) ** 2
#             expected **= 0.5
#
#         # compare BLASpy result to expected result
#         actual = nrm2(x, stride)
#         return abs(actual - expected) / expected < epsilon
#
#     # run all tests of the given type
#     def run_tests():
#
#         # Test 1 - 2-norm of a scalar
#         test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_scalar"
#         if not passed_test(ROW, n=1, stride=1):
#             tests_failed.append(test_name)
#
#         # Test 2 - 2-norm of a column vector
#         test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col"
#         if not passed_test(COL, stride=1):
#             tests_failed.append(test_name)
#
#         # Test 3 - 2-norm of a row vector
#         test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row"
#         if not passed_test(ROW, stride=1):
#             tests_failed.append(test_name)
#
#         # Test 4 - 2-norm of a column vector with a random stride
#         test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_rand_stride"
#         if not passed_test(COL):
#             tests_failed.append(test_name)
#
#         # Test 5 - 2-norm of a row vector with a random stride
#         test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_rand_stride"
#         if not passed_test(ROW):
#             tests_failed.append(test_name)
#
#     # Test dnrm2 with ndarray
#     dtype = 'float64'
#     as_matrix = NDARRAY
#     run_tests()
#
#     # Test dnrm2 with matrix
#     as_matrix = MATRIX
#     run_tests()
#
#     # Test snrm2 with ndarray
#     dtype = 'float32'
#     as_matrix = NDARRAY
#     run_tests()
#
#     # Test snrm2 with matrix
#     as_matrix = MATRIX
#     run_tests()
#
#     return tests_failed