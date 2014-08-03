"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import copy
from ..helpers import random_vector, COL, ROW, NDARRAY, MATRIX
from numpy import allclose, transpose, zeros
from numpy import copy as np_copy
import random


def test_copy():
    random.seed()
    tests_failed = []

    # run one particular test
    def passed_test(x_is_row, y_is_row, n=None, stride=None):

        # set random values for n, alpha, and stride if none are passed in
        if n is None:
            n = random.randint(2, 1e6)
        if stride is None:
            stride = random.randint(2, 1e5)

        # create the vectors to test
        x = random_vector(n, x_is_row, dtype, as_matrix)
        y = random_vector(n, y_is_row, dtype, as_matrix)
        assert x.dtype == y.dtype

        # get the expected result
        if stride == 1:
            expected = x if x_is_row == y_is_row else transpose(x)
        else:
            expected = np_copy(y)
            for i in range(0, n, stride):
                if x_is_row:
                    if y_is_row:  # y is row, x is row
                        expected[0, i] = x[0, i]
                    else:  # y is col, x is row
                        expected[i, 0] = x[0, i]
                else:
                    if y_is_row:  # y is row, x is col
                        expected[0, i] = x[i, 0]
                    else:  # y is col, x is col
                        expected[i, 0] = x[i, 0]

        # compare the actual result to the expected result
        copy(x, y, stride, stride)
        return allclose(y, expected)

    # run all tests of the given type
    def run_tests():

        # two scalars
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_scalars"
        if not passed_test(ROW, ROW, n=1, stride=1):
            tests_failed.append(test_name)

        # two column vectors
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_col"
        if not passed_test(COL, COL, stride=1):
            tests_failed.append(test_name)

        # two row vectors
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_row"
        if not passed_test(ROW, ROW, stride=1):
            tests_failed.append(test_name)

        # column vector and a row vector
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_row"
        if not passed_test(COL, ROW, stride=1):
            tests_failed.append(test_name)

        # two row vectors
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_col"
        if not passed_test(ROW, COL, stride=1):
            tests_failed.append(test_name)

        # two column vectors with the same random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_col_rand_stride"
        if not passed_test(COL, COL):
            tests_failed.append(test_name)

        # two row vectors with the same random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_row_rand_stride"
        if not passed_test(ROW, ROW):
            tests_failed.append(test_name)

        # column vector and a row vector with the same random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_row_rand_stride"
        if not passed_test(COL, ROW):
            tests_failed.append(test_name)

        # two row vectors with the same random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_col_rand_stride"
        if not passed_test(ROW, COL):
            tests_failed.append(test_name)

    # Test daxpy as ndarray
    dtype = 'float64'
    as_matrix = NDARRAY
    run_tests()

    # Test daxpy as matrix
    as_matrix = MATRIX
    run_tests()

    # Test saxpy as ndarray
    dtype = 'float32'
    as_matrix = NDARRAY
    run_tests()

    # Test saxpy as matrix
    as_matrix = MATRIX
    run_tests()

    return tests_failed