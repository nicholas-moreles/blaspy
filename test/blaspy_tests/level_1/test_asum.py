"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import asum
from ..helpers import random_vector, COL, ROW, NDARRAY, MATRIX
from numpy import sum as np_sum
from numpy import absolute
import random


def test_asum():
    random.seed()
    tests_failed = []
    epsilon = 0.0001  # account for round-off/precision error

    # run one particular test
    def passed_test(x_is_row, n=None, stride=None):

        # set random values for n, alpha, and stride if none are passed in
        if n is None:
            n = random.randint(2, 1e6)
        if stride is None:
            stride = random.randint(2, 1e5)

        # create the vector to test
        x = random_vector(n, x_is_row, dtype, as_matrix)

        # get the expected result
        if stride == 1:
            expected = np_sum(absolute(x))
        else:
            expected = 0
            for i in range(0, n, stride):
                if x_is_row:
                    expected += abs(x[0, i])
                else:
                    expected += abs(x[i, 0])

        # compare BLASpy result to expected result
        actual = asum(x, stride)
        return abs(actual - expected) / expected < epsilon

    # run all tests of the given type
    def run_tests():

        # Test 1 - sum a scalar
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_scalar"
        if not passed_test(ROW, n=1, stride=1):
            tests_failed.append(test_name)

        # Test 2 - sum a column vector
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col"
        if not passed_test(COL, stride=1):
            tests_failed.append(test_name)

        # Test 3 - sum a row vector
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row"
        if not passed_test(ROW, stride=1):
            tests_failed.append(test_name)

        # Test 4 - sum a column vector with a random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_rand_stride"
        if not passed_test(COL):
            tests_failed.append(test_name)

        # Test 5 - sum a row vector with a random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_rand_stride"
        if not passed_test(ROW):
            tests_failed.append(test_name)

    # Test dscal as ndarray
    dtype = 'float64'
    as_matrix = NDARRAY
    run_tests()

    # Test dscal as matrix
    as_matrix = MATRIX
    run_tests()

    # Test sscal as ndarray
    dtype = 'float32'
    as_matrix = NDARRAY
    run_tests()

    # Test sscal as matrix
    as_matrix = MATRIX
    run_tests()

    return tests_failed
