"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import scal
from ..helpers import random_vector, COL, ROW, NDARRAY, MATRIX
from numpy import allclose, copy
import random


def test_scal():
    random.seed()
    tests_failed = []

    # run one particular test
    def passed_test(x_is_row, n=None, alpha=None, stride=None):

        # set random values for n, alpha, and stride if none are passed in
        if n is None:
            n = random.randint(2, 1e6)
        if alpha is None:
            alpha = random.randint(-100, 100)
        if stride is None:
            stride = random.randint(2, 1e5)

        # create the vector to test
        x = random_vector(n, x_is_row, dtype, as_matrix)

        # get the expected result
        if stride == 1:
            expected = alpha * x
        else:
            expected = copy(x)
            for i in range(0, n, stride):
                if x_is_row:
                    expected[0, i] *= alpha
                else:
                    expected[i, 0] *= alpha

        # compare BLASpy result to expected result
        scal(alpha, x, stride)
        return allclose(x, expected)

    # run all tests of the given type
    def run_tests():

        # Test 1 - scale a scalar
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_scalar"
        if not passed_test(ROW, n=1, stride=1):
            tests_failed.append(test_name)

        # Test 2 - scale a column vector
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col"
        if not passed_test(COL, stride=1):
            tests_failed.append(test_name)

        # Test 3 - scale a row vector
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row"
        if not passed_test(ROW, stride=1):
            tests_failed.append(test_name)

        # Test 4 - scale a column vector by zero
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_zero"
        if not passed_test(COL, alpha=0, stride=1):
            tests_failed.append(test_name)

        # Test 5 - scale a row vector by one
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_one"
        if not passed_test(ROW, alpha=1, stride=1):
            tests_failed.append(test_name)

        # Test 6 - scale a column vector with a random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_rand_stride"
        if not passed_test(COL):
            tests_failed.append(test_name)

        # Test 7 - scale a row vector with a random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_rand_stride"
        if not passed_test(ROW):
            tests_failed.append(test_name)

    # Test dscal with ndarray
    dtype = 'float64'
    as_matrix = NDARRAY
    run_tests()

    # Test dscal with matrix
    as_matrix = MATRIX
    run_tests()

    # Test sscal with ndarray
    dtype = 'float32'
    as_matrix = NDARRAY
    run_tests()

    # Test sscal with matrix
    as_matrix = MATRIX
    run_tests()

    return tests_failed
