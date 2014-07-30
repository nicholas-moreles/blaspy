"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import scal
from ..helpers import random_vector, COL, ROW, NDARRAY, MATRIX
from numpy import allclose
import random

def test_scal():

    random.seed()
    tests_failed = []

    # run tests of the given type
    def run_tests(dtype, as_matrix):
        assert dtype == 'float32' or dtype == 'float64'

        # run one particular test
        def passed_test(x_is_row, n=None, alpha=None):

            # set random values for n and alpha if none are passed in
            if n is None:
                n = random.randint(2, 1e6)
            if alpha is None:
                alpha = random.randint(-100, 100)

            # create the vector to test
            x = random_vector(n, x_is_row, dtype, as_matrix)

            # compare BLASpy result to expected result
            expected = x * alpha
            scal(alpha, x)
            return allclose(x, expected)

        # Test 1 - scale a scalar
        test_name = dtype + "_matrix" if as_matrix else "_ndarray" + "_scalar"
        if not passed_test(ROW, n=1): tests_failed.append(test_name)

        # Test 2 - scale a column vector
        test_name = dtype + "_matrix" if as_matrix else "_ndarray" + "_col"
        if not passed_test(COL): tests_failed.append(test_name)

        # Test 3 - scale a row vector
        test_name = dtype + "_matrix" if as_matrix else "_ndarray" + "_row"
        if not passed_test(ROW): tests_failed.append(test_name)

        # Test 4 - scale a column vector by zero
        test_name = dtype + "_matrix" if as_matrix else "_ndarray" + "_col_zero"
        if not passed_test(COL, alpha=0): tests_failed.append(test_name)

        # Test 5 - scale a row vector by one
        test_name = dtype + "_matrix" if as_matrix else "_ndarray" + "_row_one"
        if not passed_test(ROW, alpha=1): tests_failed.append(test_name)

    # Test dscal as ndarray
    run_tests('float64', NDARRAY)

    # Test dscal as matrix
    run_tests('float64', MATRIX)

    # Test sscal as ndarray
    run_tests('float32', NDARRAY)

    # Test sscal as matrix
    run_tests('float32', MATRIX)

    # TODO: Add tests with different strides

    return tests_failed
