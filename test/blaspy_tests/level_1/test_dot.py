"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import dot
from ..helpers import random_vector, COL, ROW, NDARRAY, MATRIX
from numpy import dot as np_dot
from numpy import transpose
import random

def test_dot():

    random.seed()
    tests_failed = []
    epsilon = 0.0001  # account for round-off/precision error

    # run tests of the given type
    def run_tests(dtype, as_matrix):
        assert dtype == 'float32' or dtype == 'float64'

        # run one particular test
        def passed_test(dtype, x_is_row, y_is_row, n=None):

            # set random value for n if none is passed in
            if n is None:
                n = random.randint(2, 1e6)

            # create the vectors to test
            x = random_vector(n, x_is_row, dtype, as_matrix)
            y = random_vector(n, y_is_row, dtype, as_matrix)
            assert x.dtype == y.dtype

            # compare BLASpy result to expected result
            expected = np_dot(x if x_is_row else transpose(x),
                              transpose(y) if y_is_row else y)[0][0]
            actual = dot(x, y)
            return abs(actual - expected) / expected < epsilon

        # two scalars
        test_name = dtype + "_matrix" if as_matrix else "_ndarray" + "_scalars"
        if not passed_test(dtype, ROW, ROW, n=1): tests_failed.append(test_name)

        # two column vectors
        test_name = dtype + "_matrix" if as_matrix else "_ndarray" + "_col_col"
        if not passed_test(dtype, COL, COL): tests_failed.append(test_name)

        # two row vectors
        test_name = dtype + "_matrix" if as_matrix else "_ndarray" + "_row_row"
        if not passed_test(dtype, ROW, ROW): tests_failed.append(test_name)

        # column vector and a row vector
        test_name = dtype + "_matrix" if as_matrix else "_ndarray" + "_col_row"
        if not passed_test(dtype, COL, ROW): tests_failed.append(test_name)

        # two row vectors
        test_name = dtype + "_matrix" if as_matrix else "_ndarray" + "_row_col"
        if not passed_test(dtype, ROW, COL): tests_failed.append(test_name)

    # Test ddot as ndarray
    run_tests('float64', NDARRAY)

    # Test ddot as matrix
    run_tests('float64', MATRIX)

    #Test sdot as ndarray
    run_tests('float32', NDARRAY)

    #Test sdot as matrix
    run_tests('float32', MATRIX)

    # TODO: Add tests with different strides

    return tests_failed