"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import axpy
from ..helpers import random_vector, COL, ROW, NDARRAY, MATRIX
from numpy import allclose, transpose
import random

def test_axpy():

    random.seed()
    tests_failed = []

    # run tests of the given type
    def run_tests(dtype, as_matrix):

        # run one particular test
        def passed_test(dtype, x_is_row, y_is_row, n=None, alpha=None):

            # set random value for n and alpha if none are passed in
            if n is None:
                n = random.randint(2, 1e6)
            if alpha is None:
                alpha = random.randint(-100, 100)

            # create the vectors to test
            x = random_vector(n, x_is_row, dtype, as_matrix)
            y = random_vector(n, y_is_row, dtype, as_matrix)
            assert x.dtype == y.dtype

            # compare BLASpy result to expected result
            expected = alpha * (transpose(x) if x_is_row else x)\
                       + (transpose(y) if y_is_row else y)
            axpy(alpha, x, y)
            return allclose(y, (transpose(expected) if y_is_row else expected),
                            rtol=1e-04, atol=1e-06)

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

    # Test daxpy as ndarray
    run_tests('float64', NDARRAY)

    # Test daxpy as matrix
    run_tests('float64', MATRIX)

    #Test saxpy as ndarray
    run_tests('float32', NDARRAY)

    #Test saxpy as matrix
    run_tests('float32', MATRIX)

    # TODO: Add tests with different strides

    return tests_failed