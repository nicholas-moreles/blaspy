"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import symv
from ..helpers import random_vector, random_square_matrix, COL, ROW, NDARRAY, MATRIX
from numpy import allclose, copy, dot, zeros, triu, tril
from itertools import product
import random


def test_symv():
    random.seed()
    tests_failed = []

    # run one particular test
    # noinspection PyPep8Naming
    def passed_test(x_is_row, y_is_row, n=None, alpha=1, beta=1, stride=None, provide_y=True):

        # set random values for n, alpha, beta, and stride if none are passed in
        if n is None:
            n = random.randint(2, 1e3)
        if alpha is None:
            alpha = random.randint(-100, 100)
        if beta is None:
            beta = random.randint(-100, 100)
        if stride is None:
            stride = random.randint(2, 1e2)

        # create the matrix and vectors to test
        # noinspection PyPep8Naming
        A = random_square_matrix(n / stride + (n % stride > 0), dtype, as_matrix)
        x = random_vector(n, x_is_row, dtype, as_matrix)
        if provide_y:
            y = random_vector(n, y_is_row, dtype, as_matrix)
            assert x.dtype == y.dtype
        else:
            y = None
        assert A.dtype == x.dtype

        # get the expected result
        if stride == 1:
            if y is None:
                if y_is_row:
                    y_2 = zeros((1, n))
                else:
                    y_2 = zeros((n, 1))
            else:
                y_2 = y
            expected = beta * (y_2.T if y_is_row else y_2) + alpha * dot(A, x.T if x_is_row else x)
        else:
            if y_is_row:
                expected = copy(y.T)
            else:
                expected = copy(y)
            for i in range(0, n, stride):
                expected[i, 0] = \
                    beta * expected[i, 0] \
                    + alpha * dot(A[i / stride, :],
                                  x[:, :: stride].T if x_is_row else x[:: stride, :])

        # make A upper or lower triangular
        if uplo == 'u':
            A = triu(A)
        else:
            A = tril(A)

        # compare the actual result to the expected result
        y = symv(A, x, y, uplo, alpha, beta, inc_x=stride, inc_y=stride)
        return allclose(y, expected.T if y_is_row else expected, rtol=5e-02, atol=5e-04)

    # run all tests of the given type
    def run_tests():

        # three scalars
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_scalars_" + uplo
        if not passed_test(ROW, ROW, n=1, stride=1):
            tests_failed.append(test_name)

        # matrix and two column vectors
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_col_" + uplo
        if not passed_test(COL, COL, stride=1):
            tests_failed.append(test_name)

        # matrix and two row vectors
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_row_" + uplo
        if not passed_test(ROW, ROW, stride=1):
            tests_failed.append(test_name)

        # matrix, column vector and a row vector
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_row_" + uplo
        if not passed_test(COL, ROW, stride=1):
            tests_failed.append(test_name)

        # matrix and two row vectors, y not provided
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_row_no_y_" + uplo
        if not passed_test(ROW, ROW, stride=1, provide_y=False):
            tests_failed.append(test_name)

        # matrix and two column vectors, y not provided
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_col_no_y_" + uplo
        if not passed_test(COL, COL, stride=1, provide_y=False):
            tests_failed.append(test_name)

        # matrix and two row vectors
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_col_" + uplo
        if not passed_test(ROW, COL, stride=1):
            tests_failed.append(test_name)

        # matrix and two column vectors with the same random stride
        test_name = \
            dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_col_rand_stride_" + uplo
        if not passed_test(COL, COL):
            tests_failed.append(test_name)

        # matrix and two row vectors with the same random stride
        test_name = \
            dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_row_rand_stride_" + uplo
        if not passed_test(ROW, ROW):
            tests_failed.append(test_name)

        # matrix, column vector and a row vector with the same random stride
        test_name = \
            dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_row_rand_stride_" + uplo
        if not passed_test(COL, ROW):
            tests_failed.append(test_name)

        # matrix and two row vectors with the same random stride
        test_name = \
            dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_col_rand_stride_" + uplo
        if not passed_test(ROW, COL):
            tests_failed.append(test_name)

    # test all combinations of possible values
    uplo_values = ['u', 'l']
    dtype_values = ['float64', 'float32']
    as_matrix_values = [NDARRAY, MATRIX]

    for (uplo, dtype, as_matrix) in product(uplo_values, dtype_values, as_matrix_values):
        run_tests()

    return tests_failed