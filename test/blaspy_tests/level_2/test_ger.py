"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import ger
from ..helpers import random_vector, random_matrix, COL, ROW, NDARRAY, MATRIX
from numpy import allclose, copy, transpose, dot, zeros
import random


def test_ger():
    random.seed()
    tests_failed = []

    # run one particular test
    # noinspection PyPep8Naming
    def passed_test(x_is_row, y_is_row, m=None, n=None, alpha=1, stride=None, provide_A=True):

        # set random values for m, n, alpha, beta, and stride if none are passed in
        if m is None:
            m = random.randint(2, 1e3)
        if n is None:
            n = random.randint(2, 1e3)
        if alpha is None:
            alpha = random.randint(-100, 100)
        if stride is None:
            stride_x = random.randint(2, 1e2)
            stride_y = random.randint(2, 1e2)
        else:
            stride_x = stride
            stride_y = stride

        # create the matrix and vectors to test
        m_A = m / stride_x + (m % stride_x > 0)
        n_A = n / stride_y + (n % stride_y > 0)
        if provide_A:
            A = random_matrix(m_A, n_A, dtype, as_matrix)
        else:
            A = None
        x = random_vector(m, x_is_row, dtype, as_matrix)
        y = random_vector(n, y_is_row, dtype, as_matrix)

        # create variables for use in the expected result
        A_2 = zeros((m_A, n_A), dtype=dtype) if A is None else copy(A)
        x_2 = transpose(x) if x_is_row else x
        y_2 = y if y_is_row else transpose(y)

        # get the expected result
        if stride == 1:
            A_2 += alpha * dot(x_2, y_2)
        else:
            for i in range(0, m_A):
                for j in range(0, n_A):
                    A_2[i, j] += alpha * x_2[i * stride_x, 0] * y_2[0, j * stride_y]

        # compare the actual result to the expected result
        A = ger(x, y, alpha=alpha, A=A, inc_x=stride_x, inc_y=stride_y)
        return allclose(A, A_2)

    # run all tests of the given type
    def run_tests():

        # three scalars
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_scalars"
        if not passed_test(ROW, ROW, m=1, n=1, stride=1):
            tests_failed.append(test_name)

        # matrix and two column vectors
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_col"
        if not passed_test(COL, COL, stride=1):
            tests_failed.append(test_name)

        # matrix and two row vectors
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_row"
        if not passed_test(ROW, ROW, stride=1):
            tests_failed.append(test_name)

        # matrix, column vector and a row vector
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_row"
        if not passed_test(COL, ROW, stride=1):
            tests_failed.append(test_name)

        # matrix and two row vectors, A not provided
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_row_no_A"
        if not passed_test(ROW, ROW, stride=1, provide_A=False):
            tests_failed.append(test_name)

        # matrix and two column vectors, y not provided
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_col_no_A"
        if not passed_test(COL, COL, stride=1, provide_A=False):
            tests_failed.append(test_name)

        # matrix and two row vectors
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_col"
        if not passed_test(ROW, COL, stride=1):
            tests_failed.append(test_name)

        # matrix and two column vectors with independently random strides
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_col_rand_stride"
        if not passed_test(COL, COL):
            tests_failed.append(test_name)

        # matrix and two row vectors with independently random strides
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_row_rand_stride"
        if not passed_test(ROW, ROW):
            tests_failed.append(test_name)

        # matrix, column vector and a row vector with independently random strides
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_row_rand_stride"
        if not passed_test(COL, ROW):
            tests_failed.append(test_name)

        # matrix and two row vectors with independently random strides
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_col_rand_stride"
        if not passed_test(ROW, COL):
            tests_failed.append(test_name)

        # matrix, row vector and a column vector with independently random strides, A not provided
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_col_rand_stride_no_A"
        if not passed_test(ROW, COL, provide_A=False):
            tests_failed.append(test_name)

        # matrix, column vector and a row vector with independently random strides, A not provided
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_row_rand_stride_no_A"
        if not passed_test(COL, ROW, provide_A=False):
            tests_failed.append(test_name)

    # Test dgemv with ndarray
    dtype = 'float64'
    as_matrix = NDARRAY
    run_tests()

    # Test dgemv with matrix
    as_matrix = MATRIX
    run_tests()

    # Test sgemv with ndarray
    dtype = 'float32'
    as_matrix = NDARRAY
    run_tests()

    # Test sgemv with matrix
    as_matrix = MATRIX
    run_tests()

    return tests_failed