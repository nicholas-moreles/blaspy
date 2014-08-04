"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import gemv
from ..helpers import random_vector, random_matrix, COL, ROW, NDARRAY, MATRIX
from numpy import allclose, copy, transpose, dot, zeros
import random


def test_gemv():
    random.seed()
    tests_failed = []

    # run one particular test
    def passed_test(x_is_row, y_is_row, m=None, n=None, alpha=1, beta=1, stride=None,
                    provide_y=True):

        # set random values for m, n, alpha, beta, and stride if none are passed in
        if m is None:
            m = random.randint(2, 1e3)
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
        A = random_matrix(m / stride + (m % stride > 0), n / stride + (n % stride > 0), dtype,
                          as_matrix)
        x = random_vector(m if trans_a == 'trans' else n, x_is_row, dtype, as_matrix)
        if provide_y:
            y = random_vector(n if trans_a == 'trans' else m, y_is_row, dtype, as_matrix)
            assert x.dtype == y.dtype
        else:
            y = None
        assert A.dtype == x.dtype

        # get the expected result
        if stride == 1:
            if y is None:
                if y_is_row:
                    y_2 = zeros((1, n if trans_a == 'trans' else m))
                else:
                    y_2 = zeros((n if trans_a == 'trans' else m, 1))
            else:
                y_2 = y
            expected = \
                beta * (transpose(y_2) if y_is_row else y_2) \
                + alpha * dot((transpose(A) if trans_a == 'trans' else A),
                              (transpose(x) if x_is_row else x))
        else:
            if y_is_row:
                expected = copy(transpose(y))
            else:
                expected = copy(y)
            for i in range(0, n if trans_a == 'trans' else m, stride):
                expected[i, 0] = \
                    beta * expected[i, 0] \
                    + alpha * dot(transpose(A[:, i / stride]) if trans_a == 'trans'
                                  else A[i / stride, :], transpose(x[:, :: stride]) if x_is_row
                                  else x[:: stride, :])

        # compare the actual result to the expected result
        y = gemv(A, x, alpha=alpha, trans_a=trans_a, y=y, beta=beta, inc_x=stride, inc_y=stride)
        return allclose(y, transpose(expected) if y_is_row else expected, rtol=5e-02, atol=5e-04)

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

        # matrix and two row vectors, y not provided
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_row_no_y"
        if not passed_test(ROW, ROW, stride=1, provide_y=False):
            tests_failed.append(test_name)

        # matrix and two column vectors, y not provided
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_col_no_y"
        if not passed_test(COL, COL, stride=1, provide_y=False):
            tests_failed.append(test_name)

        # matrix and two row vectors
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_col"
        if not passed_test(ROW, COL, stride=1):
            tests_failed.append(test_name)

        # matrix and two column vectors with the same random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_col_rand_stride"
        if not passed_test(COL, COL):
            tests_failed.append(test_name)

        # matrix and two row vectors with the same random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_row_rand_stride"
        if not passed_test(ROW, ROW):
            tests_failed.append(test_name)

        # matrix, column vector and a row vector with the same random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_col_row_rand_stride"
        if not passed_test(COL, ROW):
            tests_failed.append(test_name)

        # matrix and two row vectors with the same random stride
        test_name = dtype + ("_matrix" if as_matrix else "_ndarray") + "_row_col_rand_stride"
        if not passed_test(ROW, COL):
            tests_failed.append(test_name)

    # Test dgemv with ndarray (no_trans)
    trans_a = 'no_trans'
    dtype = 'float64'
    as_matrix = NDARRAY
    run_tests()

    # Test dgemv with matrix (no_trans)
    as_matrix = MATRIX
    run_tests()

    # Test sgemv with ndarray (no_trans)
    dtype = 'float32'
    as_matrix = NDARRAY
    run_tests()

    # Test sgemv with matrix (no_trans)
    as_matrix = MATRIX
    run_tests()

    # Test dgemv with ndarray (trans)
    trans_a = 'trans'
    dtype = 'float64'
    as_matrix = NDARRAY
    run_tests()

    # Test dgemv with matrix (trans)
    as_matrix = MATRIX
    run_tests()

    # Test sgemv with ndarray (trans)
    dtype = 'float32'
    as_matrix = NDARRAY
    run_tests()

    # Test sgemv with matrix (trans)
    as_matrix = MATRIX
    run_tests()

    return tests_failed