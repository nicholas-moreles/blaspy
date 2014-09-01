"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_matrix
from blaspy import gemm
from numpy import copy, dot
from itertools import product
from random import uniform
import time

TRIALS = 20
SCAL_MIN, SCAL_MAX = -10, 10    # scalar values


def timing_gemm():
    """
    Test general matrix-matrix multiplication.

    Returns:
        A list of strings representing the failed tests.
    """
    # values to test
    vals = (100, 300, 500, 1000, 1500, 2000, 2500, 3000)
    dtypes = ('float64', 'float32')
    trans_tuple = ('n', 't')
    bp_total = 0.0
    np_total = 0.0

    for (k, n) in product(vals, vals, vals):
        # test all combinations of all possible values
        for (dtype, trans_a, trans_b) in product(dtypes, trans_tuple, trans_tuple):
            variables = ("gemm_", trans_a, "_", trans_b)
            test_name = "".join(variables)
            bp_time, np_time = timing_test(dtype, trans_a, trans_b, n, k)
            bp_total += bp_time
            np_total += np_time

        print("\nk: %d, n: %d, BLASpy Average: %.5fs, NumPy Average: %.5f"
              % (k, n, bp_total / 8, np_total / 8))

def timing_test(dtype, trans_a, trans_b, n, k):
    """
    Run one general matrix-matrix multiplication test.

    Arguments:
        dtype:        either 'float64' or 'float32', the NumPy dtype to test
        as_matrix:    True to test a NumPy matrix, False to test a NumPy ndarray
        provide_C:    True if C is to be provided to the BLASpy function, False otherwise
        trans_a:      BLASpy trans_a parameter to test
        trans_b:      BLASpy trans_b parameter to test

    Returns:
        True if the expected result is within the margin of error of the actual result,
        False otherwise.
    """
    as_matrix = True

    np_time = 0.0
    bp_time = 0.0

    for i in range(TRIALS):

        # create random scalars and matrices to test
        alpha = uniform(SCAL_MIN, SCAL_MAX)
        beta = uniform(SCAL_MIN, SCAL_MAX)
        A = random_matrix((n if trans_a == 'n' else k), (k if trans_a == 'n' else n), dtype, as_matrix)
        B = random_matrix((k if trans_b == 'n' else n), (n if trans_b == 'n' else k), dtype, as_matrix)
        C = random_matrix(n, n, dtype, as_matrix)

        # create copies/views for NumPy
        A_2 = A if trans_a == 'n' else A.T
        B_2 = B if trans_b == 'n' else B.T
        C_2 = copy(C)

        if i % 2 == 0:

            # BLASpy first
            start = time.time()
            gemm(A, B, C, trans_a, trans_b, alpha, beta)
            end = time.time()
            bp_time += end - start

            # then NumPy
            start = time.time()
            beta * C_2 + alpha * dot(A_2, B_2)
            end = time.time()
            np_time += end - start

        else:

            # NumPy first
            start = time.time()
            beta * C_2 + alpha * dot(A_2, B_2)
            end = time.time()
            np_time += end - start

            # then BLASpy
            start = time.time()
            gemm(A, B, C, trans_a, trans_b, alpha, beta)
            end = time.time()
            bp_time += end - start

    return bp_time / TRIALS, np_time / TRIALS