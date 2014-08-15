"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_matrix
from blaspy import gemm
from numpy import allclose, copy, dot, zeros
from itertools import product
from random import randint, uniform

N_MIN, N_MAX = 2, 1e3           # matrix/vector sizes
SCAL_MIN, SCAL_MAX = -10, 10    # scalar values
STRIDE_MAX = 1e2                # max vector stride
RTOL, ATOL = 5e-01, 5e-02       # margin of error


def test_gemm():
    """
    Test general matrix-vector multiplication.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ('float64', 'float32')
    bools = (True, False)
    trans_tuple = ('n', 't')

    # test all combinations of all possible values
    for (dtype, as_matrix, provide_C, trans_a, trans_b) \
            in product(dtypes, bools, bools, trans_tuple, trans_tuple):

        # if a test fails, create a string representation of its name and append it to the list
        # of failed tests
        if not passed_test(dtype, as_matrix, provide_C, trans_a, trans_b):
            variables = (dtype,
                         "_matrix" if as_matrix else "_ndarray",
                         "_" if provide_C else "_no_C_",
                         trans_a, "_",
                         trans_b)
            test_name = "".join(variables)
            tests_failed.append(test_name)

    return tests_failed

def passed_test(dtype, as_matrix, provide_C, trans_a, trans_b):
    """
    Run one general matrix-vector multiplication test.

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

    # generate random sizes for matrix dimensions
    m = randint(N_MIN, N_MAX)
    n = randint(N_MIN, N_MAX)
    k = randint(N_MIN, N_MAX)

    # create random scalars and matrices to test
    alpha = uniform(SCAL_MIN, SCAL_MAX)
    beta = uniform(SCAL_MIN, SCAL_MAX)
    A = random_matrix((m if trans_a == 'n' else k), (k if trans_a == 'n' else m), dtype, as_matrix)
    B = random_matrix((k if trans_b == 'n' else n), (n if trans_b == 'n' else k), dtype, as_matrix)
    C = random_matrix(m, n, dtype, as_matrix) if provide_C else None

    # create copies/views of A, B, and C that can be used to calculate the expected result
    A_2 = A if trans_a == 'n' else A.T
    B_2 = B if trans_b == 'n' else B.T
    C_2 = copy(C) if C is not None else zeros((m, n))

    # compute the expected result
    C_2 = beta * C_2 + alpha * dot(A_2, B_2)

    # get the actual result
    C = gemm(A, B, C, trans_a, trans_b, alpha, beta)

    # compare the actual result to the expected result and return result of the test
    return allclose(C, C_2, RTOL, ATOL)