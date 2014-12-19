"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ..helpers import random_matrix, random_symmetric_matrix
from blaspy import syrk
from numpy import allclose, copy, dot, tril, triu, zeros
from itertools import product
from random import randint, uniform

N_MIN, N_MAX = 2, 1e3           # matrix/vector sizes
SCAL_MIN, SCAL_MAX = -10, 10    # scalar values
STRIDE_MAX = 1e2                # max vector stride
RTOL, ATOL = 5e-01, 5e-02       # margin of error


def acceptance_test_syrk():
    """
    Test symmetric rank-k update.

    Returns:
        A list of strings representing the failed tests.
    """

    tests_failed = []

    # values to test
    dtypes = ('float64', 'float32')
    bools = (True, False)
    uplos = ('u', 'l')
    trans_tuple = ('n', 't')

    # test all combinations of all possible values
    for (dtype, as_matrix, provide_C, uplo, trans) \
            in product(dtypes, bools, bools, uplos, trans_tuple):

        # if a test fails, create a string representation of its name and append it to the list
        # of failed tests
        if not passed_test(dtype, as_matrix, provide_C, uplo, trans):
            variables = (dtype,
                         "_matrix" if as_matrix else "_ndarray",
                         "_" if provide_C else "_no_C_",
                         uplo, "_",
                         trans)
            test_name = "".join(variables)
            tests_failed.append(test_name)

    return tests_failed

def passed_test(dtype, as_matrix, provide_C, uplo, trans):
    """
    Run one symmetric rank-k update test.

    Arguments:
        dtype:        either 'float64' or 'float32', the NumPy dtype to test
        as_matrix:    True to test a NumPy matrix, False to test a NumPy ndarray
        provide_C:    True if C is to be provided to the BLASpy function, False otherwise
        uplo:         BLASpy 'uplo' parameter to test
        trans:        BLASpy 'trans_a' parameter to test

    Returns:
        True if the expected result is within the margin of error of the actual result,
        False otherwise.
    """

    transpose_a = trans == 't' or trans == 'T'
    upper = uplo == 'u' or uplo == 'U'

    # generate random sizes for matrix dimensions
    m_A = randint(N_MIN, N_MAX)
    n_A = randint(N_MIN, N_MAX)
    n = m_A if not transpose_a else n_A

    # create random scalars and matrices to test
    alpha = uniform(SCAL_MIN, SCAL_MAX)
    beta = uniform(SCAL_MIN, SCAL_MAX)
    A = random_matrix(m_A, n_A, dtype, as_matrix)
    C = random_symmetric_matrix(n, dtype, as_matrix) if provide_C else None

    # create a copy of  C that can be used to calculate the expected result
    C_2 = copy(C) if C is not None else zeros((n, n))

    # compute the expected result
    C_2 = beta * C_2 + alpha * (dot(A, A.T) if not transpose_a else dot(A.T, A))

        # ensure C and C_2 are upper or lower triangular representations of symmetric matrices
    if upper:
        C_2 = triu(C_2)
        if provide_C:
            C = triu(C)
    else:
        C_2 = tril(C_2)
        if provide_C:
            C = tril(C)

    # get the actual result
    C = syrk(A, C, uplo, trans, alpha, beta)

    # compare the actual result to the expected result and return result of the test
    return allclose(C, C_2, RTOL, ATOL)