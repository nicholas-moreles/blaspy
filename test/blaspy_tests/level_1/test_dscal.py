"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import dscal, Vec
import numpy as np
import random

def test_dscal():

    random.seed()
    tests_failed = []
    test_num = 0

    # Test 1 - scale a scalar by a positive int
    test_num += 1
    n = 1
    alpha = 10
    mat1 = np.random.random((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 2 - scale a scalar by a negative int
    test_num += 1
    n = 1
    alpha = -2000
    mat1 = np.random.random((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 3 - scale a scalar by zero (int)
    test_num += 1
    n = 1
    alpha = 0
    mat1 = np.random.random((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 4 - scale a scalar by a positive float
    test_num += 1
    n = 1
    alpha = 5634.3420
    mat1 = np.random.random((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 5 - scale a scalar by a negative float
    test_num += 1
    n = 1
    alpha = -0.212358
    mat1 = np.random.random((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 6 - scale a scalar by zero (float)
    test_num += 1
    n = 1
    alpha = 0.000
    mat1 = np.random.random((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 7 - scale a zero scalar by a positive int
    test_num += 1
    n = 1
    alpha = 123456
    mat1 = np.zeros((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 8 - scale a zero scalar by a negative int
    test_num += 1
    n = 1
    alpha = -1
    mat1 = np.zeros((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 9 - scale a zero scalar by zero (int)
    test_num += 1
    n = 1
    alpha = 0
    mat1 = np.zeros((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 10 - scale a zero scalar by a positive float
    test_num += 1
    n = 1
    alpha = 0.00358
    mat1 = np.zeros((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 11 - scale a zero scalar by a negative float
    test_num += 1
    n = 1
    alpha = -988533.22726
    mat1 = np.zeros((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 12 - scale a zero scalar by zero (float)
    test_num += 1
    n = 1
    alpha = 0.0
    mat1 = np.zeros((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 13 - scale a scalar by a float (as a row vector)
    test_num += 1
    n = 1
    alpha = 2.5486
    mat1 = np.random.random((n, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.ROW)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 14 - scale a column vector by a positive int
    test_num += 1
    n = 350
    alpha = 123456
    mat1 = np.random.random((n, 1))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 15 - scale a column vector by a negative int
    test_num += 1
    n = 8900
    alpha = -450
    mat1 = np.random.random((n, 1))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 16 - scale a column vector by zero (int)
    test_num += 1
    n = 256
    alpha = 0
    mat1 = np.random.random((n, 1))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 17 - scale a column vector by one (float)
    test_num += 1
    n = 857423
    alpha = 1.000000
    mat1 = np.random.random((n, 1))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 18 - scale a column vector by a positive float
    test_num += 1
    n = 2048
    alpha = 5343.2334454
    mat1 = np.random.random((n, 1))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 19 - scale a column vector by a negative float
    test_num += 1
    n = 512
    alpha = -0.0123498
    mat1 = np.random.random((n, 1))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 20 - scale a column vector by zero (float)
    test_num += 1
    n = 999999
    alpha = 0.0
    mat1 = np.random.random((n, 1))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.COL)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 21 - scale a row vector by a positive int
    test_num += 1
    n = random.randint(2, 2e4)
    alpha = random.randint(2, 2e6)
    mat1 = np.random.random((1, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.ROW)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 22 - scale a row vector by a negative int
    test_num += 1
    n = 20000
    alpha = random.randint(-2e6, -2)
    mat1 = np.random.random((1, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.ROW)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 23 - scale a row vector by zero (int)
    test_num += 1
    n = random.randint(2, 2e4)
    alpha = 0
    mat1 = np.random.random((1, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.ROW)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 24 - scale a row vector by one (float)
    test_num += 1
    n = random.randint(2, 2e4)
    alpha = 1.000000
    mat1 = np.random.random((1, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.ROW)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 25 - scale a row vector by a positive float
    test_num += 1
    n = random.randint(2, 2e4)
    alpha = random.uniform(2, 2e6)
    mat1 = np.random.random((1, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.ROW)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 26 - scale a row vector by a negative float
    test_num += 1
    n = random.randint(2, 2e4)
    alpha = random.uniform(-2e6, -2)
    mat1 = np.random.random((1, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.ROW)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 27 - scale a row vector by zero (float)
    test_num += 1
    n = random.randint(2, 2e4)
    alpha = 0.0
    mat1 = np.random.random((1, n))
    mat2 = np.copy(mat1)
    dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), 1, Vec.ROW)
    mat2 *= alpha
    passed = np.allclose(mat1, mat2)
    if not passed: tests_failed.append(test_num)

    return tests_failed