"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

import blaspy as bp
import numpy as np
import random


def test_ddot():

    random.seed()
    tests_failed = []
    test_num = 0
    epsilon = 0.001  # account for round-off/precision error

    # Test 1 - two row vectors
    test_num += 1
    n = random.randint(2, 1e5)
    mat1 = np.random.uniform(-1e4, 1e4, (1, n))
    mat2 = np.random.uniform(-1e4, 1e4, (1, n))
    actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), 1, np.ctypeslib.as_ctypes(mat2), 1,
                     bp.ROW_ROW)
    expected = np.dot(mat1, np.transpose(mat2))[0][0]
    passed = abs(actual - expected) < epsilon
    if not passed: tests_failed.append(test_num)

    # Test 2 - two column vectors
    test_num += 1
    n = random.randint(2, 1e5)
    mat1 = np.random.uniform(-1e4, 1e4, (n, 1))
    mat2 = np.random.uniform(-1e4, 1e4, (n, 1))
    actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), 1, np.ctypeslib.as_ctypes(mat2), 1,
                     bp.COL_COL)
    expected = np.dot(np.transpose(mat1), mat2)[0][0]
    passed = abs(actual - expected) < epsilon
    if not passed: tests_failed.append(test_num)

    # Test 3 - row and column vector
    test_num += 1
    n = random.randint(1e4, 1e5)
    mat1 = np.random.random((1, n))
    mat2 = np.random.random((n, 1))
    actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), 1, np.ctypeslib.as_ctypes(mat2), 1,
                     bp.ROW_COL)
    expected = np.dot(mat1, mat2)[0][0]
    passed = abs(actual - expected) < epsilon
    if not passed: tests_failed.append(test_num)

    # Test 4 - column and row vector
    test_num += 1
    n = random.randint(1e4, 1e5)
    mat1 = np.random.uniform(-1e4, 1e4, (n, 1))
    mat2 = np.random.uniform(-1e4, 1e4, (1, n))
    actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), 1, np.ctypeslib.as_ctypes(mat2), 1,
                     bp.COL_ROW)
    expected = np.dot(np.transpose(mat1), np.transpose(mat2))[0][0]
    passed = abs(actual - expected) < epsilon
    if not passed: tests_failed.append(test_num)

    # Test 5 - 2 scalars (as columns)
    test_num += 1
    n = 1
    mat1 = np.random.uniform(-1e4, 1e4, (n, n))
    mat2 = np.random.uniform(-1e4, 1e4, (n, n))
    actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), 1, np.ctypeslib.as_ctypes(mat2), 1,
                     bp.COL_COL)
    expected = np.dot(mat1, mat2)[0][0]
    passed = abs(actual - expected) < epsilon
    if not passed: tests_failed.append(test_num)

    # Test 6 - 2 scalars again (as rows)
    test_num += 1
    n = 1
    mat1 = np.random.uniform(-1e4, 1e4, (n, n))
    mat2 = np.random.uniform(-1e4, 1e4, (n, n))
    actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), 1, np.ctypeslib.as_ctypes(mat2), 1,
                     bp.ROW_ROW)
    expected = np.dot(mat1, mat2)[0][0]
    passed = abs(actual - expected) < epsilon
    if not passed: tests_failed.append(test_num)

    return tests_failed