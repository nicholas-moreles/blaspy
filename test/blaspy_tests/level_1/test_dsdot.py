"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import dsdot, ROW_ROW, ROW_COL, COL_COL, COL_ROW
import numpy as np
import random

def test_dsdot():

    random.seed()
    tests_failed = []
    test_num = 0

    epsilon = 0.0001  # account for precision error

    # Test 1 - two row vectors
    test_num += 1
    n = random.randint(2, 1e5)
    mat1 = np.random.uniform(-1e4, 1e4, (1, n))
    mat2 = np.random.uniform(-1e4, 1e4, (1, n))
    mat3 = np.float32(mat1)
    mat4 = np.float32(mat2)
    actual = dsdot(n, np.ctypeslib.as_ctypes(mat3), 1, np.ctypeslib.as_ctypes(mat4), 1,
                      ROW_ROW)
    expected = np.dot(mat1, np.transpose(mat2))[0][0]
    passed = abs(actual - expected) / actual < epsilon
    if not passed: tests_failed.append(test_num)

    # Test 2 - two column vectors
    test_num += 1
    n = random.randint(2, 1e5)
    mat1 = np.random.uniform(-1e4, 1e4, (n, 1))
    mat2 = np.random.uniform(-1e4, 1e4, (n, 1))
    mat3 = np.float32(mat1)
    mat4 = np.float32(mat2)
    actual = dsdot(n, np.ctypeslib.as_ctypes(mat3), 1, np.ctypeslib.as_ctypes(mat4), 1,
                      COL_COL)
    expected = np.dot(np.transpose(mat1), mat2)[0][0]
    passed = abs(actual - expected) / actual < epsilon
    if not passed: tests_failed.append(test_num)

    # Test 3 - row and column vector
    test_num += 1
    n = random.randint(1e4, 1e5)
    mat1 = np.random.random((1, n))
    mat2 = np.random.random((n, 1))
    mat3 = np.float32(mat1)
    mat4 = np.float32(mat2)
    actual = dsdot(n, np.ctypeslib.as_ctypes(mat3), 1, np.ctypeslib.as_ctypes(mat4), 1,
                      ROW_COL)
    expected = np.dot(mat1, mat2)[0][0]
    passed = abs(actual - expected) / actual < epsilon
    if not passed: tests_failed.append(test_num)


    # Test 4 - column and row vector
    test_num += 1
    n = random.randint(1e4, 1e5)
    mat1 = np.random.uniform(-1e4, 1e4, (n, 1))
    mat2 = np.random.uniform(-1e4, 1e4, (1, n))
    mat3 = np.float32(mat1)
    mat4 = np.float32(mat2)
    actual = dsdot(n, np.ctypeslib.as_ctypes(mat3), 1, np.ctypeslib.as_ctypes(mat4), 1,
                      COL_ROW)
    expected = np.dot(np.transpose(mat1), np.transpose(mat2))[0][0]
    passed = abs(actual - expected) / actual < epsilon
    if not passed: tests_failed.append(test_num)

    # Test 5 - 2 scalars (as column vectors)
    test_num += 1
    n = 1
    mat1 = np.random.uniform(-1e4, 1e4, (n, n))
    mat2 = np.random.uniform(-1e4, 1e4, (n, n))
    mat3 = np.float32(mat1)
    mat4 = np.float32(mat2)
    actual = dsdot(n, np.ctypeslib.as_ctypes(mat3), 1, np.ctypeslib.as_ctypes(mat4), 1,
                      COL_COL)
    expected = np.dot(mat1, mat2)[0][0]
    passed = abs(actual - expected) / actual < epsilon
    if not passed: tests_failed.append(test_num)

    # Test 6 - 2 scalars again (as row vectors)
    test_num += 1
    n = 1
    mat1 = np.random.uniform(-1e4, 1e4, (n, n))
    mat2 = np.random.uniform(-1e4, 1e4, (n, n))
    mat3 = np.float32(mat1)
    mat4 = np.float32(mat2)
    actual = dsdot(n, np.ctypeslib.as_ctypes(mat3), 1, np.ctypeslib.as_ctypes(mat4), 1,
                      ROW_ROW)
    expected = np.dot(mat1, mat2)[0][0]
    passed = abs(actual - expected) / actual < epsilon
    if not passed: tests_failed.append(test_num)

    return tests_failed