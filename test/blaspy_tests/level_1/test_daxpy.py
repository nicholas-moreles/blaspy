"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import daxpy, ROW_ROW, ROW_COL, COL_COL, COL_ROW
import numpy as np
import random

def test_daxpy():

    random.seed()
    tests_failed = []
    test_num = 0

    # Test 1 - two row vectors and a positive scalar
    test_num += 1
    n = random.randint(2, 100)
    alpha = random.uniform(2, 1e6)
    x1 = np.random.random((1, n))
    y1 = np.random.random((1, n))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, ROW_ROW)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 2 - two row vectors and a negative scalar
    test_num += 1
    n = random.randint(2, 100)
    alpha = random.uniform(-1e6, -2)
    x1 = np.random.uniform(-100, 100, (1, n))
    y1 = np.random.uniform(-100, 100, (1, n))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, ROW_ROW)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 3 - two column vectors and a positive scalar
    test_num += 1
    n = random.randint(2, 1000)
    alpha = random.uniform(2, 1e6)
    x1 = np.random.uniform(-100, 100, (n, 1))
    y1 = np.random.uniform(-100, 100, (n, 1))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, COL_COL)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 4 - two column vectors and a negative scalar
    test_num += 1
    n = random.randint(2, 1000)
    alpha = random.uniform(-1e6, -2)
    x1 = np.random.uniform(-100, 100, (n, 1))
    y1 = np.random.uniform(-100, 100, (n, 1))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, COL_COL)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 5 - two row vectors and a zero-valued scalar
    test_num += 1
    n = random.randint(2, 1000)
    alpha = 0
    x1 = np.random.uniform(-100, 100, (1, n))
    y1 = np.random.uniform(-100, 100, (1, n))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, ROW_ROW)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 6 - two column vectors and a zero-valued scalar
    test_num += 1
    n = random.randint(2, 1000)
    alpha = 0
    x1 = np.random.uniform(-100, 100, (n, 1))
    y1 = np.random.uniform(-100, 100, (n, 1))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, COL_COL)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 7 - two row vectors and a one-valued scalar
    test_num += 1
    n = random.randint(2, 1000)
    alpha = 1
    x1 = np.random.uniform(-100, 100, (1, n))
    y1 = np.random.uniform(-100, 100, (1, n))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, ROW_ROW)
    y2 += x1
    ppassed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 8 - two column vectors and a one-valued scalar
    test_num += 1
    n = random.randint(2, 1000)
    alpha = 1
    x1 = np.random.uniform(-100, 100, (n, 1))
    y1 = np.random.uniform(-100, 100, (n, 1))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, COL_COL)
    y2 += x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 9 - two row vectors with y being the zero vector and alpha being a positive scalar
    test_num += 1
    n = random.randint(2, 1000)
    alpha = random.uniform(2, 1e6)
    x1 = np.random.random((1, n))
    y1 = np.zeros((1, n))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, ROW_ROW)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 10 - two row vectors with y being the zero vector and alpha being a negative scalar
    test_num += 1
    n = random.randint(2, 1000)
    alpha = random.uniform(-1e6, -2)
    x1 = np.random.random((1, n))
    y1 = np.zeros((1, n))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, ROW_ROW)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 11 - two row vectors with y being the zero vector and alpha being the scalar zero
    test_num += 1
    n = random.randint(2, 1000)
    alpha = 0
    x1 = np.random.random((1, n))
    y1 = np.zeros((1, n))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, ROW_ROW)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 12 - two column vectors with y being the zero vector and alpha being a positive scalar
    test_num += 1
    n = random.randint(2, 1000)
    alpha = random.uniform(2, 1e6)
    x1 = np.random.random((n, 1))
    y1 = np.zeros((n, 1))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, COL_COL)
    y2 += alpha * x1
    ppassed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 13 - two column vectors with y being the zero vector and alpha being a negative scalar
    test_num += 1
    n = random.randint(2, 1000)
    alpha = random.uniform(2, 1e6)
    x1 = np.random.random((n, 1))
    y1 = np.zeros((n, 1))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, COL_COL)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    # Test 14 - two column vectors with y being the zero vector and alpha being the scalar zero
    test_num += 1
    n = random.randint(2, 1000)
    alpha = 0
    x1 = np.random.random((n, 1))
    y1 = np.zeros((n, 1))
    y2 = np.copy(y1)
    daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), 1, np.ctypeslib.as_ctypes(y1), 1, COL_COL)
    y2 += alpha * x1
    passed = np.allclose(y1, y2)
    if not passed: tests_failed.append(test_num)

    return tests_failed