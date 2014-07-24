"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import dger, Vec, Order
import numpy as np
import random


def test_dger():

    random.seed()
    tests_failed = []
    test_num = 0

    # Test 1 - matrix and two column vectors
    test_num += 1
    m = random.randint(2, 2e3)
    n = random.randint(2, 2e3)
    alpha = random.randint(-5, 5)
    mat = np.random.uniform(-1e3, 1e3, (m, n))
    x = np.random.uniform(-100, 100, (m, 1))
    y = np.random.uniform(-100, 100, (n, 1))
    mat2 = mat + alpha * np.dot(x, np.transpose(y))
    dger(Order.ROW_MAJOR, m, n, alpha, np.ctypeslib.as_ctypes(x), 1, np.ctypeslib.as_ctypes(y), 1,
          np.ctypeslib.as_ctypes(mat), n, Vec.COL_COL)
    passed = np.allclose(mat, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 2 - matrix and two row vectors
    test_num += 1
    m = random.randint(2, 2e3)
    n = random.randint(2, 2e3)
    alpha = random.randint(-5, 5)
    mat = np.random.uniform(-1e3, 1e3, (m, n))
    x = np.random.uniform(-100, 100, (1, m))
    y = np.random.uniform(-100, 100, (1, n))
    mat2 = mat + alpha * np.dot(np.transpose(x), y)
    dger(Order.ROW_MAJOR, m, n, alpha, np.ctypeslib.as_ctypes(x), 1, np.ctypeslib.as_ctypes(y), 1,
          np.ctypeslib.as_ctypes(mat), n, Vec.ROW_ROW)
    passed = np.allclose(mat, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 3 - matrix, x is a row vector , y is a col vector
    test_num += 1
    m = random.randint(1e3, 2e3)
    n = random.randint(1e3, 2e3)
    alpha = random.randint(-50, 50)
    mat = np.random.uniform(-1e3, 1e3, (m, n))
    x = np.random.uniform(-1e3, 1e3, (1, m))
    y = np.random.uniform(-1e3, 1e3, (n, 1))
    mat2 = mat + alpha * np.dot(np.transpose(x), np.transpose(y))
    dger(Order.ROW_MAJOR, m, n, alpha, np.ctypeslib.as_ctypes(x), 1, np.ctypeslib.as_ctypes(y), 1,
          np.ctypeslib.as_ctypes(mat), n, Vec.ROW_COL)
    passed = np.allclose(mat, mat2)
    if not passed: tests_failed.append(test_num)

    # Test 4 - matrix, x is a col vector, y is a row vector, no transpose
    test_num += 1
    m = random.randint(1e3, 2e3)
    n = random.randint(1e3, 2e3)
    alpha = random.randint(-50, 50)
    mat = np.random.uniform(-1e3, 1e3, (m, n))
    x = np.random.uniform(-1e3, 1e3, (m, 1))
    y = np.random.uniform(-1e3, 1e3, (1, n))
    mat2 = mat + alpha * np.dot(x, y)
    dger(Order.ROW_MAJOR, m, n, alpha, np.ctypeslib.as_ctypes(x), 1, np.ctypeslib.as_ctypes(y), 1,
          np.ctypeslib.as_ctypes(mat), n, Vec.COL_ROW)
    passed = np.allclose(mat, mat2)
    if not passed: tests_failed.append(test_num)

    return tests_failed