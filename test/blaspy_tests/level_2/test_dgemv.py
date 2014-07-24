"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import dgemv, Vec, Order, Trans
import numpy as np
import random


def test_dgemv():

    random.seed()
    tests_failed = []
    test_num = 0

    # Test 1 - matrix and two column vectors, no transpose
    test_num += 1
    m = random.randint(2, 2e3)
    n = random.randint(2, 2e3)
    alpha = random.randint(-5, 5)
    beta = random.randint(-5, 5)
    mat = np.random.uniform(-10, 10, (m, n))
    x = np.random.uniform(-100, 100, (n, 1))
    y = np.random.uniform(-1e3, 1e3, (m, 1))
    y2 = beta * y + alpha * np.dot(mat, x)
    dgemv(Order.ROW_MAJOR, Trans.NO_TRANS, m, n, alpha, np.ctypeslib.as_ctypes(mat), n,
          np.ctypeslib.as_ctypes(x), 1, beta, np.ctypeslib.as_ctypes(y), 1, Vec.COL_COL)
    passed = np.allclose(y, y2)
    if not passed: tests_failed.append(test_num)

    # Test 2 - matrix and two row vectors, no transpose
    test_num += 1
    m = random.randint(2, 2e3)
    n = random.randint(2, 2e3)
    alpha = 1
    beta = 1
    mat = np.random.uniform(-10, 10, (m, n))
    x = np.random.uniform(-100, 100, (1, n))
    y = np.random.uniform(-10, 10, (1, m))
    y2 = beta * np.transpose(y) + alpha * np.dot(mat, np.transpose(x))
    dgemv(Order.ROW_MAJOR, Trans.NO_TRANS, m, n, alpha, np.ctypeslib.as_ctypes(mat), n,
          np.ctypeslib.as_ctypes(x), 1, beta, np.ctypeslib.as_ctypes(y), 1, Vec.ROW_ROW)
    passed = np.allclose(np.transpose(y), y2)
    if not passed: tests_failed.append(test_num)

    # Test 3 - matrix, x is a row vector , y is a col vector, no transpose
    test_num += 1
    m = random.randint(1e3, 2e3)
    n = random.randint(1e3, 2e3)
    alpha = 1
    beta = 1
    mat = np.random.uniform(-10, 10, (m, n))
    x = np.random.uniform(-100, 100, (1, n))
    y = np.random.uniform(-10, 10, (m, 1))
    y2 = beta * y + alpha * np.dot(mat, np.transpose(x))
    dgemv(Order.ROW_MAJOR, Trans.NO_TRANS, m, n, alpha, np.ctypeslib.as_ctypes(mat), n,
          np.ctypeslib.as_ctypes(x), 1, beta, np.ctypeslib.as_ctypes(y), 1, Vec.ROW_COL)
    passed = np.allclose(y, y2)
    if not passed: tests_failed.append(test_num)

    # Test 4 - matrix, x is a col vector, y is a row vector, no transpose
    test_num += 1
    m = random.randint(1e3, 2e3)
    n = random.randint(1e3, 2e3)
    alpha = 1
    beta = 1
    mat = np.random.uniform(-10, 10, (m, n))
    x = np.random.uniform(-100, 100, (n, 1))
    y = np.random.uniform(-10, 10, (1, m))
    y2 = beta * np.transpose(y) + alpha * np.dot(mat, x)
    dgemv(Order.ROW_MAJOR, Trans.NO_TRANS, m, n, alpha, np.ctypeslib.as_ctypes(mat), n,
          np.ctypeslib.as_ctypes(x), 1, beta, np.ctypeslib.as_ctypes(y), 1, Vec.COL_ROW)
    passed = np.allclose(np.transpose(y), y2)
    if not passed: tests_failed.append(test_num)

    # Test 5 - matrix and two column vectors, transpose
    test_num += 1
    m = random.randint(2, 2e3)
    n = random.randint(2, 2e3)
    alpha = random.randint(-5, 5)
    beta = random.randint(-5, 5)
    mat = np.random.uniform(-10, 10, (m, n))
    x = np.random.uniform(-100, 100, (m, 1))
    y = np.random.uniform(-1e3, 1e3, (n, 1))
    y2 = beta * y + alpha * np.dot(np.transpose(mat), x)
    dgemv(Order.ROW_MAJOR, Trans.TRANS, m, n, alpha, np.ctypeslib.as_ctypes(mat), n,
          np.ctypeslib.as_ctypes(x), 1, beta, np.ctypeslib.as_ctypes(y), 1, Vec.COL_COL)
    passed = np.allclose(y, y2)
    if not passed: tests_failed.append(test_num)

    # Test 6 - matrix and two row vectors, transpose
    test_num += 1
    m = random.randint(2, 2e3)
    n = random.randint(2, 2e3)
    alpha = 1
    beta = 1
    mat = np.random.uniform(-10, 10, (m, n))
    x = np.random.uniform(-100, 100, (1, m))
    y = np.random.uniform(-10, 10, (1, n))
    y2 = beta * np.transpose(y) + alpha * np.dot(np.transpose(mat), np.transpose(x))
    dgemv(Order.ROW_MAJOR, Trans.TRANS, m, n, alpha, np.ctypeslib.as_ctypes(mat), n,
          np.ctypeslib.as_ctypes(x), 1, beta, np.ctypeslib.as_ctypes(y), 1, Vec.ROW_ROW)
    passed = np.allclose(np.transpose(y), y2)
    if not passed: tests_failed.append(test_num)

    # Test 7 - matrix, x is a row vector , y is a col vector, transpose
    test_num += 1
    m = random.randint(1e3, 2e3)
    n = random.randint(1e3, 2e3)
    alpha = 1
    beta = 1
    mat = np.random.uniform(-10, 10, (m, n))
    x = np.random.uniform(-100, 100, (1, m))
    y = np.random.uniform(-10, 10, (n, 1))
    y2 = beta * y + alpha * np.dot(np.transpose(mat), np.transpose(x))
    dgemv(Order.ROW_MAJOR, Trans.TRANS, m, n, alpha, np.ctypeslib.as_ctypes(mat), n,
          np.ctypeslib.as_ctypes(x), 1, beta, np.ctypeslib.as_ctypes(y), 1, Vec.ROW_COL)
    passed = np.allclose(y, y2)
    if not passed: tests_failed.append(test_num)

    # Test 8 - matrix, x is a col vector, y is a row vector, transpose
    test_num += 1
    m = random.randint(1e3, 2e3)
    n = random.randint(1e3, 2e3)
    alpha = 1
    beta = 1
    mat = np.random.uniform(-10, 10, (m, n))
    x = np.random.uniform(-100, 100, (m, 1))
    y = np.random.uniform(-10, 10, (1, n))
    y2 = beta * np.transpose(y) + alpha * np.dot(np.transpose(mat), x)
    dgemv(Order.ROW_MAJOR, Trans.TRANS, m, n, alpha, np.ctypeslib.as_ctypes(mat), n,
          np.ctypeslib.as_ctypes(x), 1, beta, np.ctypeslib.as_ctypes(y), 1, Vec.COL_ROW)
    passed = np.allclose(np.transpose(y), y2)
    if not passed: tests_failed.append(test_num)

    return tests_failed