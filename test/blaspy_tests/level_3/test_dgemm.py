"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import dgemm, Order, Trans
import numpy as np
import random


def test_dgemm():

    random.seed()
    tests_failed = []
    test_num = 0

    # Test 1 - NO_TRANS for A, NO_TRANS for B
    test_num += 1
    m = random.randint(2, 2000)
    n = random.randint(2, 2000)
    k = random.randint(2, 2000)
    alpha = random.randint(-5, 5)
    beta = random.randint(-5, 5)
    A = np.random.uniform(-10, 10, (m, k))
    B = np.random.uniform(-10, 10, (k, n))
    C = np.random.uniform(-1e3, 1e3, (m, n))
    expected = beta * C + alpha * np.dot(A, B)
    dgemm(Order.ROW_MAJOR, Trans.NO_TRANS, Trans.NO_TRANS, m, n, k, alpha,
          np.ctypeslib.as_ctypes(A), k, np.ctypeslib.as_ctypes(B), n, beta,
          np.ctypeslib.as_ctypes(C), n)
    passed = np.allclose(C, expected)
    if not passed: tests_failed.append(test_num)

    # Test 2 - NO_TRANS for A, TRANS for B
    test_num += 1
    m = random.randint(2, 2000)
    n = random.randint(2, 2000)
    k = random.randint(2, 2000)
    alpha = random.randint(-5, 5)
    beta = random.randint(-5, 5)
    A = np.random.uniform(-10, 10, (m, k))
    B = np.random.uniform(-10, 10, (n, k))
    C = np.random.uniform(-1e3, 1e3, (m, n))
    expected = beta * C + alpha * np.dot(A, np.transpose(B))
    dgemm(Order.ROW_MAJOR, Trans.NO_TRANS, Trans.TRANS, m, n, k, alpha,
          np.ctypeslib.as_ctypes(A), k, np.ctypeslib.as_ctypes(B), k, beta,
          np.ctypeslib.as_ctypes(C), n)
    passed = np.allclose(C, expected)
    if not passed: tests_failed.append(test_num)

    # Test 3 - TRANS for A, TRANS for B
    test_num += 1
    m = random.randint(1000, 3000)
    n = random.randint(1000, 3000)
    k = random.randint(1000, 3000)
    alpha = random.randint(-5, 5)
    beta = random.randint(-5, 5)
    A = np.random.uniform(-10, 10, (k, m))
    B = np.random.uniform(-10, 10, (n, k))
    C = np.random.uniform(-1e3, 1e3, (m, n))
    expected = beta * C + alpha * np.dot(np.transpose(A), np.transpose(B))
    dgemm(Order.ROW_MAJOR, Trans.TRANS, Trans.TRANS, m, n, k, alpha,
          np.ctypeslib.as_ctypes(A), m, np.ctypeslib.as_ctypes(B), k, beta,
          np.ctypeslib.as_ctypes(C), n)
    passed = np.allclose(C, expected)
    if not passed: tests_failed.append(test_num)

    # Test 4 - TRANS for A, NO_TRANS for B
    test_num += 1
    m = random.randint(1000, 3000)
    n = random.randint(1000, 3000)
    k = random.randint(1000, 3000)
    alpha = random.randint(-5, 5)
    beta = random.randint(-5, 5)
    A = np.random.uniform(-10, 10, (k, m))
    B = np.random.uniform(-10, 10, (k, n))
    C = np.random.uniform(-1e3, 1e3, (m, n))
    expected = beta * C + alpha * np.dot(np.transpose(A), B)
    dgemm(Order.ROW_MAJOR, Trans.TRANS, Trans.NO_TRANS, m, n, k, alpha,
          np.ctypeslib.as_ctypes(A), m, np.ctypeslib.as_ctypes(B), n, beta,
          np.ctypeslib.as_ctypes(C), n)
    passed = np.allclose(C, expected)
    if not passed: tests_failed.append(test_num)

    return tests_failed