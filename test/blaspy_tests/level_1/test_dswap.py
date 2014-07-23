"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import dswap, ROW_ROW, ROW_COL, COL_COL, COL_ROW
import numpy as np
import random

def test_dswap():

    random.seed()
    tests_failed = []
    test_num = 0

    # Test 1 - two row vectors
    test_num += 1
    n = random.randint(2, 1e3)
    x1 = np.random.uniform(-1e5, 1e5, (1, n))
    y1 = np.random.uniform(-1e5, 1e5, (1, n))
    x2 = np.copy(x1)
    y2 = np.copy(y1)
    dswap(n, np.ctypeslib.as_ctypes(x2), 1, np.ctypeslib.as_ctypes(y2), 1, ROW_ROW)
    passed = np.allclose(x1, y2) and np.allclose(x2, y1)
    if not passed: tests_failed.append(test_num)

    # Test 2 - a row vector and a column vector
    test_num += 1
    n = random.randint(2, 1e3)
    x1 = np.random.uniform(-1e5, 1e5, (1, n))
    y1 = np.random.uniform(-1e5, 1e5, (n, 1))
    x2 = np.copy(x1)
    y2 = np.copy(y1)
    dswap(n, np.ctypeslib.as_ctypes(x2), 1, np.ctypeslib.as_ctypes(y2), 1, ROW_COL)
    passed = np.allclose(np.transpose(x1), y2) and np.allclose(np.transpose(x2), y1)
    if not passed: tests_failed.append(test_num)

    # Test 3 - a column vector and a row vector
    test_num += 1
    n = random.randint(1e3, 1e4)
    x1 = np.random.uniform(-1e5, 1e5, (n, 1))
    y1 = np.random.uniform(-1e5, 1e5, (1, n))
    x2 = np.copy(x1)
    y2 = np.copy(y1)
    dswap(n, np.ctypeslib.as_ctypes(x2), 1, np.ctypeslib.as_ctypes(y2), 1, COL_ROW)
    passed = np.allclose(np.transpose(x1), y2) and np.allclose(np.transpose(x2), y1)
    if not passed: tests_failed.append(test_num)

    # Test 4 - two column vectors
    test_num += 1
    n = random.randint(1e3, 1e4)
    x1 = np.random.uniform(-1e5, 1e5, (n, 1))
    y1 = np.random.uniform(-1e5, 1e5, (n, 1))
    x2 = np.copy(x1)
    y2 = np.copy(y1)
    dswap(n, np.ctypeslib.as_ctypes(x2), 1, np.ctypeslib.as_ctypes(y2), 1, COL_COL)
    passed = np.allclose(x1, y2) and np.allclose(x2, y1)
    if not passed: tests_failed.append(test_num)

    return tests_failed