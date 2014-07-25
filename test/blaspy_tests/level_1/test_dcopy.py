"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import dcopy, Vec
import numpy as np
import random

def test_dcopy():

    random.seed()

    tests_failed = []
    test_num = 0

    # Test 1 - two row vectors
    test_num += 1
    n = random.randint(2, 1e5)
    x = np.random.uniform(-1e5, 1e5, (1, n))
    y = np.zeros((1, n))
    dcopy(n, np.ctypeslib.as_ctypes(x), 1, np.ctypeslib.as_ctypes(y), 1, Vec.ROW_ROW)
    passed = np.allclose(x, y)
    if not passed: tests_failed.append(test_num)

    # Test 2 - a row vector and a column vector
    test_num += 1
    n = random.randint(2, 1e5)
    x = np.random.uniform(-1e5, 1e5, (1, n))
    y = np.zeros((n, 1))
    dcopy(n, np.ctypeslib.as_ctypes(x), 1, np.ctypeslib.as_ctypes(y), 1, Vec.ROW_COL)
    passed = np.allclose(np.transpose(x), y)
    if not passed: tests_failed.append(test_num)

    # Test 3 - a column vector and a row vector
    test_num += 1
    n = random.randint(1e4, 1e5)
    x = np.random.uniform(-1e5, 1e5, (n, 1))
    y = np.zeros((1, n))
    dcopy(n, np.ctypeslib.as_ctypes(x), 1, np.ctypeslib.as_ctypes(y), 1, Vec.COL_ROW)
    passed = np.allclose(np.transpose(x), y)
    if not passed: tests_failed.append(test_num)

    # Test 4 - two column vectors
    test_num += 1
    n = random.randint(1e4, 1e5)
    x = np.random.uniform(-1e5, 1e5, (n, 1))
    y = np.zeros((n, 1))
    dcopy(n, np.ctypeslib.as_ctypes(x), 1, np.ctypeslib.as_ctypes(y), 1, Vec.COL_COL)
    passed = np.allclose(x, y)
    if not passed: tests_failed.append(test_num)

    return tests_failed