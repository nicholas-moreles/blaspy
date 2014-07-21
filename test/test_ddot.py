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

random.seed()

epsilon = 0.001  # account for round-off/precision error

# Test 1 - two row vectors
n = random.randint(2, 1e5)
mat1 = np.random.uniform(-1e4, 1e4, (1, n))
mat2 = np.random.uniform(-1e4, 1e4, (1, n))
actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), False, 1, np.ctypeslib.as_ctypes(mat2), False, 1)
expected = np.dot(mat1, np.transpose(mat2))[0][0]
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 1")

# Test 2 - two column vectors
n = random.randint(2, 1e5)
mat1 = np.random.uniform(-1e4, 1e4, (n, 1))
mat2 = np.random.uniform(-1e4, 1e4, (n, 1))
actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), True, 1, np.ctypeslib.as_ctypes(mat2), True, 1)
expected = np.dot(np.transpose(mat1), mat2)[0][0]
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 2")

# Test 3 - row and column vector
n = random.randint(1e4, 1e5)
mat1 = np.random.random((1, n))
mat2 = np.random.random((n, 1))
actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), False, 1, np.ctypeslib.as_ctypes(mat2), True, 1)
expected = np.dot(mat1, mat2)[0][0]
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 3")


# Test 4 - column and row vector
n = random.randint(1e4, 1e5)
mat1 = np.random.uniform(-1e4, 1e4, (n, 1))
mat2 = np.random.uniform(-1e4, 1e4, (1, n))
actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), True, 1, np.ctypeslib.as_ctypes(mat2), False, 1)
expected = np.dot(np.transpose(mat1), np.transpose(mat2))[0][0]
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 4")

# Test 5 - 2 scalars (_is_col set to True)
n = 1
mat1 = np.random.uniform(-1e4, 1e4, (n, n))
mat2 = np.random.uniform(-1e4, 1e4, (n, n))
actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), True, 1, np.ctypeslib.as_ctypes(mat2), True, 1)
expected = np.dot(mat1, mat2)[0][0]
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 5")

# Test 6 - 2 scalars again (_is_col set to False)
n = 1
mat1 = np.random.uniform(-1e4, 1e4, (n, n))
mat2 = np.random.uniform(-1e4, 1e4, (n, n))
actual = bp.ddot(n, np.ctypeslib.as_ctypes(mat1), False, 1, np.ctypeslib.as_ctypes(mat2), False, 1)
expected = np.dot(mat1, mat2)[0][0]
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 6")