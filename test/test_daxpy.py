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

# Test 1 - two row vectors and a positive scalar
n = random.randint(2, 100)
alpha = random.uniform(2, 1e6)
x1 = np.random.random((1, n))
y1 = np.random.random((1, n))
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), False, 1, np.ctypeslib.as_ctypes(y1), False, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 1")

# Test 2 - two row vectors and a negative scalar
n = random.randint(2, 100)
alpha = random.uniform(-1e6, -2)
x1 = np.random.random((1, n)) * random.uniform(-100, 100)
y1 = np.random.random((1, n)) * random.uniform(-100, 100)
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), False, 1, np.ctypeslib.as_ctypes(y1), False, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 2")

# Test 3 - two column vectors and a positive scalar
n = random.randint(2, 1000)
alpha = random.uniform(2, 1e6)
x1 = np.random.random((n, 1)) * random.uniform(-100, 100)
y1 = np.random.random((n, 1)) * random.uniform(-100, 100)
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), True, 1, np.ctypeslib.as_ctypes(y1), True, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 3")

# Test 4 - two column vectors and a negative scalar
n = random.randint(2, 1000)
alpha = random.uniform(-1e6, -2)
x1 = np.random.random((n, 1)) * random.uniform(-100, 100)
y1 = np.random.random((n, 1)) * random.uniform(-100, 100)
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), True, 1, np.ctypeslib.as_ctypes(y1), True, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 4")

# Test 5 - two row vectors and a zero-valued scalar
n = random.randint(2, 1000)
alpha = 0
x1 = np.random.random((1, n)) * random.uniform(-100, 100)
y1 = np.random.random((1, n)) * random.uniform(-100, 100)
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), False, 1, np.ctypeslib.as_ctypes(y1), False, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 5")

# Test 6 - two column vectors and a zero-valued scalar
n = random.randint(2, 1000)
alpha = 0
x1 = np.random.random((n, 1)) * random.uniform(-100, 100)
y1 = np.random.random((n, 1)) * random.uniform(-100, 100)
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), True, 1, np.ctypeslib.as_ctypes(y1), True, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 6")

# Test 7 - two row vectors and a one-valued scalar
n = random.randint(2, 1000)
alpha = 1
x1 = np.random.random((1, n)) * random.uniform(-100, 100)
y1 = np.random.random((1, n)) * random.uniform(-100, 100)
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), False, 1, np.ctypeslib.as_ctypes(y1), False, 1)
y2 += x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 7")

# Test 8 - two column vectors and a one-valued scalar
n = random.randint(2, 1000)
alpha = 1
x1 = np.random.random((n, 1)) * random.uniform(-100, 100)
y1 = np.random.random((n, 1)) * random.uniform(-100, 100)
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), True, 1, np.ctypeslib.as_ctypes(y1), True, 1)
y2 += x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 8")

# Test 9 - two row vectors with y being the zero vector and alpha being a positive scalar
n = random.randint(2, 1000)
alpha = random.uniform(2, 1e6)
x1 = np.random.random((1, n))
y1 = np.zeros((1, n))
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), False, 1, np.ctypeslib.as_ctypes(y1), False, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 9")

# Test 10 - two row vectors with y being the zero vector and alpha being a negative scalar
n = random.randint(2, 1000)
alpha = random.uniform(-1e6, -2)
x1 = np.random.random((1, n))
y1 = np.zeros((1, n))
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), False, 1, np.ctypeslib.as_ctypes(y1), False, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 10")

# Test 11 - two row vectors with y being the zero vector and alpha being the scalar zero
n = random.randint(2, 1000)
alpha = 0
x1 = np.random.random((1, n))
y1 = np.zeros((1, n))
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), False, 1, np.ctypeslib.as_ctypes(y1), False, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 10")

# Test 12 - two column vectors with y being the zero vector and alpha being a positive scalar
n = random.randint(2, 1000)
alpha = random.uniform(2, 1e6)
x1 = np.random.random((n, 1))
y1 = np.zeros((n, 1))
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), True, 1, np.ctypeslib.as_ctypes(y1), True, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 12")

# Test 13 - two column vectors with y being the zero vector and alpha being a negative scalar
n = random.randint(2, 1000)
alpha = random.uniform(2, 1e6)
x1 = np.random.random((n, 1))
y1 = np.zeros((n, 1))
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), True, 1, np.ctypeslib.as_ctypes(y1), True, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 13")

# Test 14 - two column vectors with y being the zero vector and alpha being the scalar zero
n = random.randint(2, 1000)
alpha = 0
x1 = np.random.random((n, 1))
y1 = np.zeros((n, 1))
y2 = np.copy(y1)
bp.daxpy(n, alpha, np.ctypeslib.as_ctypes(x1), True, 1, np.ctypeslib.as_ctypes(y1), True, 1)
y2 += alpha * x1
print(("Passed" if np.allclose(y1, y2) else "*FAILED*:") + " test 14")