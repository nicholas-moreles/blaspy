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

# Test 1 - scale a scalar by a positive int
n = 1
alpha = 10
mat1 = np.random.random((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 1")

# Test 2 - scale a scalar by a negative int
n = 1
alpha = -2000
mat1 = np.random.random((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 1")

# Test 3 - scale a scalar by zero (int)
n = 1
alpha = 0
mat1 = np.random.random((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 1")

# Test 4 - scale a scalar by a positive float
n = 1
alpha = 5634.3420
mat1 = np.random.random((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 4")

# Test 5 - scale a scalar by a negative float
n = 1
alpha = -0.212358
mat1 = np.random.random((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 5")

# Test 6 - scale a scalar by zero (float)
n = 1
alpha = 0.000
mat1 = np.random.random((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 6")

# Test 7 - scale a zero scalar by a positive int
n = 1
alpha = 123456
mat1 = np.zeros((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 7")

# Test 8 - scale a zero scalar by a negative int
n = 1
alpha = -1
mat1 = np.zeros((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 8")

# Test 9 - scale a zero scalar by zero (int)
n = 1
alpha = 0
mat1 = np.zeros((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 9")

# Test 10 - scale a zero scalar by a positive float
n = 1
alpha = 0.00358
mat1 = np.zeros((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 10")

# Test 11 - scale a zero scalar by a negative float
n = 1
alpha = -988533.22726
mat1 = np.zeros((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 11")

# Test 12 - scale a zero scalar by zero (float)
n = 1
alpha = 0.0
mat1 = np.zeros((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 12")

# Test 13 - scale a scalar by a float (with x_is_col set to False)
n = 1
alpha = 2.5486
mat1 = np.random.random((n, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), False, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 13")

# Test 14 - scale a column vector by a positive int
n = 350
alpha = 123456
mat1 = np.random.random((n, 1))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 14")

# Test 15 - scale a column vector by a negative int
n = 8900
alpha = -450
mat1 = np.random.random((n, 1))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 15")

# Test 16 - scale a column vector by zero (int)
n = 256
alpha = 0
mat1 = np.random.random((n, 1))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 16")

# Test 17 - scale a column vector by one (float)
n = 857423
alpha = 1.000000
mat1 = np.random.random((n, 1))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 17")

# Test 18 - scale a column vector by a positive float
n = 2048
alpha = 5343.2334454
mat1 = np.random.random((n, 1))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 18")

# Test 19 - scale a column vector by a negative float
n = 512
alpha = -0.0123498
mat1 = np.random.random((n, 1))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 19")

# Test 20 - scale a column vector by zero (float)
n = 999999
alpha = 0.0
mat1 = np.random.random((n, 1))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), True, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 20")

# Test 21 - scale a row vector by a positive int
n = random.randint(2, 2e4)
alpha = random.randint(2, 2e6)
mat1 = np.random.random((1, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), False, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 21")

# Test 22 - scale a row vector by a negative int
n = 20000
alpha = random.randint(-2e6, -2)
mat1 = np.random.random((1, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), False, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 22")

# Test 23 - scale a row vector by zero (int)
n = random.randint(2, 2e4)
alpha = 0
mat1 = np.random.random((1, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), False, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 23")

# Test 24 - scale a column vector by one (float)
n = random.randint(2, 2e4)
alpha = 1.000000
mat1 = np.random.random((1, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), False, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 24")

# Test 25 - scale a column vector by a positive float
n = random.randint(2, 2e4)
alpha = random.uniform(2, 2e6)
mat1 = np.random.random((1, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), False, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 25")

# Test 26 - scale a column vector by a negative float
n = random.randint(2, 2e4)
alpha = random.uniform(-2e6, -2)
mat1 = np.random.random((1, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), False, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 26")

# Test 27 - scale a column vector by zero (float)
n = random.randint(2, 2e4)
alpha = 0.0
mat1 = np.random.random((1, n))
mat2 = np.copy(mat1)
bp.dscal(n, alpha, np.ctypeslib.as_ctypes(mat1), False, 1)
mat2 *= alpha
print(("Passed" if np.allclose(mat1, mat2) else "*FAILED*") + " test 27")