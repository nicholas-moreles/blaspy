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

# Test 1 - two row vectors
n = random.randint(2, 1e3)
x1 = np.random.uniform(-1e5, 1e5, (1, n))
y1 = np.random.uniform(-1e5, 1e5, (1, n))
x2 = np.copy(x1)
y2 = np.copy(y1)
bp.dswap(n, np.ctypeslib.as_ctypes(x2), False, 1, np.ctypeslib.as_ctypes(y2), False, 1)
print(("Passed" if np.allclose(x1, y2) and np.allclose(x2, y1) else "*FAILED*") + " test 1")

# Test 2 - a row vector and a column vector
n = random.randint(2, 1e3)
x1 = np.random.uniform(-1e5, 1e5, (1, n))
y1 = np.random.uniform(-1e5, 1e5, (n, 1))
x2 = np.copy(x1)
y2 = np.copy(y1)
bp.dswap(n, np.ctypeslib.as_ctypes(x2), False, 1, np.ctypeslib.as_ctypes(y2), True, 1)
print(("Passed" if np.allclose(np.transpose(x1), y2) and np.allclose(np.transpose(x2), y1)
       else "*FAILED*") + " test 2")

# Test 3 - a column vector and a row vector
n = random.randint(1e3, 1e4)
x1 = np.random.uniform(-1e5, 1e5, (n, 1))
y1 = np.random.uniform(-1e5, 1e5, (1, n))
x2 = np.copy(x1)
y2 = np.copy(y1)
bp.dswap(n, np.ctypeslib.as_ctypes(x2), True, 1, np.ctypeslib.as_ctypes(y2), False, 1)
print(("Passed" if np.allclose(np.transpose(x1), y2) and np.allclose(np.transpose(x2), y1)
       else "*FAILED*") + " test 3")

# Test 4 - two column vectors
n = random.randint(1e3, 1e4)
x1 = np.random.uniform(-1e5, 1e5, (n, 1))
y1 = np.random.uniform(-1e5, 1e5, (n, 1))
x2 = np.copy(x1)
y2 = np.copy(y1)
bp.dswap(n, np.ctypeslib.as_ctypes(x2), True, 1, np.ctypeslib.as_ctypes(y2), True, 1)
print(("Passed" if np.allclose(x1, y2) and np.allclose(x2, y1) else "*FAILED*") + " test 4")