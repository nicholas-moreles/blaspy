"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

import blaspy as bp
import numpy as np

epsilon = 0.000001  # account for round-off error

# test 1 - typical row vector
mat1 = np.array([[-7.29544522, -52.85619672, -61.41985094, -53.34163619,  96.1944303, 70.26068289,
                -59.34631157, -67.28938408, -98.77254851,  16.11719105]])
n = max(mat1.shape[0], mat1.shape[1])
actual = bp.dnrm2(n, np.ctypeslib.as_ctypes(mat1), False, 1)
expected = np.linalg.norm(mat1)
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 1")

# test 2 - typical column vector
mat1 = np.array([[-19.81752462],
                 [-97.44630062],
                 [ 13.2843847 ],
                 [ 81.47844988],
                 [-89.35922821],
                 [-41.72931453],
                 [-81.04715189],
                 [ 46.26887276],
                 [  7.11428604],
                 [-76.06536253]])
n = max(mat1.shape[0], mat1.shape[1])
actual = bp.dnrm2(n, np.ctypeslib.as_ctypes(mat1), True, 1)
expected = np.linalg.norm(mat1)
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 2")

# test 3 - single element column vector
mat1 = np.random.uniform(-100, 100, (1,1))
n = max(mat1.shape[0], mat1.shape[1])
actual = bp.dnrm2(n, np.ctypeslib.as_ctypes(mat1), True, 1)
expected = np.linalg.norm(mat1)
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 3")

# test 4 - single element row vector
mat1 = np.random.uniform(-100, 100, (1,1))
n = max(mat1.shape[0], mat1.shape[1])
actual = bp.dnrm2(n, np.ctypeslib.as_ctypes(mat1), False, 1)
expected = np.linalg.norm(mat1)
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 4")

# test 5 - large row vector with large values
mat1 = np.random.uniform(-1e4, 1e4, (1, 1e4))
n = max(mat1.shape[0], mat1.shape[1])
actual = bp.dnrm2(n, np.ctypeslib.as_ctypes(mat1), False, 1)
expected = np.linalg.norm(mat1)
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 5")

# test 6 - large column vector with large values
mat1 = np.random.uniform(-1e4, 1e4, (1e4, 1))
n = max(mat1.shape[0], mat1.shape[1])
actual = bp.dnrm2(n, np.ctypeslib.as_ctypes(mat1), True, 1)
expected = np.linalg.norm(mat1)
print(("Passed" if abs(actual - expected) < epsilon else "*FAILED*") + " test 6")