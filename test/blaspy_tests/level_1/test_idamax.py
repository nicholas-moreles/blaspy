"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

import blaspy as bp
import numpy as np

def test_idamax():

    tests_failed = []
    test_num = 0

    # test 1 - typical row vector
    test_num += 1
    mat1 = np.array([[-7.29544522, -52.85619672, -61.41985094, -53.34163619,  96.1944303, 70.26068289,
                    -59.34631157, -67.28938408, -98.77254851,  16.11719105]])
    n = max(mat1.shape[0], mat1.shape[1])
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.ROW)
    expected = np.argmax(np.absolute(mat1), 1)
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    # test 2 - typical column vector
    test_num += 1
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
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.COL)
    expected = np.argmax(np.absolute(mat1), 0)
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    # test 3 - row vector with first element as max
    test_num += 1
    mat1 = np.array([[99.32049224, -52.85619672, -61.41985094, -53.34163619,  96.1944303, 70.26068289,
                    -59.34631157, -67.28938408, -98.77254851,  16.11719105]])
    n = max(mat1.shape[0], mat1.shape[1])
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.ROW)
    expected = np.argmax(np.absolute(mat1), 1)
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    # test 4 - row vector with last element as max
    test_num += 1
    mat1 = np.array([[-7.29544522, -52.85619672, -61.41985094, -53.34163619,  96.1944303, 70.26068289,
                    -59.34631157, -67.28938408, -98.77254851,  -99.25841845]])
    n = max(mat1.shape[0], mat1.shape[1])
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.ROW)
    expected = np.argmax(np.absolute(mat1), 1)
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    # test 5 - column vector with first element as max
    test_num += 1
    mat1 = np.array([[103.21584219],
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
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.COL)
    expected = np.argmax(np.absolute(mat1), 0)
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    # test 6 - column vector with last element as max
    test_num += 1
    mat1 = np.array([[-19.81752462],
                     [-97.44630062],
                     [ 13.2843847 ],
                     [ 81.47844988],
                     [-89.35922821],
                     [-41.72931453],
                     [-81.04715189],
                     [ 46.26887276],
                     [  7.11428604],
                     [-98.06536253]])
    n = max(mat1.shape[0], mat1.shape[1])
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.COL)
    expected = np.argmax(np.absolute(mat1), 0)
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    # test 7 - single element column vector
    test_num += 1
    mat1 = np.random.uniform(-100, 100, (1,1))
    n = max(mat1.shape[0], mat1.shape[1])
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.COL)
    expected = 0
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    # test 8 - single element row vector
    test_num += 1
    mat1 = np.random.uniform(-100, 100, (1,1))
    n = max(mat1.shape[0], mat1.shape[1])
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.ROW)
    expected = 0
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    # test 9 - large row vector with large values
    test_num += 1
    mat1 = np.random.uniform(-1e6, 1e6, (1, 1e5))
    n = max(mat1.shape[0], mat1.shape[1])
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.ROW)
    expected = np.argmax(np.absolute(mat1), 1)
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    # test 10 - large column vector with large values
    test_num += 1
    mat1 = np.random.uniform(-1e6, 1e6, (1e5, 1))
    n = max(mat1.shape[0], mat1.shape[1])
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.COL)
    expected = np.argmax(np.absolute(mat1), 0)
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    # test 11 - duplicate of max value, ensure first index is returned
    test_num += 1
    mat1 = np.array([[-7.29544522, -52.85619672, -98.77254851, -53.34163619,  96.1944303, 70.26068289,
                    -59.34631157, -67.28938408, -98.77254851,  16.11719105]])
    n = max(mat1.shape[0], mat1.shape[1])
    actual = bp.idamax(n, np.ctypeslib.as_ctypes(mat1), 1, bp.ROW)
    expected = 2
    passed = actual == expected
    if not passed: tests_failed.append(test_num)

    return tests_failed