"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy_tests import *

TEST_DICT = {'amax': test_amax,  # level 1
             'asum': test_asum,
             'axpy': test_axpy,
             'copy': test_copy,
             'dot':  test_dot,
             'nrm2': test_nrm2,
             'scal': test_scal,
             'sdot': test_sdot,
             'swap': test_swap,
             'gemv': test_gemv,   # level 2
             'ger':  test_ger,
             'symv': test_symv,
             'syr':  test_syr,
             'syr2': test_syr2,
             'trmv': test_trmv,
             'gemm': test_gemm,   # level 3
             'symm': test_symm,
             'trmm': test_trmm}


def run_test(function):
    result = function()
    num_tests_failed = len(result)
    if num_tests_failed == 0:
        print("Passed all tests")
    else:
        print("*FAILED* " + str(num_tests_failed) + " tests, test numbers shown below:")
        print(result)
    print("")
    return num_tests_failed


total = 0

for name, function in sorted(TEST_DICT.iteritems()):
    print("Testing " + name)
    total += run_test(function)

# Give totals
if total == 0:
    print("No tests failed!")
else:
    print("Total tests failed: " + str(total))