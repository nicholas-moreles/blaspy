"""

    Copyright (c) 2014-2015-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from bp_acceptance_tests import *

ACCEPTANCE_TESTS = {'amax':  acceptance_test_amax,  # level 1
                    'asum':  acceptance_test_asum,
                    'axpy':  acceptance_test_axpy,
                    'copy':  acceptance_test_copy,
                    'dot':   acceptance_test_dot,
                    'nrm2':  acceptance_test_nrm2,
                    'scal':  acceptance_test_scal,
                    'sdot':  acceptance_test_sdot,
                    'swap':  acceptance_test_swap,
                    'gemv':  acceptance_test_gemv,   # level 2
                    'ger':   acceptance_test_ger,
                    'symv':  acceptance_test_symv,
                    'syr':   acceptance_test_syr,
                    'syr2':  acceptance_test_syr2,
                    'trmv':  acceptance_test_trmv,
                    'trsv':  acceptance_test_trsv,
                    'gemm':  acceptance_test_gemm,   # level 3
                    'symm':  acceptance_test_symm,
                    'syrk':  acceptance_test_syrk,
                    'syr2k': acceptance_test_syr2k,
                    'trmm':  acceptance_test_trmm,
                    'trsm':  acceptance_test_trsm}


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

for name, function in sorted(ACCEPTANCE_TESTS.iteritems()):
    print("Running acceptance tests for " + name)
    total += run_test(function)

# Give totals
if total == 0:
    print("No tests failed!")
else:
    print("Total tests failed: " + str(total))