"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy_tests import *


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

###########
# Level 1 #
###########

print("Testing Level 1 BLAS\n")

print("Testing amax:")
total += run_test(test_amax)

print("Testing asum:")
total += run_test(test_asum)

print("Testing axpy:")
total += run_test(test_axpy)

print("Testing copy:")
total += run_test(test_copy)

print("Testing dot:")
total += run_test(test_dot)

print("Testing nrm2:")
total += run_test(test_nrm2)

print("Testing scal:")
total += run_test(test_scal)

print("Testing sdot:")
total += run_test(test_sdot)

print("Testing swap:")
total += run_test(test_swap)


###########
# Level 2 #
###########

print("Testing Level 2 BLAS\n")

print("Testing gemv:")
total += run_test(test_gemv)

print("Testing ger:")
total += run_test(test_ger)

print("Testing symv:")
total += run_test(test_symv)

print("Testing syr:")
total += run_test(test_syr)

print("Testing syr2:")
total += run_test(test_syr2)


###########
# Level 3 #
###########

print("Testing Level 3 BLAS\n")


# Give totals
print("Total tests failed: " + str(total))