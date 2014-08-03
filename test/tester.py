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

print("Testing asum:")
total += run_test(test_asum)

print("Testing axpy:")
total += run_test(test_axpy)

print("Testing copy:")
total += run_test(test_copy)

print("Testing dot:")
total += run_test(test_dot)

print("Testing scal:")
total += run_test(test_scal)

print("Testing sdot:")
total += run_test(test_sdot)


###########
# Level 2 #
###########

print("Testing Level 2 BLAS\n")


###########
# Level 3 #
###########

print("Testing Level 3 BLAS\n")


# Give totals
print("Total tests failed: " + str(total))