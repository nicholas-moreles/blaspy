"""

    Copyright (c) 2014, The University of Texas at Austin
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

# Test dasum
print("Testing dasum:")
total += run_test(test_dasum)

# Test daxpy
print("Testing daxpy:")
total += run_test(test_daxpy)

# Test dcopy
print("Testing dcopy:")
total += run_test(test_dcopy)

# Test ddot
print("Testing ddot:")
total += run_test(test_ddot)

# Test dnrm2
print("Testing dnrm2:")
total += run_test(test_dnrm2)

# Test dscal
print("Testing dscal:")
total += run_test(test_dscal)

# Test dsdot
print("Testing dsdot:")
total += run_test(test_dsdot)

# Test dswap
print("Testing dswap:")
total += run_test(test_dswap)

# Test idamax
print("Testing idamax:")
total += run_test(test_idamax)


# Give totals
print("Total tests failed: " + str(total))