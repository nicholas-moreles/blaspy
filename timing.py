"""

    Copyright (c) 2014-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from bp_timing import timing_gemm

TRIALS = 10
K = 1500
TEST_DICT = {'gemm': timing_gemm}


for name, function in sorted(TEST_DICT.iteritems()):
    print("Testing " + name)
    function(TRIALS, K)