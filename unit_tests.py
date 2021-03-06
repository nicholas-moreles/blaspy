"""

    Copyright (c) 2014-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from bp_unit_tests import *
from unittest import TestLoader, TestSuite, TextTestRunner

TEST_CASES = (TestAmax,  # level 1
              TestAsum,
              TestAxpy,
              TestCopy,
              TestDot,
              TestNrm2,
              TestScal,
              TestSdot,
              TestSwap,
              TestGemv,  # level 2
              TestGer,
              TestSymv,
              TestSyr,
              TestSyr2,
              TestTrsv)

suite = TestSuite()

for test_case in TEST_CASES:
    suite.addTest(TestLoader().loadTestsFromTestCase(test_case))

TextTestRunner(verbosity=2).run(suite)