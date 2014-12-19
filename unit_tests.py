"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from bp_unit_tests import *
from unittest import TestLoader, TestSuite, TextTestRunner

TEST_CASES = (TestAmax,
              )

suite = TestSuite()

for test_case in TEST_CASES:
    suite.addTest(TestLoader().loadTestsFromTestCase(test_case))

TextTestRunner().run(suite)