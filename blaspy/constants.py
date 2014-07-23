"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""


# Constants for level 1 BLAS
# least significant bit is 1 if x is a row vectors, 0 if x is a column vector
# second least significant bit is 1 if y is a row vector, 0 if y is a column vector
COL = 0  # 00
ROW = 1  # 01
COL_COL = 0  # 00
ROW_COL = 1  # 01
COL_ROW = 2  # 10
ROW_ROW = 3  # 11