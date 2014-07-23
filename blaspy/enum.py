"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

# Because of the lack of enum support prior to Python 3.4, enums in BLASpy are implemented using
# a custom enumeration function. No error-checking is done to ensure the correct enum (or any) is
# being used with a given function; however, this is in keeping with BLASpy's philosophy of not
# explicitly error checking to maintain speed.
#
# It is recommended that these enums are stored as variables (i.e. COL = Vec.COL) if they are going
# to be used repeatedly in a loop for fastest performance.

def enumeration(**named_values):
    return type('enumeration', (), named_values)

# Custom enum used by BLASpy for level 1 and level 2 BLAS routines to let ctypes know the
# orientation of the array to be passed in.
Vec = enumeration(COL = 0, ROW = 1, COL_COL = 0, ROW_COL = 1, COL_ROW = 2, ROW_ROW = 3)

# Enum for CBLAS_ORDER
Order = enumeration(ROW_MAJOR = 101, COL_MAJOR = 102)

# Enum for CBLAS_TRANSPOSE
Trans = enumeration(NO_TRANS = 111, TRANS = 112, CONJ_TRANS = 113, CONJ_NO_TRANS = 114)