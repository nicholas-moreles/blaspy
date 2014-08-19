"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from ctypes import cdll
from os import path
from platform import system
from struct import calcsize

# The name of the BLAS .so or .dll file. By default this is the OpenBLAS reference
# implementation bundled with BLASpy. Only modify if you wish to use a different version of BLAS.
BLAS_NAME_OVERRIDE = ""  # default is ""

# True if the BLAS .so or .dll file is in the blaspy/lib subdirectory,
# False if Python should search for it in /usr/lib
IN_BLASPY_SUBDIR = True  # default is True

##################################
# DO NOT EDIT BELOW THIS COMMENT #
##################################

# find the appropriate BLAS to use
if BLAS_NAME_OVERRIDE == "":
    if system() == "Windows":
        if calcsize("P") == 8:  # 64-bit
            BLAS_NAME = "NOT_IMPLEMENTED.dll"
        else:  # 32-bit
            BLAS_NAME = "NOT_IMPLEMENTED.dll"
    else:  # Linux
        if calcsize("P") == 8:  # 64-bit
            BLAS_NAME = "libopenblasp-r0.2.9-64ref32threads.so"
        else:  # 32-bit
            BLAS_NAME = "NOT_IMPLEMENTED.dll"

# Create the appropriate path to _libblas
BLAS_PATH = str(path.dirname(__file__))[:-6] + "lib/"
_libblas = cdll.LoadLibrary((BLAS_PATH + BLAS_NAME) if IN_BLASPY_SUBDIR else BLAS_NAME)