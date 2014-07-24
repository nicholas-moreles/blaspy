"""

    Copyright (c) 2014, The University of Texas at Austin
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""


import ctypes as c
import os

# The name of the BLAS .so file. By default this is the OpenBLAS reference implementation included
# with BLASpy. Only modify if you wish to use a different version of BLAS you have installed.
BLAS_NAME = "libopenblasp-r0.2.9-64ref32threads.so"

# True if the BLAS .so file is in the blaspy/lib subdirectory,
# False if Python should search for it in /usr/lib
IN_BLASPY_SUBDIR = True

##################################
# DO NOT EDIT BELOW THIS COMMENT #
##################################

# Create the appropriate path to _libblas
BLAS_PATH = str(os.path.dirname(__file__))[:-6] + "lib/"
_libblas = c.cdll.LoadLibrary((BLAS_PATH + BLAS_NAME) if IN_BLASPY_SUBDIR else BLAS_NAME)
