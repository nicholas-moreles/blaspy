"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from .errors import raise_blas_os_error
from ctypes import cdll
from os import chdir, path
from platform import system
from struct import calcsize

# The name of the BLAS .so or .dll file. By default this is the OpenBLAS reference
# implementation bundled with BLASpy. Only modify if you wish to use a different version of BLAS
# or if your operating system is not supported by BLASpy out of the box.
BLAS_NAME_OVERRIDE = ""  # default is ""

# True if the BLAS .so or .dll file is in the blaspy/lib subdirectory,
# False if Python should search for it.
IN_BLASPY_SUBDIR = True  # default is True

###############################
# DO NOT EDIT BELOW THIS LINE #
###############################

# find the appropriate BLAS to use
if BLAS_NAME_OVERRIDE == "":  # try to use included OpenBLAS
    if system() == "Windows":
        if calcsize("P") == 8:  # 64-bit
            BLAS_NAME = "libopenblas-0.2.13-win64-int32.dll"
        else:  # 32-bit
            BLAS_NAME = "libopenblas-0.2.13-win32.dll"
    elif system() == "Linux":
        if calcsize("P") == 8:  # 64-bit
            BLAS_NAME = "libopenblas-0.2.13-linux64.so"
        else:  # 32-bit
            BLAS_NAME = "libopenblas-0.2.10-linux32.so"
    else:  # no appropriate OpenBLAS included, BLAS_NAME_OVERRIDE must be used
        raise_blas_os_error()
else:
    BLAS_NAME = BLAS_NAME_OVERRIDE

# Change the directory (required for Windows 64-bit) and load the library
if IN_BLASPY_SUBDIR:
    chdir(str(path.dirname(__file__))[:-6] + "lib")
_libblas = cdll.LoadLibrary(BLAS_NAME)