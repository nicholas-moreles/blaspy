"""

    Copyright (c) 2014-2015-2015, The University of Texas at Austin.
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
BLAS_NAME = ""  # default is ""

# True if the BLAS .so or .dll file is in the blaspy/lib subdirectory,
# False if Python should search for it.
IN_BLASPY_SUBDIRECTORY = True  # default is True

###############################
# DO NOT EDIT BELOW THIS LINE #
###############################

# find the appropriate BLAS to use
if BLAS_NAME == "":  # try to use included OpenBLAS
    PREPEND = str(path.dirname(__file__))[:-6] + "lib/"
    if system() == "Windows":
        if calcsize("P") == 8:  # 64-bit
            BLAS_NAME = "libopenblas-0.2.13-win64-int32.dll"
            chdir(PREPEND + "win64")
        else:  # 32-bit
            BLAS_NAME = "libopenblas-0.2.13-win32.dll"
            chdir(PREPEND + "win32")
        PREPEND = ""
    elif system() == "Linux":
        if calcsize("P") == 8:  # 64-bit
            BLAS_NAME = "libopenblas-0.2.13-linux64.so"
            PREPEND += "linux64/"
        else:  # 32-bit
            BLAS_NAME = "libopenblas-0.2.13-linux32.so"
            PREPEND += "linux32/"
    else:  # no appropriate OpenBLAS included, BLAS_NAME_OVERRIDE must be used
        raise_blas_os_error()
else:
   PREPEND = ""

# Change the directory and load the library
_libblas = cdll.LoadLibrary(PREPEND + BLAS_NAME)