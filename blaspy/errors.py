"""

    Copyright (c) 2014-2015-2015, The University of Texas at Austin..
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

def raise_not_2d_numpy(name):
    raise ValueError("'%s' should be two-dimensional and of NumPy type ndarray or matrix. BLASpy "
                     "expects vectors to be two-dimensional, so ensure you are not trying to pass "
                     "in a one-dimensional ndarray as a vector." % name)


def raise_not_vector(name, rows, cols):
    raise ValueError("'%s' should be a vector. Either the number of rows or number of "
                     "columns must equal one. Number of rows: %i. Number of columns: %i." %
                     (name, rows, cols))


def raise_not_square(name, rows, cols):
    raise ValueError("'%s' should be a square matrix. The number of rows must equal the number of "
                     "columns. Number of rows: %i. Number of columns: %i" % (name, rows, cols))


def raise_size_mismatch(name1, size_1, name2, size_2):
    raise ValueError("There was a size mismatch between '%s' (%d) and '%s' (%d). Double check that their "
                     "sizes conform in a manner appropriate for the BLASpy function being "
                     "called." % (name1, size_1, name2, size_2))


def raise_strides_not_one():
    raise ValueError("If 'y' is not provided, then the stride of all vectors should equal one.")


def raise_invalid_dtypes(allowed):
    raise ValueError("All matrix and vector parameters should have the same dtype and that dtype "
                     "should be in the following list of allowed dtypes (which may change from "
                     "one BLASpy function to another): %s. This error can be fixed calling "
                     ".astype('dtype') on the problem ndarray or matrix before calling a BLASpy "
                     "function with that ndarray or matrix as a parameter." % (allowed,))


def raise_invalid_parameter(name, allowed, actual):
    raise ValueError("Parameter '%s' should equal one of the following values: %s. Actual value: "
                     "%s" % (name, (allowed,), actual))


def raise_blas_os_error():
    raise RuntimeError("BLASpy does not have a bundled BLAS implementation appropriate for "
                       "your operating system. Please download and compile one, "
                       "such as OpenBLAS, then modify config.py to point to that BLAS "
                       "implementation.")