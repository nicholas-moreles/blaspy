"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from numpy import copy, dot, zeros


def expected_symv(A, x, x_is_row, y, y_is_row, n, alpha, beta, stride):
    """ Calculate the expected result for symmetric matrix-vector multiplication. """

    # create x_2 and y_2 vectors that are column vectors with the same elements as x and y
    x_2 = x.T if x_is_row else x
    if y is None:
        y_2 = zeros((1, n))
    else:
        y_2 = copy(y.T) if y_is_row else copy(y)

    # compute expected results
    if stride == 1:
        y_2 = beta * y_2 + alpha * dot(A, x_2)
    else:
        for i in range(0, y_2.shape[0], stride):
            A_partition = A[i / stride, :]
            x_partition = x_2[:: stride, :]
            y_2[i, 0] = (beta * y_2[i, 0]) + (alpha * dot(A_partition, x_partition))

    return y_2