"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import nrm2
from numpy import array, asmatrix
from unittest import TestCase


class TestNrm2(TestCase):

    def test_scalar_as_ndarray(self):
        x = array([[1.]])
        self.assertEqual(nrm2(x), 1)

    def test_row_vector_as_ndarray(self):
        x = array([[1., 2., 3., 3., 1., 1.]])
        self.assertEqual(nrm2(x), 5)

    def test_negative_element(self):
        x = array([[1., -2., 3., 3., -1., 1.]])
        self.assertEqual(nrm2(x), 5)

    def test_column_vector_as_ndarray(self):
        x = array([[1.], [2.], [3.], [3.], [1.], [1.]])
        self.assertEqual(nrm2(x), 5)

    def test_vector_as_matrix(self):
        x = asmatrix(array([[1.], [2.], [3.], [3.], [1.], [1.]]))
        self.assertEqual(nrm2(x), 5)

    def test_stride_less_than_length(self):
        x = array([[1., 2., 2., 3., 2., 1.]])
        self.assertEqual(nrm2(x, inc_x=2), 3)

    def test_stride_greater_than_length(self):
        x = array([[1., -2., 3., 3., -1., 1.]])
        self.assertEqual(nrm2(x, inc_x=6), 1)

    def test_float32_dtype(self):
        x = array([[1., 2., 3., 3., 1., 1.]], dtype='float32')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(nrm2(x), 5)

    def test_float64_dtype(self):
        x = array([[1., 2., 3., 3., 1., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(nrm2(x), 5)

    def test_not_numpy_with_list_raises_ValueError(self):
        x = [[1., 2., 3., 3., 1., 1.]]
        self.assertRaises(ValueError, nrm2, x)

    def test_not_numpy_with_scalar_raises_ValueError(self):
        x = 1.
        self.assertRaises(ValueError, nrm2, x)

    def test_not_2d_numpy_with_1d_raises_ValueError(self):
        x = array([1., 2.])
        self.assertRaises(ValueError, nrm2, x)

    def test_not_2d_numpy_with_3d_raises_ValueError(self):
        x = array([[[1.], [2.]]], ndmin=3)
        self.assertRaises(ValueError, nrm2, x)

    def test_not_vector_raises_ValueError(self):
        x = array([[1., 2.], [3., 4.]])
        self.assertRaises(ValueError, nrm2, x)

    def test_integer_dtype_raises_ValueError(self):
        x = array([[1., 2.]], dtype='int')
        self.assertRaises(ValueError, nrm2, x)

    def test_complex_dtype_raises_ValueError(self):
        x = array([[1., 2.]], dtype='complex')
        self.assertRaises(ValueError, nrm2, x)