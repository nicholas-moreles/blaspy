"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import amax
from numpy import array, asmatrix
from unittest import TestCase


class TestAmax(TestCase):

    def test_scalar_as_ndarray(self):
        x = array([[1.]])
        self.assertEqual(amax(x), 0)

    def test_row_vector_as_ndarray(self):
        x = array([[1., 2., 3.]])
        self.assertEqual(amax(x), 2)

    def test_first_element(self):
        x = x = array([[5., 2., 3.]])
        self.assertEqual(amax(x), 0)

    def test_middle_element(self):
        x = x = array([[5., 8., 3.]])
        self.assertEqual(amax(x), 1)

    def test_last_element(self):
        x = x = array([[5., 2., 10.]])
        self.assertEqual(amax(x), 2)

    def test_negative_element(self):
        x = x = array([[5., -6.5, 3.]])
        self.assertEqual(amax(x), 1)

    def test_column_vector_as_ndarray(self):
        x = array([[1.], [2.], [3.]])
        self.assertEqual(amax(x), 2)

    def test_vector_as_matrix(self):
        x = asmatrix(array([[1., 2., 3.]]))
        self.assertEqual(amax(x), 2)

    def test_stride_less_than_length(self):
        x = array([[1., 2., 3., 4.]])
        self.assertEqual(amax(x, 2), 1)  # 3. is at index 1

    def test_stride_greater_than_length(self):
        x = array([[1., 2., 3., 4.]])
        self.assertEqual(amax(x, 5), 0)  # 1 is at index 0

    def test_float32_dtype(self):
        x = array([[1., 2., 3.]], dtype='float32')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(amax(x), 2)

    def test_float64_dtype(self):
        x = array([[1., 2., 3.]], dtype='float64')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(amax(x), 2)

    def test_not_numpy_with_list_raises_ValueError(self):
        x = [[1., 2.]]
        self.assertRaises(ValueError, amax, x)

    def test_not_numpy_with_scalar_raises_ValueError(self):
        x = 1.
        self.assertRaises(ValueError, amax, x)

    def test_not_2d_numpy_with_1d_raises_ValueError(self):
        x = array([1., 2.])
        self.assertRaises(ValueError, amax, x)

    def test_not_2d_numpy_with_3d_raises_ValueError(self):
        x = array([[[1.], [2.]]], ndmin=3)
        self.assertRaises(ValueError, amax, x)

    def test_not_vector_raises_ValueError(self):
        x = array([[1., 2.], [3., 4.]])
        self.assertRaises(ValueError, amax, x)

    def test_integer_dtype_raises_ValueError(self):
        x = array([[1., 2.]], dtype='int')
        self.assertRaises(ValueError, amax, x)

    def test_complex_dtype_raises_ValueError(self):
        x = array([[1., 2.]], dtype='complex')
        self.assertRaises(ValueError, amax, x)