"""

    Copyright (c) 2014-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import dot
from numpy import array, array_equal, asmatrix
from unittest import TestCase


class TestDot(TestCase):

    def test_scalar_as_ndarray(self):
        x = array([[1.]])
        y = array([[2.]])
        self.assertEqual(dot(x, y), 2)

    def test_two_row_vectors_as_ndarrays(self):
        x = array([[1., 2., 3.]])
        y = array([[3., 2., 1.]])
        self.assertEqual(dot(x, y), 10)

    def test_two_column_vectors_as_ndarrays(self):
        x = array([[1.], [2.], [3.]])
        y = array([[3.], [2.], [1.]])
        self.assertEqual(dot(x, y), 10)

    def test_col_and_row_vectors_as_ndarrays(self):
        x = array([[1.], [2.], [3.]])
        y = array([[3., 2., 1.]])
        self.assertEqual(dot(x, y), 10)

    def test_row_and_col_vectors_as_ndarrays(self):
        x = array([[1., 2., 3.]])
        y = array([[3.], [2.], [1.]])
        self.assertEqual(dot(x, y), 10)

    def test_vectors_with_negatives_in_values(self):
        x = array([[-1., -2., 3.]])
        y = array([[3., 2., 1.]])
        self.assertEqual(dot(x, y), -4)

    def test_vectors_as_matrices(self):
        x = asmatrix(array([[1., 2., 3.]]))
        y = asmatrix(array([[3., 2., 1.]]))
        self.assertEqual(dot(x, y), 10)

    def test_vectors_as_mixed_matrices_and_ndarrays(self):
        x = asmatrix(array([[1., 2., 3.]]))
        y = array([[3., 2., 1.]])
        self.assertEqual(dot(x, y), 10)

    def test_strides_less_than_length(self):
        x = array([[1., 2., 3.]])
        y = array([[3., 2., 1.]])
        self.assertEqual(dot(x, y, inc_x=2, inc_y=2), 6)

    def test_strides_greater_than_length(self):
        x = array([[1., 2., 3.]])
        y = array([[3., 2., 1.]])
        self.assertEqual(dot(x, y, inc_x=3, inc_y=3), 3)

    def test_unequal_strides(self):
        x = array([[1., 2., 3., 4., 5., 6.]])
        y = array([[3., 2., 1.]])
        self.assertEqual(dot(x, y, inc_x=3, inc_y=2), 7)

    def test_float32_dtype(self):
        x = array([[1., 2., 3.]], dtype='float32')
        y = array([[3., 2., 1.]], dtype='float32')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(y.dtype, 'float32')
        self.assertEqual(dot(x, y), 10)

    def test_float64_dtype(self):
        x = array([[1., 2., 3.]], dtype='float64')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float64')
        self.assertEqual(dot(x, y), 10)

    def test_not_numpy_with_list_for_x_raises_ValueError(self):
        x = [[1., 2., 3.]]
        y = array([[3., 2., 1.]])
        self.assertRaises(ValueError, dot, x, y)

    def test_not_numpy_with_list_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = [[3., 2., 1.]]
        self.assertRaises(ValueError, dot, x, y)

    def test_not_numpy_with_scalar_for_x_raises_ValueError(self):
        x = 1.
        y = array([[3.]])
        self.assertRaises(ValueError, dot, x, y)

    def test_not_numpy_with_scalar_for_y_raises_ValueError(self):
        x = array([[1.]])
        y = 2.
        self.assertRaises(ValueError, dot, x, y)

    def test_not_2d_numpy_with_1d_for_x_raises_ValueError(self):
        x = array([1., 2., 3.])
        y = array([[3., 2., 1.]])
        self.assertRaises(ValueError, dot, x, y)

    def test_not_2d_numpy_with_1d_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = array([3., 2., 1.])
        self.assertRaises(ValueError, dot, x, y)

    def test_not_2d_numpy_with_3d_for_x_raises_ValueError(self):
        x = array([[[1.], [2.], [3.]]], ndmin=3)
        y = array([[3., 2., 1.]])
        self.assertRaises(ValueError, dot, x, y)

    def test_not_2d_numpy_with_3d_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = array([[[3.], [2.], [1.]]], ndmin=3)
        self.assertRaises(ValueError, dot, x, y)

    def test_not_vector_raises_ValueError(self):
        x = array([[1., 2.], [3., 4.]])
        y = array([[1., 2.], [3., 4.]])
        self.assertRaises(ValueError, dot, x, y)

    def test_unequal_vector_length_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = array([[3., 2.]])
        self.assertRaises(ValueError, dot, x, y)

    def test_unequal_vector_length_with_strides_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = array([[3., 2.]])
        self.assertRaises(ValueError, dot, x, y, 2, 3)

    def test_mixed_dtypes_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='float32')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, dot, x, y)

    def test_integer_dtype_for_both_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='int')
        y = array([[3., 2., 1.]], dtype='int')
        self.assertEqual(x.dtype, 'int')
        self.assertEqual(y.dtype, 'int')
        self.assertRaises(ValueError, dot, x, y)

    def test_integer_dtype_for_x_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='int')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'int')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, dot, x, y)

    def test_integer_dtype_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='float64')
        y = array([[3, 2., 1.]], dtype='int')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'int')
        self.assertRaises(ValueError, dot, x, y)

    def test_complex_dtype_for_both_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='complex')
        y = array([[3., 2., 1.]], dtype='complex')
        self.assertEqual(x.dtype, 'complex')
        self.assertEqual(y.dtype, 'complex')
        self.assertRaises(ValueError, dot, x, y)

    def test_complex_dtype_for_x_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='complex')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'complex')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, dot, x, y)

    def test_complex_dtype_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='float64')
        y = array([[3., 2., 1.]], dtype='complex')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'complex')
        self.assertRaises(ValueError, dot, x, y)