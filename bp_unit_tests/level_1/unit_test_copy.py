"""

    Copyright (c) 2014-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import copy
from numpy import array, asmatrix, reshape
from unittest import TestCase


class TestCopy(TestCase):

    def test_scalar_as_ndarray(self):
        x = array([[1.]])
        y = array([[2.]])

        expected = x.tolist()
        self.assertEqual(copy(x, y).tolist(), expected)
        self.assertEqual(y.tolist(), expected)

    def test_two_row_vectors_as_ndarrays(self):
        x = array([[1., 2., 3.]])
        y = array([[4., 5., 6.]])

        expected = x.tolist()
        self.assertEqual(copy(x, y).tolist(), expected)
        self.assertEqual(y.tolist(), expected)

    def test_two_column_vectors_as_ndarrays(self):
        x = array([[1.], [2.], [3.]])
        y = array([[4.], [5.], [6.]])

        expected = x.tolist()
        self.assertEqual(copy(x, y).tolist(), expected)
        self.assertEqual(y.tolist(), expected)

    def test_col_and_row_vectors_as_ndarrays(self):
        x = array([[1.], [2.], [3.]])
        y = array([[4., 5., 6.]])

        expected = [[1., 2., 3.]]
        self.assertEqual(copy(x, y).tolist(), expected)
        self.assertEqual(y.tolist(), expected)

    def test_row_and_col_vectors_as_ndarrays(self):
        x = array([[1., 2., 3.]])
        y = array([[4.], [5.], [6.]])

        expected = [[1.], [2.], [3.]]
        self.assertEqual(copy(x, y).tolist(), expected)
        self.assertEqual(y.tolist(), expected)

    def test_negative_element(self):
        x = array([[1., -2., 3.]])
        y = array([[4., 5., 6.]])

        expected = x.tolist()
        self.assertEqual(copy(x, y).tolist(), expected)
        self.assertEqual(y.tolist(), expected)

    def test_vectors_as_matrices(self):
        x = asmatrix(array([[1., 2., 3.]]))
        y = asmatrix(array([[4., 5., 6.]]))

        expected = x.tolist()
        self.assertEqual(copy(x, y).tolist(), expected)
        self.assertEqual(y.tolist(), expected)

    def test_vectors_as_mixed_matrices_and_ndarrays(self):
        x = array([[1., 2., 3.]])
        y = asmatrix(array([[4., 5., 6.]]))

        expected = x.tolist()
        self.assertEqual(copy(x, y).tolist(), expected)
        self.assertEqual(y.tolist(), expected)

    def test_y_not_provided_with_row_vector(self):
        x = array([[1., -2., 3.]])

        expected = x.tolist()
        self.assertEqual(copy(x).tolist(), expected)

    def test_y_not_provided_with_column_vector(self):
        x = array([[1.], [2.], [3.]])

        expected = x.tolist()
        self.assertEqual(copy(x).tolist(), expected)

    def test_y_not_provided_with_stride_greater_than_one(self):
        x = array([[1., -2., 3.]])

        expected = [[1., 3.]]
        self.assertEqual(copy(x, inc_x=2).tolist(), expected)

    def test_strides_less_than_length(self):
        x = array([[1., 2., 3.]])
        y = array([[4., 5., 6.]])

        expected = [[1., 5., 3.]]
        self.assertListEqual(copy(x, y, inc_x=2, inc_y=2).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_strides_greater_than_length(self):
        x = array([[1., 2., 3.]])
        y = array([[4., 5., 6.]])

        expected = [[1., 5., 6.]]
        self.assertListEqual(copy(x, y, inc_x=3, inc_y=3).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_unequal_strides(self):
        x = array([[1., 2., 3., 4., 5., 6.]])
        y = array([[4., 5., 6.]])

        expected = [[1., 3., 5.]]
        self.assertListEqual(copy(x, y, inc_x=2, inc_y=1).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_float32_dtype(self):
        x = array([[1., 2., 3.]], dtype='float32')
        y = array([[3., 2., 1.]], dtype='float32')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(y.dtype, 'float32')

        expected = x.tolist()
        self.assertListEqual(copy(x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_float64_dtype(self):
        x = array([[1., 2., 3.]], dtype='float64')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float64')

        expected = x.tolist()
        self.assertListEqual(copy(x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_not_numpy_with_list_for_x_raises_ValueError(self):
        x = [[1., 2., 3.]]
        y = array([[3., 2., 1.]])
        self.assertRaises(ValueError, copy , x, y)

    def test_not_numpy_with_list_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = [[3., 2., 1.]]
        self.assertRaises(ValueError, copy, x, y)

    def test_not_numpy_with_scalar_for_x_raises_ValueError(self):
        x = 1.
        y = array([[3.]])
        self.assertRaises(ValueError, copy, x, y)

    def test_not_numpy_with_scalar_for_y_raises_ValueError(self):
        x = array([[1.]])
        y = 2.
        self.assertRaises(ValueError, copy, x, y)

    def test_not_2d_numpy_with_1d_for_x_raises_ValueError(self):
        x = array([1., 2., 3.])
        y = array([[3., 2., 1.]])
        self.assertRaises(ValueError, copy, x, y)

    def test_not_2d_numpy_with_1d_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = array([3., 2., 1.])
        self.assertRaises(ValueError, copy, x, y)

    def test_not_2d_numpy_with_3d_for_x_raises_ValueError(self):
        x = array([[[1.], [2.], [3.]]], ndmin=3)
        y = array([[3., 2., 1.]])
        self.assertRaises(ValueError, copy, x, y)

    def test_not_2d_numpy_with_3d_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = array([[[3.], [2.], [1.]]], ndmin=3)
        self.assertRaises(ValueError, copy, x, y)

    def test_not_vector_raises_ValueError(self):
        x = array([[1., 2.], [3., 4.]])
        y = array([[1., 2.], [3., 4.]])
        self.assertRaises(ValueError, copy, x, y)

    def test_unequal_vector_length_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = array([[3., 2.]])
        self.assertRaises(ValueError, copy, x, y)

    def test_unequal_vector_length_with_strides_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = array([[3., 2.]])
        self.assertRaises(ValueError, copy, x, y, 2, 3)

    def test_mixed_dtypes_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='float32')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, copy, x, y)

    def test_integer_dtype_for_both_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='int')
        y = array([[3., 2., 1.]], dtype='int')
        self.assertEqual(x.dtype, 'int')
        self.assertEqual(y.dtype, 'int')
        self.assertRaises(ValueError, copy, x, y)

    def test_integer_dtype_for_x_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='int')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'int')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, copy, x, y)

    def test_integer_dtype_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='float64')
        y = array([[3, 2., 1.]], dtype='int')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'int')
        self.assertRaises(ValueError, copy, x, y)

    def test_complex_dtype_for_both_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='complex')
        y = array([[3., 2., 1.]], dtype='complex')
        self.assertEqual(x.dtype, 'complex')
        self.assertEqual(y.dtype, 'complex')
        self.assertRaises(ValueError, copy, x, y)

    def test_complex_dtype_for_x_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='complex')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'complex')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, copy, x, y)

    def test_complex_dtype_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='float64')
        y = array([[3., 2., 1.]], dtype='complex')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'complex')
        self.assertRaises(ValueError, copy, x, y)