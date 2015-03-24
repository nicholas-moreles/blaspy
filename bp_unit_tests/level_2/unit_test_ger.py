"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import ger
from numpy import array, asmatrix
from unittest import TestCase


class TestGer(TestCase):

    def test_scalars_as_ndarray_provide_A(self):
        A = array([[1.]])
        x = array([[2.]])
        y = array([[3.]])
        self.assertEqual(ger(x, y, A), 7)
        self.assertEqual(A, 7)

    def test_scalars_as_ndarray_no_A(self):
        x = array([[2.]])
        y = array([[3.]])
        self.assertEqual(ger(x, y), 6)

    def test_two_row_vectors_as_ndarrays_provide_A(self):
        A = array([[1., 0., 1.],
                   [-1., 3., 2.],
                   [0., 1., 0.]])
        x = array([[1., 2., 3.]])
        y = array([[3., 2., 1.]])

        expected = [[4., 2., 2.],
                    [5., 7., 4.],
                    [9., 7., 3.]]
        self.assertListEqual(ger(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_two_row_vectors_as_ndarrays_no_A(self):
        x = array([[1., 2., 3.]])
        y = array([[3., 2., 1.]])

        expected = [[3., 2., 1.],
                    [6., 4., 2.],
                    [9., 6., 3.]]
        self.assertListEqual(ger(x, y).tolist(), expected)

    def test_two_column_vectors_as_ndarrays_provide_A(self):
        A = array([[1., 0., 1.],
                   [-1., 3., 2.],
                   [0., 1., 0.]])
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3.],
                   [2.],
                   [1.]])

        expected = [[4., 2., 2.],
                    [5., 7., 4.],
                    [9., 7., 3.]]
        self.assertListEqual(ger(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_two_column_vectors_as_ndarrays_no_A(self):
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3.],
                   [2.],
                   [1.]])

        expected = [[3., 2., 1.],
                    [6., 4., 2.],
                    [9., 6., 3.]]
        self.assertListEqual(ger(x, y).tolist(), expected)

    def test_col_and_row_vectors_as_ndarrays_provide_A(self):
        A = array([[1., 0., 1.],
                   [-1., 3., 2.],
                   [0., 1., 0.]])
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3., 2., 1.]])

        expected = [[4., 2., 2.],
                    [5., 7., 4.],
                    [9., 7., 3.]]
        self.assertListEqual(ger(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_col_and_row_vectors_as_ndarrays_no_A(self):
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3., 2., 1.]])

        expected = [[3., 2., 1.],
                    [6., 4., 2.],
                    [9., 6., 3.]]
        self.assertListEqual(ger(x, y).tolist(), expected)

    def test_row_and_col_vectors_as_ndarrays_provide_A(self):
        A = array([[1., 0., 1.],
                   [-1., 3., 2.],
                   [0., 1., 0.]])
        x = array([[1., 2., 3.]])
        y = array([[3.],
                   [2.],
                   [1.]])

        expected = [[4., 2., 2.],
                    [5., 7., 4.],
                    [9., 7., 3.]]
        self.assertListEqual(ger(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_row_and_col_vectors_as_ndarrays_no_A(self):
        x = array([[1., 2., 3.]])
        y = array([[3.],
                   [2.],
                   [1.]])

        expected = [[3., 2., 1.],
                    [6., 4., 2.],
                    [9., 6., 3.]]
        self.assertListEqual(ger(x, y).tolist(), expected)

    def test_vectors_with_negatives_in_values(self):
        x = array([[-1.],
                   [-2.],
                   [3.]])
        y = array([[3., 2., 1.]])

        expected = [[-3., -2., -1.],
                    [-6., -4., -2.],
                    [9., 6., 3.]]
        self.assertListEqual(ger(x, y).tolist(), expected)

    def test_as_matrices_provide_A(self):
        A = asmatrix(array([[1., 0., 1.],
                            [-1., 3., 2.],
                            [0., 1., 0.]]))
        x = asmatrix(array([[1.],
                            [2.],
                            [3.]]))
        y = asmatrix(array([[3., 2., 1.]]))

        expected = [[4., 2., 2.],
                    [5., 7., 4.],
                    [9., 7., 3.]]
        self.assertListEqual(ger(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_as_matrices_no_A(self):
        x = asmatrix(array([[1.],
                            [2.],
                            [3.]]))
        y = asmatrix(array([[3., 2., 1.]]))

        expected = [[3., 2., 1.],
                    [6., 4., 2.],
                    [9., 6., 3.]]
        self.assertListEqual(ger(x, y).tolist(), expected)

    def test_vectors_as_mixed_matrices_and_ndarrays(self):
        x = asmatrix(array([[1.],
                            [2.],
                            [3.]]))
        y = array([[3., 2., 1.]])

        expected = [[3., 2., 1.],
                    [6., 4., 2.],
                    [9., 6., 3.]]
        self.assertListEqual(ger(x, y).tolist(), expected)

    def test_strides_less_than_length(self):
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3., 2., 1.]])

        expected = [[3., 1.],
                    [9., 3.]]
        self.assertListEqual(ger(x, y, inc_x=2, inc_y=2).tolist(), expected)

    def test_strides_greater_than_length(self):
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3., 2., 1.]])

        expected = [[3.]]
        self.assertListEqual(ger(x, y, inc_x=3, inc_y=3).tolist(), expected)

    def test_unequal_strides(self):
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3., 2., 1.]])

        expected = [[3.],
                    [9.]]
        self.assertListEqual(ger(x, y, inc_x=2, inc_y=3).tolist(), expected)

    def test_float32_dtype(self):
        x = array([[1.],
                   [2.],
                   [3.]], dtype='float32')
        y = array([[3., 2., 1.]], dtype='float32')

        expected = [[3., 2., 1.],
                    [6., 4., 2.],
                    [9., 6., 3.]]
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(y.dtype, 'float32')
        self.assertListEqual(ger(x, y).tolist(), expected)

    def test_float64_dtype(self):
        x = array([[1.],
                   [2.],
                   [3.]], dtype='float64')
        y = array([[3., 2., 1.]], dtype='float64')

        expected = [[3., 2., 1.],
                    [6., 4., 2.],
                    [9., 6., 3.]]
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float64')
        self.assertListEqual(ger(x, y).tolist(), expected)

    def test_alpha(self):
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3., 2., 1.]])

        expected = [[4.5, 3., 1.5],
                    [9., 6., 3.],
                    [13.5, 9., 4.5]]
        self.assertListEqual(ger(x, y, alpha=1.5).tolist(), expected)

    def test_not_numpy_with_list_for_x_raises_ValueError(self):
        x = [[1.],
             [2.],
             [3.]]
        y = array([[3., 2., 1.]])
        self.assertRaises(ValueError, ger, x, y)

    def test_not_numpy_with_list_for_y_raises_ValueError(self):
        x = array([[1.],
                   [2.],
                   [3.]])
        y = [[3., 2., 1.]]
        self.assertRaises(ValueError, ger, x, y)

    def test_not_numpy_with_list_for_A_raises_ValueError(self):
        A = [[1., 0., 1.],
             [-1., 3., 2.],
             [0., 1., 0.]]
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3., 2., 1.]])
        self.assertRaises(ValueError, ger, x, y, A)

    def test_not_numpy_with_scalar_for_x_raises_ValueError(self):
        x = 1.
        y = array([[3.]])
        self.assertRaises(ValueError, ger, x, y)

    def test_not_numpy_with_scalar_for_y_raises_ValueError(self):
        x = array([[1.]])
        y = 2.
        self.assertRaises(ValueError, ger, x, y)

    def test_not_2d_numpy_with_1d_for_x_raises_ValueError(self):
        x = array([1., 2., 3.])
        y = array([[3., 2., 1.]])
        self.assertRaises(ValueError, ger, x, y)

    def test_not_2d_numpy_with_1d_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = array([3., 2., 1.])
        self.assertRaises(ValueError, ger, x, y)

    def test_not_2d_numpy_with_3d_for_x_raises_ValueError(self):
        x = array([[[1.],
                    [2.],
                    [3.]]], ndmin=3)
        y = array([[3., 2., 1.]])
        self.assertRaises(ValueError, ger, x, y)

    def test_not_2d_numpy_with_3d_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]])
        y = array([[[3.],
                    [2.],
                    [1.]]], ndmin=3)
        self.assertRaises(ValueError, ger, x, y)

    def test_not_vector_raises_ValueError(self):
        x = array([[1., 2.],
                   [3., 4.]])
        y = array([[1., 2.],
                   [3., 4.]])
        self.assertRaises(ValueError, ger, x, y)

    def test_mixed_dtypes_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='float32')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, ger, x, y)

    def test_integer_dtype_for_both_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='int')
        y = array([[3., 2., 1.]], dtype='int')
        self.assertEqual(x.dtype, 'int')
        self.assertEqual(y.dtype, 'int')
        self.assertRaises(ValueError, ger, x, y)

    def test_integer_dtype_for_x_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='int')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'int')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, ger, x, y)

    def test_integer_dtype_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='float64')
        y = array([[3, 2., 1.]], dtype='int')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'int')
        self.assertRaises(ValueError, ger, x, y)

    def test_integer_dtype_for_A_raises_ValueError(self):
        A = array([[1., 0., 1.],
                   [-1., 3., 2.],
                   [0., 1., 0.]], dtype='int')
        x = array([[1., 2., 3.]], dtype='float64')
        y = array([[3, 2., 1.]], dtype='float64')
        self.assertEqual(A.dtype, 'int')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, ger, x, y, A)

    def test_complex_dtype_for_both_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='complex')
        y = array([[3., 2., 1.]], dtype='complex')
        self.assertEqual(x.dtype, 'complex')
        self.assertEqual(y.dtype, 'complex')
        self.assertRaises(ValueError, ger, x, y)

    def test_complex_dtype_for_x_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='complex')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(x.dtype, 'complex')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, ger, x, y)

    def test_complex_dtype_for_y_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='float64')
        y = array([[3., 2., 1.]], dtype='complex')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'complex')
        self.assertRaises(ValueError, ger, x, y)

    def test_complex_dtype_for_A_raises_ValueError(self):
        A = array([[1., 0., 1.],
                   [-1., 3., 2.],
                   [0., 1., 0.]], dtype='complex')
        x = array([[1., 2., 3.]], dtype='float64')
        y = array([[3., 2., 1.]], dtype='float64')
        self.assertEqual(A.dtype, 'complex')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, ger, x, y, A)