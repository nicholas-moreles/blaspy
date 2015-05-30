"""

    Copyright (c) 2014-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import syr2
from numpy import array, asmatrix
from unittest import TestCase


class TestSyr2(TestCase):

    def test_scalars_as_ndarray_provide_A(self):
        A = array([[1.]])
        x = array([[2.]])
        y = array([[3.]])
        self.assertEqual(syr2(x, y, A), 13)
        self.assertEqual(A, 13)

    def test_scalars_as_ndarray_no_A(self):
        x = array([[2.]])
        y = array([[3.]])
        self.assertEqual(syr2(x, y), 12)

    def test_row_row_as_ndarray_provide_A(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1., 2.]])
        y = array([[3., 4.]])

        expected = [[7., 12.],
                    [2., 19.]]
        self.assertListEqual(syr2(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_row_row_as_ndarray_no_A(self):
        x = array([[1., 2.]])
        y = array([[3., 4.]])

        expected = [[6., 10.],
                    [0., 16.]]
        self.assertListEqual(syr2(x, y).tolist(), expected)

    def test_matrix_row_col_as_ndarray_provide_A(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1., 2.]])
        y = array([[3.],
                   [4.]])

        expected = [[7., 12.],
                    [2., 19.]]
        self.assertListEqual(syr2(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_matrix_col_row_as_ndarray_provide_A(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])
        y = array([[3., 4.]])

        expected = [[7., 12.],
                    [2., 19.]]
        self.assertListEqual(syr2(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_matrix_col_col_as_ndarray_provide_y(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])

        expected = [[7., 12.],
                    [2., 19.]]
        self.assertListEqual(syr2(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_matrix_col_col_as_ndarray_no_A(self):
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])

        expected = [[6., 10.],
                    [0., 16.]]
        self.assertListEqual(syr2(x, y).tolist(), expected)

    def test_lower_triangle_ignored_with_uplo_u(self):
        A = array([[1., 2.],
                   [-100., 3.]])
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])

        expected = [[7., 12.],
                    [-100., 19.]]
        self.assertListEqual(syr2(x, y, A, uplo='u').tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_lower_triangle_ignored_with_uplo_U(self):
        A = array([[1., 2.],
                   [-100., 3.]])
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])

        expected = [[7., 12.],
                    [-100., 19.]]
        self.assertListEqual(syr2(x, y, A, uplo='U').tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_upper_triangle_ignored_with_uplo_l(self):
        A = array([[1., 55.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])

        expected = [[7., 55.],
                    [12., 19.]]
        self.assertListEqual(syr2(x, y, A, uplo='l').tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_upper_triangle_ignored_with_uplo_L(self):
        A = array([[1., 55.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])

        expected = [[7., 55.],
                    [12., 19.]]
        self.assertListEqual(syr2(x, y, A, uplo='L').tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_alpha(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])
        alpha = 2.5

        expected = [[16., 27.],
                    [2., 43.]]
        self.assertListEqual(syr2(x, y, A, alpha=alpha).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_as_matrix_mixed_no_A(self):
        x = asmatrix(array([[1.],
                            [2.]]))
        y = array([[3.],
                   [4.]])

        expected = [[6., 10.],
                    [0., 16.]]
        self.assertListEqual(syr2(x, y).tolist(), expected)

    def test_strides_less_than_length_provide_A(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.],
                   [3.],
                   [4.]])
        y = array([[3.],
                   [4.],
                   [5.],
                   [6.]])

        expected = [[7., 16.],
                    [2., 33.]]
        self.assertListEqual(syr2(x, y, A, inc_x=2, inc_y=2).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_strides_less_than_length_no_A(self):
        x = array([[1.],
                   [2.],
                   [3.],
                   [4.]])
        y = array([[3.],
                   [4.],
                   [5.],
                   [6.]])

        expected = [[6., 14.],
                    [0., 30.]]
        self.assertListEqual(syr2(x, y, inc_x=2, inc_y=2).tolist(), expected)

    def test_strides_greater_than_length_provide_A(self):
        A = array([[3.]])
        x = array([[1.],
                   [2.],
                   [3.],
                   [4.]])
        y = array([[3.],
                   [4.],
                   [5.],
                   [6.]])

        expected = [[9.]]
        self.assertListEqual(syr2(x, y, A, inc_x=4, inc_y=4).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_unequal_strides(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.],
                   [3.],
                   [4.]])
        y = array([[3.],
                   [5.]])

        expected = [[7., 16.],
                    [2., 33.]]
        self.assertListEqual(syr2(x, y, A, inc_x=2, inc_y=1).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_float32_dtype(self):
        A = array([[1., 2.],
                   [2., 3.]], dtype='float32')
        x = array([[1.],
                   [2.]], dtype='float32')
        y = array([[3.],
                   [4.]], dtype='float32')
        self.assertEqual(A.dtype, 'float32')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(y.dtype, 'float32')

        expected = [[7., 12.],
                    [2., 19.]]
        self.assertListEqual(syr2(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_float64_dtype(self):
        A = array([[1., 2.],
                   [2., 3.]], dtype='float64')
        x = array([[1.],
                   [2.]], dtype='float64')
        y = array([[3.],
                   [4.]], dtype='float64')
        self.assertEqual(A.dtype, 'float64')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float64')

        expected = [[7., 12.],
                    [2., 19.]]
        self.assertListEqual(syr2(x, y, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_not_numpy_with_list_for_A_raises_ValueError(self):
        A = [[1., 2.],
             [3., 4.]]
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_numpy_with_list_for_x_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        x = [[1.],
             [2.]]
        y = array([[3.],
                   [4.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_numpy_with_list_for_y_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        x = array([[1.],
                   [2.]])
        y = [[3.],
             [4.]]
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_numpy_with_scalar_for_A_raises_ValueError(self):
        A = 1.
        x = array([[1.]])
        y = array([[3.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_numpy_with_scalar_for_x_raises_ValueError(self):
        A = array([[1.]])
        x = 1.
        y = array([[3.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_numpy_with_scalar_for_y_raises_ValueError(self):
        A = array([[1.]])
        x = array([[1.]])
        y = 3.
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_2d_numpy_with_1d_for_A_raises_ValueError(self):
        A = array([1., 2., 2., 1.])
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_2d_numpy_with_1d_for_x_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        x = array([1., 2.])
        y = array([[3., 4.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_2d_numpy_with_1d_for_y_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        x = array([[1., 2.]])
        y = array([3., 4.])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_2d_numpy_with_3d_for_A_raises_ValueError(self):
        A = array([[[1., 2.],
                    [2., 3.]]], ndmin=3)
        x = array([[1.],
                   [2.]])
        y = array([[3.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_2d_numpy_with_3d_for_x_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[[1., 2.]]], ndmin=3)
        y = array([[3., 4.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_not_2d_numpy_with_3d_for_y_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1., 2.]])
        y = array([[[3., 4.]]], ndmin=3)
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_nonconforming_x_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3.],
                   [4.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_nonconforming_y_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])
        y = array([[3.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_non_square_A_raises_ValueError(self):
        A = array([[1., 2., 3.],
                   [2., 3., 4.]])
        x = array([[1.],
                   [2.],
                   [3.]])
        y = array([[3.],
                   [4.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_nonconforming_x_with_strides_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])
        self.assertRaises(ValueError, syr2, x, y, A, 'u', 1., None, 2)

    def test_nonconforming_y_with_strides_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])
        self.assertRaises(ValueError, syr2, x, y, A, 'u', 1., None, 1, 2)

    def test_not_vector_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1., 2.],
                   [2., 3.]])
        y = array([[1., 2.],
                   [2., 3.]])
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_mixed_dtypes_A_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='float32')
        x = array([[1.],
                   [2.]], dtype='float64')
        y = array([[3.],
                   [4.]], dtype='float64')
        self.assertEqual(A.dtype, 'float32')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_mixed_dtypes_x_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='float64')
        x = array([[1.],
                   [2.]], dtype='float32')
        y = array([[3.],
                   [4.]], dtype='float64')
        self.assertEqual(A.dtype, 'float64')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_mixed_dtypes_y_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='float64')
        x = array([[1.],
                   [2.]], dtype='float64')
        y = array([[3.],
                   [4.]], dtype='float32')
        self.assertEqual(A.dtype, 'float64')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float32')
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_integer_dtype_for_all_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='int')
        x = array([[1.],
                   [2.]], dtype='int')
        y = array([[3.],
                   [4.]], dtype='int')
        self.assertEqual(A.dtype, 'int')
        self.assertEqual(x.dtype, 'int')
        self.assertEqual(y.dtype, 'int')
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_complex_dtype_for_all_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='complex')
        x = array([[1.],
                   [2.]], dtype='complex')
        y = array([[3.],
                   [4.]], dtype='complex')
        self.assertEqual(A.dtype, 'complex')
        self.assertEqual(x.dtype, 'complex')
        self.assertEqual(y.dtype, 'complex')
        self.assertRaises(ValueError, syr2, x, y, A)

    def test_invalid_values_for_uplo_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        x = array([[1.],
                   [2.]])
        y = array([[3.],
                   [4.]])
        self.assertRaises(ValueError, syr2, x, y, A, 'z')