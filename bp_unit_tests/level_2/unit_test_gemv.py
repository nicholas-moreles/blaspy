"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import gemv
from numpy import array, asmatrix
from unittest import TestCase


class TestGemv(TestCase):

    def test_scalars_as_ndarray_provide_y(self):
        A = array([[1.]])
        x = array([[2.]])
        y = array([[3.]])
        self.assertEqual(gemv(A, x, y), 5)
        self.assertEqual(y, 5)

    def test_scalars_as_ndarray_no_y(self):
        A = array([[1.]])
        x = array([[2.]])
        self.assertEqual(gemv(A, x), 2)

    def test_row_col_scalar_as_ndarray_provide_y(self):
        A = array([[1., 2., 3.]])
        x = array([[3.], [2.], [1.]])
        y = array([[4.]])
        self.assertEqual(gemv(A, x, y), 14)
        self.assertEqual(y, 14)

    def test_row_col_scalar_as_ndarray_no_y(self):
        A = array([[1., 2., 3.]])
        x = array([[3.], [2.], [1.]])
        self.assertEqual(gemv(A, x), 10)

    def test_row_row_scalar_as_ndarray_provide_y(self):
        A = array([[1., 2., 3.]])
        x = array([[3., 2., 1.]])
        y = array([[4.]])
        self.assertEqual(gemv(A, x, y), 14)
        self.assertEqual(y, 14)

    def test_row_row_scalar_as_ndarray_no_y(self):
        A = array([[1., 2., 3.]])
        x = array([[3., 2., 1.]])
        self.assertEqual(gemv(A, x), 10)

    def test_col_scalar_col_as_ndarray_provide_y(self):
        A = array([[1.], [2.], [3.]])
        x = array([[3.]])
        y = array([[4.], [5.], [6.]])

        expected = [[7.], [11.], [15.]]
        self.assertListEqual(gemv(A, x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_col_scalar_row_as_ndarray_provide_y(self):
        A = array([[1.], [2.], [3.]])
        x = array([[3.]])
        y = array([[4., 5., 6.]])

        expected = [[7., 11., 15.]]
        self.assertListEqual(gemv(A, x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_col_scalar_as_ndarray_no_y(self):
        A = array([[1.], [2.], [3.]])
        x = array([[3.]])

        expected = [[3., 6., 9.]]  # scalar x is treated as row vector when creating similar zero vector
        self.assertListEqual(gemv(A, x).tolist(), expected)

    def test_matrix_row_row_as_ndarray_provide_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1., 2.]])
        y = array([[3., 4.]])

        expected = [[8., 15.]]
        self.assertListEqual(gemv(A, x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_matrix_row_row_as_ndarray_no_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1., 2.]])

        expected = [[5., 11.]]
        self.assertListEqual(gemv(A, x).tolist(), expected)

    def test_matrix_row_col_as_ndarray_provide_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1., 2.]])
        y = array([[3.], [4.]])

        expected = [[8.], [15.]]
        self.assertListEqual(gemv(A, x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_matrix_col_row_as_ndarray_provide_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3., 4.]])

        expected = [[8., 15.]]
        self.assertListEqual(gemv(A, x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_matrix_col_col_as_ndarray_provide_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])

        expected = [[8.], [15.]]
        self.assertListEqual(gemv(A, x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_matrix_col_col_as_ndarray_no_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])

        expected = [[5.], [11.]]
        self.assertListEqual(gemv(A, x).tolist(), expected)

    def test_alpha(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])
        alpha = 2.5

        expected = [[15.5], [31.5]]
        self.assertListEqual(gemv(A, x, y, alpha=alpha).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_beta(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])
        beta = -1.5

        expected = [[0.5], [5.]]
        self.assertListEqual(gemv(A, x, y, beta=beta).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_alpha_and_beta(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])
        alpha = 2.5
        beta = -1.5

        expected = [[8.], [21.5]]
        self.assertListEqual(gemv(A, x, y, alpha=alpha, beta=beta).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_beta_ignored_if_no_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        beta = 1.5

        expected = [[5.], [11.]]
        self.assertListEqual(gemv(A, x, beta=beta).tolist(), expected)

    def test_trans_a_lowercase(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])

        expected = [[10.], [14.]]
        self.assertListEqual(gemv(A, x, y, trans_a='t').tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_trans_a_uppercase(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])

        expected = [[10.], [14.]]
        self.assertListEqual(gemv(A, x, y, trans_a='T').tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_matrix_col_col_as_matrix_all_provide_y(self):
        A = asmatrix(array([[1., 2.], [3., 4.]]))
        x = asmatrix(array([[1.], [2.]]))
        y = asmatrix(array([[3.], [4.]]))

        expected = [[8.], [15.]]
        self.assertListEqual(gemv(A, x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_matrix_col_col_as_matrix_all_no_y(self):
        A = asmatrix(array([[1., 2.], [3., 4.]]))
        x = asmatrix(array([[1.], [2.]]))

        expected = [[5.], [11.]]
        self.assertListEqual(gemv(A, x).tolist(), expected)

    def test_matrix_col_col_as_matrix_mixed_provide_y(self):
        A = asmatrix(array([[1., 2.], [3., 4.]]))
        x = array([[1.], [2.]])
        y = asmatrix(array([[3.], [4.]]))

        expected = [[8.], [15.]]
        self.assertListEqual(gemv(A, x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_matrix_col_col_as_matrix_mixed_no_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = asmatrix(array([[1.], [2.]]))

        expected = [[5.], [11.]]
        self.assertListEqual(gemv(A, x).tolist(), expected)

    def test_strides_less_than_length_provide_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.], [3.], [4.]])
        y = array([[3.], [4.], [5.], [6.]])

        expected = [[10.], [4.], [20.], [6.]]
        self.assertListEqual(gemv(A, x, y, inc_x=2, inc_y=2).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_strides_less_than_length_no_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.], [3.], [4.]])

        expected = [[7.], [15.]]
        self.assertListEqual(gemv(A, x, inc_x=2).tolist(), expected)

    def test_inc_y_ignored_if_no_y(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.], [3.], [4.]])

        expected = [[7.], [15.]]
        self.assertListEqual(gemv(A, x, inc_x=2, inc_y=2).tolist(), expected)

    def test_strides_greater_than_length_provide_y(self):
        A = array([[3.]])
        x = array([[1.], [2.], [3.], [4.]])
        y = array([[3.], [4.], [5.], [6.]])

        expected = [[6.], [4.], [5.], [6.]]
        self.assertListEqual(gemv(A, x, y, inc_x=4, inc_y=4).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_unequal_strides(self):
        A = array([[1.], [3.]])
        x = array([[1.], [2.], [3.], [4.]])
        y = array([[3.], [4.], [5.], [6.]])

        expected = [[4.], [4.], [8.], [6.]]
        self.assertListEqual(gemv(A, x, y, inc_x=4, inc_y=2).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_float32_dtype(self):
        A = array([[1., 2.], [3., 4.]], dtype='float32')
        x = array([[1.], [2.]], dtype='float32')
        y = array([[3.], [4.]], dtype='float32')
        self.assertEqual(A.dtype, 'float32')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(y.dtype, 'float32')

        expected = [[8.], [15.]]
        self.assertListEqual(gemv(A, x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_float64_dtype(self):
        A = array([[1., 2.], [3., 4.]], dtype='float64')
        x = array([[1.], [2.]], dtype='float64')
        y = array([[3.], [4.]], dtype='float64')
        self.assertEqual(A.dtype, 'float64')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float64')

        expected = [[8.], [15.]]
        self.assertListEqual(gemv(A, x, y).tolist(), expected)
        self.assertListEqual(y.tolist(), expected)

    def test_not_numpy_with_list_for_A_raises_ValueError(self):
        A = [[1., 2.], [3., 4.]]
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_numpy_with_list_for_x_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = [[1.], [2.]]
        y = array([[3.], [4.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_numpy_with_list_for_y_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = [[3.], [4.]]
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_numpy_with_scalar_for_A_raises_ValueError(self):
        A = 1.
        x = array([[1.]])
        y = array([[3.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_numpy_with_scalar_for_x_raises_ValueError(self):
        A = array([[1.]])
        x = 1.
        y = array([[3.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_numpy_with_scalar_for_y_raises_ValueError(self):
        A = array([[1.]])
        x = array([[1.]])
        y = 3.
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_2d_numpy_with_1d_for_A_raises_ValueError(self):
        A = array([1., 2.])
        x = array([[1.], [2.]])
        y = array([[3.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_2d_numpy_with_1d_for_x_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([1., 2.])
        y = array([[3., 4.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_2d_numpy_with_1d_for_y_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1., 2.]])
        y = array([3., 4.])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_2d_numpy_with_3d_for_A_raises_ValueError(self):
        A = array([[[1., 2.]]])
        x = array([[1.], [2.]])
        y = array([[3.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_2d_numpy_with_3d_for_x_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[[1., 2.]]])
        y = array([[3., 4.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_not_2d_numpy_with_3d_for_y_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1., 2.]])
        y = array([[[3., 4.]]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_nonconforming_x_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.], [3.]])
        y = array([[3.], [4.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_nonconforming_y_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    # used in the following two unit tests, so we want to show this functionality works correctly first
    def test_trans_conforms_correctly(self):
        A = array([[1., 2., 3.], [3., 4., 5.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.], [5.]])

        expected = [[10.], [14.], [18.]]
        self.assertListEqual(gemv(A, x, y, trans_a='t').tolist(), expected)

    def test_nonconforming_x_with_trans_raises_ValueError(self):
        A = array([[1., 2., 3.], [3., 4., 5.]])
        x = array([[1.], [2.], [3.]])
        y = array([[3.], [4.], [5.]])
        self.assertRaises(ValueError, gemv, A, x, y, 't')

    def test_nonconforming_y_with_trans_raises_ValueError(self):
        A = array([[1., 2., 3.], [3., 4., 5.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])
        self.assertRaises(ValueError, gemv, A, x, y, 't')

    def test_nonconforming_x_with_strides_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])
        self.assertRaises(ValueError, gemv, A, x, y, 'n', 1., 1., None, 2)

    def test_nonconforming_y_with_strides_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])
        self.assertRaises(ValueError, gemv, A, x, y, 'n', 1., 1., None, 1, 2)

    def test_not_vector_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1., 2.], [3., 4.]])
        y = array([[1., 2.], [3., 4.]])
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_mixed_dtypes_A_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]], dtype='float32')
        x = array([[1.], [2.]], dtype='float64')
        y = array([[3.], [4.]], dtype='float64')
        self.assertEqual(A.dtype, 'float32')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_mixed_dtypes_x_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]], dtype='float64')
        x = array([[1.], [2.]], dtype='float32')
        y = array([[3.], [4.]], dtype='float64')
        self.assertEqual(A.dtype, 'float64')
        self.assertEqual(x.dtype, 'float32')
        self.assertEqual(y.dtype, 'float64')
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_mixed_dtypes_y_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]], dtype='float64')
        x = array([[1.], [2.]], dtype='float64')
        y = array([[3.], [4.]], dtype='float32')
        self.assertEqual(A.dtype, 'float64')
        self.assertEqual(x.dtype, 'float64')
        self.assertEqual(y.dtype, 'float32')
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_integer_dtype_for_all_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]], dtype='int')
        x = array([[1.], [2.]], dtype='int')
        y = array([[3.], [4.]], dtype='int')
        self.assertEqual(A.dtype, 'int')
        self.assertEqual(x.dtype, 'int')
        self.assertEqual(y.dtype, 'int')
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_complex_dtype_for_all_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]], dtype='complex')
        x = array([[1.], [2.]], dtype='complex')
        y = array([[3.], [4.]], dtype='complex')
        self.assertEqual(A.dtype, 'complex')
        self.assertEqual(x.dtype, 'complex')
        self.assertEqual(y.dtype, 'complex')
        self.assertRaises(ValueError, gemv, A, x, y)

    def test_uppercase_trans_a(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])

        expected = [[10.], [14.]]
        self.assertListEqual(gemv(A, x, y, 'T').tolist(), expected)

    def test_invalid_values_for_trans_a_raises_ValueError(self):
        A = array([[1., 2.], [3., 4.]])
        x = array([[1.], [2.]])
        y = array([[3.], [4.]])
        self.assertRaises(ValueError, gemv, A, x, y, 'no')