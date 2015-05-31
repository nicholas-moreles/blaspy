"""

    Copyright (c) 2014-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import syr
from numpy import array, asmatrix
from unittest import TestCase


class TestSyr(TestCase):

    def test_scalars_as_ndarray_provide_A(self):
        A = array([[1.]])
        x = array([[2.]])
        self.assertEqual(syr(x, A), 5)
        self.assertEqual(A, 5)

    def test_scalars_as_ndarray_no_A(self):
        x = array([[2.]])
        self.assertEqual(syr(x), 4)

    def test_row_as_ndarray_provide_A(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1., 2.]])

        expected = [[2., 4.],
                    [2., 7.]]  # syr only updates one triangle of A, upper by default
        self.assertListEqual(syr(x, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_row_as_ndarray_no_A(self):
        x = array([[1., 2.]])

        expected = [[1., 2.],
                    [0., 4.]]
        self.assertListEqual(syr(x).tolist(), expected)

    def test_col_as_ndarray_provide_A(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])

        expected = [[2., 4.],
                    [2., 7.]]
        self.assertListEqual(syr(x, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_col_as_ndarray_no_A(self):
        x = array([[1.],
                   [2.]])

        expected = [[1., 2.],
                    [0., 4.]]
        self.assertListEqual(syr(x).tolist(), expected)

    def test_lower_triangle_ignored_with_uplo_u(self):
        A = array([[1., 2.],
                   [-100., 3.]])
        x = array([[1.],
                   [2.]])

        expected = [[2., 4.],
                    [-100., 7.]]
        self.assertListEqual(syr(x, A, uplo='u').tolist(), expected)

    def test_lower_triangle_ignored_with_uplo_U(self):
        A = array([[1., 2.],
                   [-100., 3.]])
        x = array([[1.],
                   [2.]])

        expected = [[2., 4.],
                    [-100., 7.]]
        self.assertListEqual(syr(x, A, uplo='U').tolist(), expected)

    def test_upper_triangle_ignored_with_uplo_l(self):
        A = array([[1., 55.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])

        expected = [[2., 55.],
                    [4., 7.]]
        self.assertListEqual(syr(x, A, uplo='l').tolist(), expected)

    def test_upper_triangle_ignored_with_uplo_L(self):
        A = array([[1., 55.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])

        expected = [[2., 55.],
                    [4., 7.]]
        self.assertListEqual(syr(x, A, uplo='L').tolist(), expected)

    def test_alpha(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])
        alpha = 2.5

        expected = [[3.5, 7.],
                    [2., 13.]]
        self.assertListEqual(syr(x, A, alpha=alpha).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_as_matrix_all_provide_A(self):
        A = asmatrix(array([[1., 2.],
                            [2., 3.]]))
        x = asmatrix(array([[1.],
                            [2.]]))

        expected = [[2., 4.],
                    [2., 7.]]
        self.assertListEqual(syr(x, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_as_matrix_all_no_A(self):
        x = asmatrix(array([[1.],
                            [2.]]))

        expected = [[1., 2.],
                    [0., 4.]]
        self.assertListEqual(syr(x).tolist(), expected)

    def test_as_matrix_mixed(self):
        A = asmatrix(array([[1., 2.],
                            [2., 3.]]))
        x = array([[1.],
                   [2.]])

        expected = [[2., 4.],
                    [2., 7.]]
        self.assertListEqual(syr(x, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_stride_less_than_length_provide_A(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.],
                   [3.],
                   [4.]])

        expected = [[2., 5.],
                    [2., 12.]]
        self.assertListEqual(syr(x, A, inc_x=2).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_stride_less_than_length_no_A(self):
        x = array([[1.],
                   [2.],
                   [3.],
                   [4.]])

        expected = [[1., 3.],
                    [0., 9.]]
        self.assertListEqual(syr(x, inc_x=2).tolist(), expected)

    def test_stride_greater_than_length_provide_A(self):
        A = array([[3.]])
        x = array([[1.],
                   [2.],
                   [3.],
                   [4.]])

        expected = [[4.]]
        self.assertListEqual(syr(x, A, inc_x=4).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_stride_greater_than_length_no_A(self):
        x = array([[1.],
                   [2.],
                   [3.],
                   [4.]])

        expected = [[1.]]
        self.assertListEqual(syr(x, inc_x=4).tolist(), expected)

    def test_float32_dtype(self):
        A = array([[1., 2.],
                   [2., 3.]], dtype='float32')
        x = array([[1.],
                   [2.]], dtype='float32')
        self.assertEqual(A.dtype, 'float32')
        self.assertEqual(x.dtype, 'float32')

        expected = [[2., 4.],
                    [2., 7.]]
        self.assertListEqual(syr(x, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_float64_dtype(self):
        A = array([[1., 2.],
                   [2., 3.]], dtype='float64')
        x = array([[1.],
                   [2.]], dtype='float64')
        self.assertEqual(A.dtype, 'float64')
        self.assertEqual(x.dtype, 'float64')

        expected = [[2., 4.],
                    [2., 7.]]
        self.assertListEqual(syr(x, A).tolist(), expected)
        self.assertListEqual(A.tolist(), expected)

    def test_not_numpy_with_list_for_A_raises_ValueError(self):
        A = [[1., 2.],
             [3., 4.]]
        x = array([[1.],
                   [2.]])
        self.assertRaises(ValueError, syr, x, A)

    def test_not_numpy_with_list_for_x_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        x = [[1.],
             [2.]]
        self.assertRaises(ValueError, syr, x, A)

    def test_not_numpy_with_scalar_for_A_raises_ValueError(self):
        A = 1.
        x = array([[1.]])
        self.assertRaises(ValueError, syr, x, A)

    def test_not_numpy_with_scalar_for_x_raises_ValueError(self):
        A = array([[1.]])
        x = 1.
        self.assertRaises(ValueError, syr, x, A)

    def test_not_2d_numpy_with_1d_for_A_raises_ValueError(self):
        A = array([1., 2., 2., 1.])
        x = array([[1.],
                   [2.]])
        self.assertRaises(ValueError, syr, x, A)

    def test_not_2d_numpy_with_1d_for_x_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        x = array([1., 2.])
        self.assertRaises(ValueError, syr, x, A)

    def test_not_2d_numpy_with_3d_for_A_raises_ValueError(self):
        A = array([[[1., 2.],
                    [2., 3.]]], ndmin=3)
        x = array([[1.],
                   [2.]])
        self.assertRaises(ValueError, syr, x, A)

    def test_not_2d_numpy_with_3d_for_x_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[[1., 2.]]], ndmin=3)
        self.assertRaises(ValueError, syr, x, A)

    def test_nonconforming_x_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.],
                   [3.]])
        self.assertRaises(ValueError, syr, x, A)

    def test_non_square_A_raises_ValueError(self):
        A = array([[1., 2., 3.],
                   [2., 3., 4.]])
        x = array([[1.],
                   [2.],
                   [3.]])
        self.assertRaises(ValueError, syr, x, A)

    def test_nonconforming_x_with_strides_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1.],
                   [2.]])
        self.assertRaises(ValueError, syr, x, A, 'u', 1., None, 2)

    def test_not_vector_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        x = array([[1., 2.],
                   [2., 3.]])
        self.assertRaises(ValueError, syr, x, A)

    def test_mixed_dtypes1_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='float32')
        x = array([[1.],
                   [2.]], dtype='float64')
        self.assertEqual(A.dtype, 'float32')
        self.assertEqual(x.dtype, 'float64')
        self.assertRaises(ValueError, syr, x, A)

    def test_mixed_dtypes2_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='float64')
        x = array([[1.],
                   [2.]], dtype='float32')
        self.assertEqual(A.dtype, 'float64')
        self.assertEqual(x.dtype, 'float32')
        self.assertRaises(ValueError, syr, x, A)

    def test_integer_dtype_for_all_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='int')
        x = array([[1.],
                   [2.]], dtype='int')
        self.assertEqual(A.dtype, 'int')
        self.assertEqual(x.dtype, 'int')
        self.assertRaises(ValueError, syr, x, A)

    def test_complex_dtype_for_all_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='complex')
        x = array([[1.],
                   [2.]], dtype='complex')
        self.assertEqual(A.dtype, 'complex')
        self.assertEqual(x.dtype, 'complex')
        self.assertRaises(ValueError, syr, x, A)

    def test_invalid_value_for_uplo_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        x = array([[1.],
                   [2.]])
        self.assertRaises(ValueError, syr, x, A, 'a')