"""

    Copyright (c) 2014-2015, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import trsv
from numpy import array, asmatrix
from string import ascii_letters
from unittest import TestCase


class TestTrsv(TestCase):

    def test_scalars_as_ndarray(self):
        A = array([[1.]])
        b = array([[2.]])
        self.assertEqual(trsv(A, b), 2)
        self.assertEqual(b, 2)

    def test_row_as_ndarray(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[5., 6.]])

        expected = [[1., 2.]]
        self.assertListEqual(trsv(A, b).tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_col_as_ndarray(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[5.],
                   [6.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b).tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_lower_triangle_ignored_with_uplo_u(self):
        A = array([[1., 2.],
                   [-100., 3.]])
        b = array([[5.],
                   [6.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, uplo='u').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_lower_triangle_ignored_with_uplo_U(self):
        A = array([[1., 2.],
                   [-100., 3.]])
        b = array([[5.],
                   [6.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, uplo='U').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_upper_triangle_ignored_with_uplo_l(self):
        A = array([[1., 55.],
                   [2., 3.]])
        b = array([[1.],
                   [8.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, uplo='l').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_upper_triangle_ignored_with_uplo_L(self):
        A = array([[1., 55.],
                   [2., 3.]])
        b = array([[1.],
                   [8.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, uplo='L').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_A_not_transposed_with_trans_a_n(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[5.],
                   [6.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, trans_a='n').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_A_not_transposed_with_trans_a_N(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[5.],
                   [6.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, trans_a='N').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_A_transposed_with_trans_a_t(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[1.],
                   [8.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, trans_a='t').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_A_transposed_with_trans_a_T(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[1.],
                   [8.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, trans_a='T').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_A_non_unit_diag_with_diag_n(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[5.],
                   [6.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, diag='n').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_A_non_unit_diag_with_diag_N(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[5.],
                   [6.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, diag='N').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_A_unit_diag_with_diag_u(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[5.],
                   [2.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, diag='u').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_A_unit_diag_with_diag_U(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[5.],
                   [2.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, diag='U').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_uplo_l_and_trans_a_t(self):
        A = array([[1., -100.],
                   [2., 3.]])
        b = array([[5.],
                   [6.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, uplo='l', trans_a='t').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_uplo_l_and_trans_a_t_and_diag_u(self):
        A = array([[1., -100.],
                   [2., 3.]])
        b = array([[5.],
                   [2.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b, uplo='l', trans_a='t', diag='u').tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_as_matrix_all(self):
        A = asmatrix(array([[1., 2.],
                            [0., 3.]]))
        b = asmatrix(array([[5.],
                            [6.]]))

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b).tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_as_matrix_mixed(self):
        A = asmatrix(array([[1., 2.],
                            [0., 3.]]))
        b = array([[5.],
                   [6.]])

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b).tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_stride_less_than_length(self):
        A = array([[1., 2.],
                   [0., 3.]])
        b = array([[5.],
                   [3.],
                   [6.],
                   [4.]])

        expected = [[1.],
                    [3.],
                    [2.],
                    [4.]]
        self.assertListEqual(trsv(A, b, inc_b=2).tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_stride_greater_than_length(self):
        A = array([[1.]])
        b = array([[5.],
                   [2.],
                   [3.],
                   [4.]])

        expected = [[5.],
                    [2.],
                    [3.],
                    [4.]]
        self.assertListEqual(trsv(A, b, inc_b=4).tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_float32_dtype(self):
        A = array([[1., 2.],
                   [0., 3.]], dtype='float32')
        b = array([[5.],
                   [6.]], dtype='float32')
        self.assertEqual(A.dtype, 'float32')
        self.assertEqual(b.dtype, 'float32')

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b).tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_float64_dtype(self):
        A = array([[1., 2.],
                   [0., 3.]], dtype='float64')
        b = array([[5.],
                   [6.]], dtype='float64')
        self.assertEqual(A.dtype, 'float64')
        self.assertEqual(b.dtype, 'float64')

        expected = [[1.],
                    [2.]]
        self.assertListEqual(trsv(A, b).tolist(), expected)
        self.assertListEqual(b.tolist(), expected)

    def test_not_numpy_with_list_for_A_raises_ValueError(self):
        A = [[1., 2.],
             [3., 4.]]
        b = array([[1.],
                   [2.]])
        self.assertRaises(ValueError, trsv, A, b)

    def test_not_numpy_with_list_for_b_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        b = [[1.],
             [2.]]
        self.assertRaises(ValueError, trsv, A, b)

    def test_not_numpy_with_scalar_for_A_raises_ValueError(self):
        A = 1.
        b = array([[1.]])
        self.assertRaises(ValueError, trsv, A, b)

    def test_not_numpy_with_scalar_for_b_raises_ValueError(self):
        A = array([[1.]])
        b = 1.
        self.assertRaises(ValueError, trsv, A, b)

    def test_not_2d_numpy_with_1d_for_A_raises_ValueError(self):
        A = array([1., 2., 2., 1.])
        b = array([[1.],
                   [2.]])
        self.assertRaises(ValueError, trsv, A, b)

    def test_not_2d_numpy_with_1d_for_b_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        b = array([1., 2.])
        self.assertRaises(ValueError, trsv, A, b)

    def test_not_2d_numpy_with_3d_for_A_raises_ValueError(self):
        A = array([[[1., 2.],
                    [2., 3.]]], ndmin=3)
        b = array([[1.],
                   [2.]])
        self.assertRaises(ValueError, trsv, A, b)

    def test_not_2d_numpy_with_3d_for_b_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        b = array([[[1., 2.]]], ndmin=3)
        self.assertRaises(ValueError, trsv, A, b)

    def test_nonconforming_b_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        b = array([[1.],
                   [2.],
                   [3.]])
        self.assertRaises(ValueError, trsv, A, b)

    def test_non_square_A_raises_ValueError(self):
        A = array([[1., 2., 3.],
                   [2., 3., 4.]])
        b = array([[1.],
                   [2.],
                   [3.]])
        self.assertRaises(ValueError, trsv, A, b)

    def test_nonconforming_b_with_strides_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        b = array([[1.],
                   [2.]])
        self.assertRaises(ValueError, trsv, A, b, 'u', 'n', 'n', None, 2)

    def test_not_vector_raises_ValueError(self):
        A = array([[1., 2.],
                   [2., 3.]])
        b = array([[1., 2.],
                   [2., 3.]])
        self.assertRaises(ValueError, trsv, A, b)

    def test_mixed_dtypes1_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='float32')
        b = array([[1.],
                   [2.]], dtype='float64')
        self.assertEqual(A.dtype, 'float32')
        self.assertEqual(b.dtype, 'float64')
        self.assertRaises(ValueError, trsv, A, b)

    def test_mixed_dtypes2_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='float64')
        b = array([[1.],
                   [2.]], dtype='float32')
        self.assertEqual(A.dtype, 'float64')
        self.assertEqual(b.dtype, 'float32')
        self.assertRaises(ValueError, trsv, A, b)

    def test_integer_dtype_for_all_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='int')
        b = array([[1.],
                   [2.]], dtype='int')
        self.assertEqual(A.dtype, 'int')
        self.assertEqual(b.dtype, 'int')
        self.assertRaises(ValueError, trsv, A, b)

    def test_complex_dtype_for_all_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]], dtype='complex')
        b = array([[1.],
                   [2.]], dtype='complex')
        self.assertEqual(A.dtype, 'complex')
        self.assertEqual(b.dtype, 'complex')
        self.assertRaises(ValueError, trsv, A, b)

    def test_invalid_values_for_uplo_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        b = array([[1.],
                   [2.]])
        for char in ascii_letters:
            if char not in ('u', 'U', 'l', 'L'):
                self.assertRaises(ValueError, trsv, A, b, char)

    def test_invalid_values_for_trans_a_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        b = array([[1.],
                   [2.]])
        for char in ascii_letters:
            if char not in ('n', 'N', 't', 'T'):
                self.assertRaises(ValueError, trsv, A, b, 'u', char)

    def test_invalid_values_for_diag_raises_ValueError(self):
        A = array([[1., 2.],
                   [3., 4.]])
        b = array([[1.],
                   [2.]])
        for char in ascii_letters:
            if char not in ('n', 'N', 'u', 'U'):
                self.assertRaises(ValueError, trsv, A, b, 'u', 'n', char)
