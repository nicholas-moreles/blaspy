"""

    Copyright (c) 2014, The University of Texas at Austin.
    All rights reserved.

    This file is part of BLASpy and is available under the 3-Clause
    BSD License, which can be found in the LICENSE file at the top-level
    directory or at http://opensource.org/licenses/BSD-3-Clause

"""

from blaspy import scal
from numpy import array, asmatrix
from unittest import TestCase


class TestScal(TestCase):

    def test_scalar_as_ndarray(self):
        x = array([[1.]])
        expected = [[2.]]
        self.assertListEqual(scal(2, x).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_alpha_equal_to_zero_with_scalar(self):
        x = array([[1.]])
        expected = [[0.]]
        self.assertListEqual(scal(0, x).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_negative_alpha_with_scalar(self):
        x = array([[1.]])
        expected = [[-1.]]
        self.assertListEqual(scal(-1, x).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_row_vector_as_ndarray(self):
        x = array([[1., 2., 3.]])
        expected = [[2., 4., 6.]]
        self.assertListEqual(scal(2, x).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_two_vector_as_ndarray(self):
        x = array([[1.], [2.], [3.]])
        expected = [[2.], [4.], [6.]]
        self.assertListEqual(scal(2, x).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_vector_with_negatives_elements(self):
        x = array([[-1., -2., 3.]])
        expected = [[-2., -4., 6.]]
        self.assertListEqual(scal(2, x).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_alpha_equal_to_zero_with_vectors(self):
        x = array([[-1., -2., 3.]])
        expected = [[0., 0., 0.]]
        self.assertListEqual(scal(0, x).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_negative_alpha_with_vectors(self):
        x = array([[-1., -2., 3.]])
        expected = [[1.5, 3., -4.5]]
        self.assertListEqual(scal(-1.5, x).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_vector_as_matrix(self):
        x = asmatrix(array([[1., 2., 3.]]))
        expected = [[2., 4., 6.]]
        self.assertListEqual(scal(2, x).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_stride_less_than_length(self):
        x = array([[1., 2., 3.]])
        expected = [[2., 2., 6.]]
        self.assertListEqual(scal(2, x, inc_x=2).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_stride_greater_than_length(self):
        x = array([[1., 2., 3.]])
        expected = [[2., 2., 3.]]
        self.assertListEqual(scal(2, x, inc_x=3).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_float32_dtype(self):
        x = array([[1., 2., 3.]], dtype='float32')
        self.assertEqual(x.dtype, 'float32')
        expected = [[2., 4., 6.]]
        self.assertListEqual(scal(2, x,).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_float64_dtype(self):
        x = array([[1., 2., 3.]], dtype='float64')
        self.assertEqual(x.dtype, 'float64')
        expected = [[2., 4., 6.]]
        self.assertListEqual(scal(2, x,).tolist(), expected)
        self.assertListEqual(x.tolist(), expected)

    def test_not_numpy_with_list_raises_ValueError(self):
        x = [[1., 2., 3.]]
        self.assertRaises(ValueError, scal, 1, x)

    def test_not_numpy_with_scalar_raises_ValueError(self):
        x = 1.
        self.assertRaises(ValueError, scal, 1, x)

    def test_not_2d_numpy_with_1d__raises_ValueError(self):
        x = array([1., 2., 3.])
        self.assertRaises(ValueError, scal, 1, x)

    def test_not_2d_numpy_with_3d_raises_ValueError(self):
        x = array([[[1.], [2.], [3.]]], ndmin=3)
        self.assertRaises(ValueError, scal, 1, x)

    def test_not_vector_raises_ValueError(self):
        x = array([[1., 2.], [3., 4.]])
        self.assertRaises(ValueError, scal, 1, x)

    def test_integer_dtype_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='int')
        self.assertEqual(x.dtype, 'int')
        self.assertRaises(ValueError, scal, 1, x)

    def test_complex_dtype_raises_ValueError(self):
        x = array([[1., 2., 3.]], dtype='complex')
        self.assertEqual(x.dtype, 'complex')
        self.assertRaises(ValueError, scal, 1, x)