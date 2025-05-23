import logging
import os
import numpy as np
import unittest
import subprocess
import sys
from scipy.datasets import face
import matplotlib as mpl
mpl.use('Agg') # use non-gui backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from magiceye_solve.solver import InteractiveSolver

logging.basicConfig(level=logging.DEBUG)

class TestInteractiveSolver(unittest.TestCase):

    def setUp(self) -> None:
        """Create sample float images for testing."""
        # float images, normalization handled in class
        self.test_img_gray: np.ndarray = np.random.rand(50, 60)
        self.test_img_color: np.ndarray = np.random.rand(50, 60, 3)
        # rgba, alpha ignored
        self.test_img_rgba: np.ndarray = np.random.rand(50, 60, 4)
        # uint8 for normalization check
        self.test_img_uint8: np.ndarray = (np.random.rand(50, 60, 3) * 255).astype(np.uint8)

    def test_init(self) -> None:
        """Test initialization, normalization, and default offset calculation."""
        # grayscale
        solver_gray = InteractiveSolver(self.test_img_gray)
        self.assertEqual(solver_gray.image.shape, (50, 60))
        self.assertFalse(solver_gray.color_image)
        self.assertEqual(solver_gray.c, 1)
        self.assertIsInstance(solver_gray.default_offset, int)
        self.assertGreaterEqual(solver_gray.default_offset, 1)

        # rgb
        solver_color = InteractiveSolver(self.test_img_color)
        self.assertEqual(solver_color.image.shape, (50, 60, 3))
        self.assertTrue(solver_color.color_image)
        self.assertEqual(solver_color.c, 3)
        self.assertIsInstance(solver_color.default_offset, int)
        self.assertGreaterEqual(solver_color.default_offset, 1)

        # rgba, alpha removed
        solver_rgba = InteractiveSolver(self.test_img_rgba)
        self.assertEqual(solver_rgba.image.shape, (50, 60, 3))
        self.assertTrue(solver_rgba.color_image)
        self.assertEqual(solver_rgba.c, 3)
        self.assertIsInstance(solver_rgba.default_offset, int)
        self.assertGreaterEqual(solver_rgba.default_offset, 1)

        # uint8 normalization
        solver_uint8 = InteractiveSolver(self.test_img_uint8)
        self.assertEqual(solver_uint8.image.dtype, float)
        self.assertLessEqual(np.max(solver_uint8.image), 1.0)
        self.assertGreaterEqual(np.min(solver_uint8.image), 0.0)
        self.assertEqual(solver_uint8.image.shape, (50, 60, 3))
        self.assertTrue(solver_uint8.color_image)
        self.assertEqual(solver_uint8.c, 3)

    def test_solve_with_valid_offset_gray(self) -> None:
        """Test solving grayscale with a valid user offset."""
        solver = InteractiveSolver(self.test_img_gray)
        user_offset = 10
        # grayscale only supports 'separate'
        solved_img: np.ndarray = solver.solve_with_offset(user_offset, channel_mode='separate')

        expected_width = max(0, solver.n - user_offset)
        self.assertEqual(solved_img.shape, (solver.m, expected_width))
        if expected_width > 0:
             self.assertGreater(np.sum(np.abs(solved_img)), 0)

        # average mode, should match
        solved_avg: np.ndarray = solver.solve_with_offset(user_offset, channel_mode='average')
        self.assertEqual(solved_avg.shape, (solver.m, expected_width))
        if expected_width > 0:
             self.assertGreater(np.sum(np.abs(solved_avg)), 0)
             np.testing.assert_allclose(solved_img, solved_avg)

    def test_solve_with_valid_offset_color_separate(self) -> None:
        """Test solving color with a valid user offset using 'separate' mode."""
        solver = InteractiveSolver(self.test_img_color)
        user_offset = 15
        solved_img: np.ndarray = solver.solve_with_offset(user_offset, channel_mode='separate')

        expected_width_per_channel = max(0, solver.n - user_offset)
        expected_total_width = expected_width_per_channel * solver.c
        self.assertEqual(solved_img.shape, (solver.m, expected_total_width))
        if expected_total_width > 0:
             self.assertGreater(np.sum(np.abs(solved_img)), 0)

    def test_solve_with_valid_offset_color_average(self) -> None:
        """Test solving color with a valid user offset using 'average' mode."""
        solver = InteractiveSolver(self.test_img_color)
        user_offset = 15
        solved_img: np.ndarray = solver.solve_with_offset(user_offset, channel_mode='average')

        expected_width = max(0, solver.n - user_offset)
        self.assertEqual(solved_img.shape, (solver.m, expected_width))
        self.assertEqual(solved_img.ndim, 2)
        if expected_width > 0:
             self.assertGreater(np.sum(np.abs(solved_img)), 0)

    def test_solve_with_zero_offset(self) -> None:
        """Test solving with zero offset (should return empty/zeros)."""
        solver_color = InteractiveSolver(self.test_img_color)
        solved_sep: np.ndarray = solver_color.solve_with_offset(0, channel_mode='separate')
        expected_shape_sep = (solver_color.m, solver_color.n * solver_color.c)
        self.assertEqual(solved_sep.shape, expected_shape_sep)
        self.assertTrue((solved_sep == 0).all())

        solved_avg: np.ndarray = solver_color.solve_with_offset(0, channel_mode='average')
        expected_shape_avg = (solver_color.m, solver_color.n)
        self.assertEqual(solved_avg.shape, expected_shape_avg)
        self.assertTrue((solved_avg == 0).all())

    def test_solve_with_large_offset(self) -> None:
        """Test solving with an offset larger than image width."""
        solver_color = InteractiveSolver(self.test_img_color)
        user_offset = solver_color.n + 10 # Offset larger than width

        solved_sep: np.ndarray = solver_color.solve_with_offset(user_offset, channel_mode='separate')
        expected_shape_sep = (solver_color.m, 0)
        self.assertEqual(solved_sep.shape, expected_shape_sep)

        solved_avg: np.ndarray = solver_color.solve_with_offset(user_offset, channel_mode='average')
        expected_shape_avg = (solver_color.m, 0)
        self.assertEqual(solved_avg.shape, expected_shape_avg)

    def test_solve_with_invalid_mode(self) -> None:
        """Test solving with an invalid channel_mode."""
        solver = InteractiveSolver(self.test_img_color)
        user_offset = 10
        solved_img: np.ndarray = solver.solve_with_offset(user_offset, channel_mode='invalid_mode')

        # should return empty array matching 'separate' mode shape
        expected_width_per_channel = max(0, solver.n - user_offset)
        expected_total_width = expected_width_per_channel * solver.c
        expected_shape = (solver.m, expected_total_width)

        self.assertEqual(solved_img.shape, expected_shape)
        self.assertTrue((solved_img == 0).all())


if __name__ == '__main__':
    unittest.main()
