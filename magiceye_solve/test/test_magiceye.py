import logging
import os
import numpy as np
import unittest
import subprocess
import sys
from magiceye_solve import magiceye_solver, magiceye_solve_file
from scipy.datasets import face
import matplotlib as mpl
mpl.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from magiceye_solve.solver import InteractiveSolver # Import the new class
"""
Some basic tests for magiceye_solver
"""

logging.basicConfig(level=logging.DEBUG)

class TestSTL(unittest.TestCase):

    def test_null(self) -> None:
        # Test with an all-zero image (should have std=0)
        A: np.ndarray = np.zeros((128, 128))
        # magiceye_solver should handle std=0 and return zeros matching output shape logic
        # The offset calculation might return width, shift_pic might return zeros
        # Let's check the expected output shape based on the implementation
        # Offset will be width (128). shift_pic will return (128, 0).
        # Filters applied to (128, 0) will result in (128, 0).
        # Final solution shape should be (128, 0)
        out: np.ndarray = magiceye_solver(A)
        self.assertEqual(out.shape, (128, 0)) # Expect empty width due to offset=width

    def test_magiceye_solve_file(self) -> None:
        # Create a temporary test image file
        test_file = "test_face.png"
        solution_file = "test_face-solution.png"
        try:
            f: Figure = plt.figure(frameon=False)
            ax: Axes = f.add_subplot(111)
            # Use a known image like scipy.datasets.face
            img_data = face(gray=True) # Use grayscale face
            plt.imshow(img_data, cmap=plt.cm.gray)
            ax.set_axis_off()
            ax.autoscale_view(True, True, True)
            f.savefig(test_file, bbox_inches='tight', pad_inches=0)
            plt.close(f)

            # Run the file solver function
            magiceye_solve_file(test_file)

            # Check if the solution file was created
            self.assertTrue(os.path.exists(solution_file))

            # Optional: Check if the solution file can be read and has content
            try:
                 solution_img = plt.imread(solution_file)
                 self.assertGreater(solution_img.size, 0)
            except Exception as e:
                 self.fail(f"Failed to read solution file {solution_file}: {e}")

        finally:
            # Clean up created files
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(solution_file):
                os.remove(solution_file)

    def test_cli(self) -> None:
        # Create a temporary test image file
        test_file = "test_cli.png"
        solution_file = "test_cli-solution.png"
        try:
            f: Figure = plt.figure(frameon=False)
            ax: Axes = f.add_subplot(111)
            img_data = face(gray=True)
            plt.imshow(img_data, cmap=plt.cm.gray)
            ax.set_axis_off()
            ax.autoscale_view(True, True, True)
            f.savefig(test_file, bbox_inches='tight', pad_inches=0)
            plt.close(f)

            # Find the path to the current python interpreter
            python_executable = sys.executable

            # Run the CLI using 'python -m magiceye_solve.solver' for better env handling
            # This assumes the solver module has a runnable block or entry point logic
            # A better approach might be to directly call magiceye_solve_cli if possible
            # Or use the installed script name if setup.py entry_points are configured
            try:
                # Try running as a module first
                result: subprocess.CompletedProcess = subprocess.run(
                    [python_executable, '-m', 'magiceye_solve.solver', test_file],
                    capture_output=True, text=True, check=True, cwd=os.getcwd() # Run from project root
                )
            except subprocess.CalledProcessError as e:
                 # If module execution fails, try the installed script name
                 print(f"Running as module failed: {e.stderr}. Trying installed script 'magiceye_solver'...")
                 try:
                      result = subprocess.run(
                           ['magiceye_solver', test_file],
                           capture_output=True, text=True, check=True, cwd=os.getcwd()
                      )
                 except (subprocess.CalledProcessError, FileNotFoundError) as e2:
                      print(f"CLI execution failed: {e2}")
                      if isinstance(e2, subprocess.CalledProcessError):
                           print(f"Stderr: {e2.stderr}")
                      self.fail(f"CLI execution failed: {e2}")

            # Check if the output file exists
            if not os.path.exists(solution_file):
                print(f"Assertion Failed: Output file '{solution_file}' not found.")
                print(f"Subprocess stdout:\n{result.stdout}")
                print(f"Subprocess stderr:\n{result.stderr}")
            self.assertTrue(os.path.exists(solution_file))

        finally:
            # Clean up created files
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(solution_file):
                os.remove(solution_file)


class TestInteractiveSolver(unittest.TestCase):

    def setUp(self) -> None:
        """Create sample float images for testing."""
        # Use float images as the class now normalizes internally
        self.test_img_gray: np.ndarray = np.random.rand(50, 60)
        self.test_img_color: np.ndarray = np.random.rand(50, 60, 3)
        # Also test RGBA (alpha should be ignored)
        self.test_img_rgba: np.ndarray = np.random.rand(50, 60, 4)
        # Test uint8 image for normalization check
        self.test_img_uint8: np.ndarray = (np.random.rand(50, 60, 3) * 255).astype(np.uint8)

    def test_init(self) -> None:
        """Test initialization, normalization, and default offset calculation."""
        # Test Grayscale
        solver_gray = InteractiveSolver(self.test_img_gray)
        self.assertEqual(solver_gray.image.shape, (50, 60)) # Should be 2D
        self.assertFalse(solver_gray.color_image)
        self.assertEqual(solver_gray.c, 1)
        self.assertIsInstance(solver_gray.default_offset, int)
        self.assertGreaterEqual(solver_gray.default_offset, 1) # Offset can be >= 1

        # Test RGB Color
        solver_color = InteractiveSolver(self.test_img_color)
        self.assertEqual(solver_color.image.shape, (50, 60, 3))
        self.assertTrue(solver_color.color_image)
        self.assertEqual(solver_color.c, 3)
        self.assertIsInstance(solver_color.default_offset, int)
        self.assertGreaterEqual(solver_color.default_offset, 1)

        # Test RGBA Color (should ignore alpha)
        solver_rgba = InteractiveSolver(self.test_img_rgba)
        self.assertEqual(solver_rgba.image.shape, (50, 60, 3)) # Alpha channel removed
        self.assertTrue(solver_rgba.color_image)
        self.assertEqual(solver_rgba.c, 3) # Channel count should be 3
        self.assertIsInstance(solver_rgba.default_offset, int)
        self.assertGreaterEqual(solver_rgba.default_offset, 1)

        # Test uint8 normalization
        solver_uint8 = InteractiveSolver(self.test_img_uint8)
        self.assertEqual(solver_uint8.image.dtype, float) # Should be float
        self.assertLessEqual(np.max(solver_uint8.image), 1.0) # Max value <= 1.0
        self.assertGreaterEqual(np.min(solver_uint8.image), 0.0) # Min value >= 0.0
        self.assertEqual(solver_uint8.image.shape, (50, 60, 3))
        self.assertTrue(solver_uint8.color_image)
        self.assertEqual(solver_uint8.c, 3)

    def test_solve_with_valid_offset_gray(self) -> None:
        """Test solving grayscale with a valid user offset."""
        solver = InteractiveSolver(self.test_img_gray)
        user_offset = 10
        # Grayscale only supports 'separate' mode effectively (c=1)
        solved_img: np.ndarray = solver.solve_with_offset(user_offset, channel_mode='separate')

        # Expected shape: (height, width - offset)
        expected_width = max(0, solver.n - user_offset)
        self.assertEqual(solved_img.shape, (solver.m, expected_width))
        if expected_width > 0:
             self.assertGreater(np.sum(np.abs(solved_img)), 0) # Check it's not all zeros

        # Test 'average' mode on grayscale (should be identical)
        solved_avg: np.ndarray = solver.solve_with_offset(user_offset, channel_mode='average')
        self.assertEqual(solved_avg.shape, (solver.m, expected_width))
        if expected_width > 0:
             self.assertGreater(np.sum(np.abs(solved_avg)), 0)
             np.testing.assert_allclose(solved_img, solved_avg) # Results should match

    def test_solve_with_valid_offset_color_separate(self) -> None:
        """Test solving color with a valid user offset using 'separate' mode."""
        solver = InteractiveSolver(self.test_img_color)
        user_offset = 15
        solved_img: np.ndarray = solver.solve_with_offset(user_offset, channel_mode='separate')

        # Expected shape: (height, (width - offset) * channels)
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

        # Expected shape: (height, width - offset) - single channel output
        expected_width = max(0, solver.n - user_offset)
        self.assertEqual(solved_img.shape, (solver.m, expected_width))
        self.assertEqual(solved_img.ndim, 2) # Should be a 2D array
        if expected_width > 0:
             self.assertGreater(np.sum(np.abs(solved_img)), 0)

    def test_solve_with_zero_offset(self) -> None:
        """Test solving with zero offset (should return empty/zeros)."""
        solver_color = InteractiveSolver(self.test_img_color)
        # Test separate mode
        solved_sep: np.ndarray = solver_color.solve_with_offset(0, channel_mode='separate')
        expected_shape_sep = (solver_color.m, solver_color.n * solver_color.c)
        self.assertEqual(solved_sep.shape, expected_shape_sep)
        self.assertTrue((solved_sep == 0).all())

        # Test average mode
        solved_avg: np.ndarray = solver_color.solve_with_offset(0, channel_mode='average')
        expected_shape_avg = (solver_color.m, solver_color.n)
        self.assertEqual(solved_avg.shape, expected_shape_avg)
        self.assertTrue((solved_avg == 0).all())

    def test_solve_with_large_offset(self) -> None:
        """Test solving with an offset larger than image width."""
        solver_color = InteractiveSolver(self.test_img_color)
        user_offset = solver_color.n + 10 # Offset larger than width

        # Test separate mode
        solved_sep: np.ndarray = solver_color.solve_with_offset(user_offset, channel_mode='separate')
        expected_shape_sep = (solver_color.m, 0) # Width should be 0 * c = 0
        self.assertEqual(solved_sep.shape, expected_shape_sep)

        # Test average mode
        solved_avg: np.ndarray = solver_color.solve_with_offset(user_offset, channel_mode='average')
        expected_shape_avg = (solver_color.m, 0) # Width should be 0
        self.assertEqual(solved_avg.shape, expected_shape_avg)

    def test_solve_with_invalid_mode(self) -> None:
        """Test solving with an invalid channel_mode."""
        solver = InteractiveSolver(self.test_img_color)
        user_offset = 10
        solved_img: np.ndarray = solver.solve_with_offset(user_offset, channel_mode='invalid_mode')

        # Should return empty array matching the 'separate' mode output shape
        expected_width_per_channel = max(0, solver.n - user_offset)
        expected_total_width = expected_width_per_channel * solver.c
        expected_shape = (solver.m, expected_total_width)

        self.assertEqual(solved_img.shape, expected_shape)
        self.assertTrue((solved_img == 0).all())


if __name__ == '__main__':
    unittest.main()
