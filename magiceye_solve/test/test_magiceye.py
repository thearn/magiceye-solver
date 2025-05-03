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

        A: np.ndarray = np.zeros((128, 128))
        out: np.ndarray = magiceye_solver(A)
        assert (out == 0.0).all()

    def test_magiceye_solve_file(self) -> None:
        f: Figure = plt.figure(frameon=False)
        ax: Axes = f.add_subplot(111)
        plt.imshow(face(), cmap=plt.cm.gray)
        ax.set_axis_off()
        ax.autoscale_view(True, True, True)
        f.savefig("test.png", bbox_inches='tight')
        plt.close()

        magiceye_solve_file("test.png")

        assert os.path.exists("test-solution.png")
        os.remove("test.png")
        os.remove("test-solution.png")

    def test_cli(self) -> None:
        f: Figure = plt.figure(frameon=False)
        ax: Axes = f.add_subplot(111)
        plt.imshow(face(), cmap=plt.cm.gray)
        ax.set_axis_off()
        ax.autoscale_view(True, True, True)
        f.savefig("test.png", bbox_inches='tight')
        plt.close()

        import os
        # Use subprocess to run the CLI script with the current interpreter
        # Note: Running the module directly might not pick up the entry point correctly depending on installation.
        # A more robust way might be to find the script path if installed.
        # For now, we assume the 'magiceye_solver' script installed via entry_points is available in the PATH.
        try:
            # Execute the installed command-line script directly
            result: subprocess.CompletedProcess = subprocess.run(['magiceye_solver', 'test.png'], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
             print(f"CLI execution failed: {e.stderr}")
             raise e
        except FileNotFoundError:
             # This might happen if the script isn't in the PATH (e.g., venv not activated correctly)
             print("Command 'magiceye_solver' not found. Ensure the package is installed and the environment is active.")
             raise

        # Check if the output file exists, provide debug info if not
        output_file: str = "test-solution.png"
        if not os.path.exists(output_file):
            print(f"Assertion Failed: Output file '{output_file}' not found.")
            print(f"Subprocess stdout:\n{result.stdout}")
            print(f"Subprocess stderr:\n{result.stderr}")
        assert os.path.exists(output_file)

        # Cleanup
        os.remove("test.png")
        os.remove(output_file)


class TestInteractiveSolver(unittest.TestCase):

    def setUp(self) -> None:
        """Create a sample image for testing."""
        # Use a smaller, predictable image for interactive tests
        self.test_img_gray: np.ndarray = np.random.rand(50, 60) * 255
        self.test_img_color: np.ndarray = np.random.rand(50, 60, 3) * 255

    def test_init(self) -> None:
        """Test initialization and default offset calculation."""
        solver_gray = InteractiveSolver(self.test_img_gray)
        self.assertIsInstance(solver_gray.default_offset, int)
        self.assertGreater(solver_gray.default_offset, 0) # Expect a positive offset

        solver_color = InteractiveSolver(self.test_img_color)
        self.assertIsInstance(solver_color.default_offset, int)
        self.assertGreater(solver_color.default_offset, 0)

    def test_solve_with_valid_offset_gray(self) -> None:
        """Test solving grayscale with a valid user offset."""
        solver = InteractiveSolver(self.test_img_gray)
        user_offset = 10
        solved_img: np.ndarray = solver.solve_with_offset(user_offset)

        # Expected shape: (height, width - offset)
        expected_width = self.test_img_gray.shape[1] - user_offset
        self.assertEqual(solved_img.shape, (self.test_img_gray.shape[0], expected_width))
        self.assertGreater(np.sum(solved_img), 0) # Check it's not all zeros

    def test_solve_with_valid_offset_color(self) -> None:
        """Test solving color with a valid user offset."""
        solver = InteractiveSolver(self.test_img_color)
        user_offset = 15
        solved_img: np.ndarray = solver.solve_with_offset(user_offset)

        # Expected shape: (height, (width - offset) * channels) - check implementation detail
        # The current implementation returns shape (height, (width-offset)*channels) effectively
        # Let's re-check the implementation return shape logic.
        # The loop processes each channel, applying filters to a (m, n-offset) shifted image.
        # It places these into a solution array of shape (m, n*c).
        # The final return is solution[:, :final_cols_solved] where final_cols_solved = max(0, n - effective_gap) * c
        expected_width = max(0, self.test_img_color.shape[1] - user_offset) * self.test_img_color.shape[2]
        self.assertEqual(solved_img.shape, (self.test_img_color.shape[0], expected_width))
        self.assertGreater(np.sum(solved_img), 0) # Check it's not all zeros

    def test_solve_with_zero_offset(self) -> None:
        """Test solving with zero offset (should return empty/zeros)."""
        solver = InteractiveSolver(self.test_img_gray)
        solved_img: np.ndarray = solver.solve_with_offset(0)
        # Expect an array of zeros with original dimensions * channels (or just original for gray)
        expected_shape = (solver.m, solver.n * solver.c)
        self.assertEqual(solved_img.shape, expected_shape)
        self.assertTrue((solved_img == 0).all())

    def test_solve_with_large_offset(self) -> None:
        """Test solving with an offset larger than image width."""
        solver = InteractiveSolver(self.test_img_gray)
        user_offset = solver.n + 10 # Offset larger than width
        solved_img: np.ndarray = solver.solve_with_offset(user_offset)
        # Expected shape should have 0 width
        expected_shape = (solver.m, 0)
        self.assertEqual(solved_img.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
