import logging
import os
import numpy as np
import unittest
import subprocess
import sys
from magiceye_solve import magiceye_solver, magiceye_solve_file
from scipy.datasets import face
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
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

if __name__ == '__main__':
    unittest.main()
