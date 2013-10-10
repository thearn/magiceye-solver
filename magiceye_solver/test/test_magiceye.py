import logging
import os
import numpy as np
import unittest
from magiceye_solver import magiceye_solver, magiceye_solve_file
from scipy.misc import lena
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
"""
Some basic tests for magiceye_solver
"""

logging.basicConfig(level=logging.DEBUG)


class TestSTL(unittest.TestCase):

    def test_magiceye_solve(self):

        A = lena()
        n = A.shape[1]
        out = magiceye_solver(A)
        assert out.shape[1] <= n

    def test_null(self):

        A = np.zeros((128, 128))
        out = magiceye_solver(A)
        assert (out == 0.0).all()

    def test_magiceye_solve_file(self):
        f = plt.figure(frameon=False)
        ax = f.add_subplot(111)
        plt.imshow(lena(), cmap=plt.cm.gray)
        ax.set_axis_off()
        ax.autoscale_view(True, True, True)
        f.savefig("test.png", bbox_inches='tight')
        plt.close()

        magiceye_solve_file("test.png")

        assert os.path.exists("test-solution.png")
        os.remove("test.png")
        os.remove("test-solution.png")

    def test_cli(self):
        f = plt.figure(frameon=False)
        ax = f.add_subplot(111)
        plt.imshow(lena(), cmap=plt.cm.gray)
        ax.set_axis_off()
        ax.autoscale_view(True, True, True)
        f.savefig("test.png", bbox_inches='tight')
        plt.close()

        import os
        os.system('magiceye_solver test.png')
        assert os.path.exists("test-solution.png")
        os.remove("test.png")
        os.remove("test-solution.png")

if __name__ == '__main__':
    unittest.main()
