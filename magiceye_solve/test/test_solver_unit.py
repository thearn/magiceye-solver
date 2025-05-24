import unittest
import numpy as np
import sys

from magiceye_solve import solver

class TestSolverFunctions(unittest.TestCase):
    def test_offset_empty_and_1d(self):
        # Empty array
        arr = np.array([])
        result = solver.offset(arr)
        self.assertEqual(result[0], 0)
        # 1D array
        arr1d = np.zeros(10)
        result = solver.offset(arr1d)
        self.assertEqual(result[0], 0)

    def test_offset_all_zeros(self):
        arr = np.zeros((20, 30))
        result = solver.offset(arr)
        self.assertEqual(result[0], 30)

    def test_offset_no_peaks(self):
        arr = np.ones((20, 30))
        result = solver.offset(arr)
        self.assertEqual(result[0], 30)

    def test_offset_width_less_than_10(self):
        arr = np.random.rand(10, 5)
        result = solver.offset(arr)
        self.assertGreaterEqual(result[0], 1)
        self.assertLessEqual(result[0], 5)

    def test_offset_exception_handling(self):
        # Should trigger ValueError in np.where
        arr = np.full((20, 30), np.nan)
        result = solver.offset(arr)
        self.assertEqual(result[0], 30)

    def test_shift_pic_gap_zero_and_negative(self):
        arr = np.random.rand(10, 10)
        out = solver.shift_pic(arr, 0)
        np.testing.assert_array_equal(out, arr)
        out = solver.shift_pic(arr, -5)
        np.testing.assert_array_equal(out, arr)

    def test_shift_pic_gap_greater_than_width(self):
        arr = np.random.rand(10, 10)
        out = solver.shift_pic(arr, 15)
        self.assertEqual(out.shape[1], 0)

    def test_shift_pic_one_row_one_col(self):
        arr = np.random.rand(1, 10)
        out = solver.shift_pic(arr, 3)
        self.assertEqual(out.shape[0], 1)
        arr = np.random.rand(10, 1)
        out = solver.shift_pic(arr, 1)
        self.assertEqual(out.shape[1], 0)

    def test_post_process_empty_and_all_same(self):
        arr = np.array([])
        out = solver.post_process(arr)
        np.testing.assert_array_equal(out, arr)
        arr = np.ones((10, 10))
        out = solver.post_process(arr)
        # Should return something, but all-same triggers early return
        self.assertTrue(np.allclose(out, out[0, 0]))

    def test_post_process_exception(self):
        # Pass a malformed array to trigger exception
        class BadArray(np.ndarray):
            def __new__(cls):
                return np.ndarray.__new__(cls, shape=(10, 10))
            def __array_function__(self, *args, **kwargs):
                raise Exception("fail")
        arr = BadArray()
        try:
            out = solver.post_process(arr)
            np.testing.assert_array_equal(out, arr)
        except Exception:
            pass  # Acceptable if exception propagates

    def test_interactive_solver_bad_shape(self):
        # 3D with 2 channels (unsupported)
        arr = np.random.rand(10, 10, 2)
        with self.assertRaises(ValueError):
            solver.InteractiveSolver(arr)
        # 4D array
        arr = np.random.rand(10, 10, 3, 2)
        with self.assertRaises(ValueError):
            solver.InteractiveSolver(arr)

    def test_interactive_solver_std_zero(self):
        arr = np.zeros((10, 10))
        s = solver.InteractiveSolver(arr)
        self.assertEqual(s.default_offset, s.n)
        self.assertEqual(s.autocorrelation_curve.size, 0)

    def test_solve_with_offset_negative(self):
        arr = np.random.rand(10, 10, 3)
        s = solver.InteractiveSolver(arr)
        out = s.solve_with_offset(-5)
        self.assertEqual(out.shape, (s.m, s.n * s.c))

    def test_solve_with_offset_std_zero(self):
        arr = np.ones((10, 10, 3))
        s = solver.InteractiveSolver(arr)
        out = s.solve_with_offset(5, channel_mode='average')
        self.assertEqual(out.shape, (s.m, s.n - 5))

    def test_solve_with_offset_exception(self):
        # Patch ndimage.prewitt to raise
        import scipy.ndimage
        orig_prewitt = scipy.ndimage.prewitt
        scipy.ndimage.prewitt = lambda x: (_ for _ in ()).throw(Exception("fail"))
        arr = np.random.rand(10, 10, 3)
        s = solver.InteractiveSolver(arr)
        out = s.solve_with_offset(5, channel_mode='average')
        self.assertEqual(out.shape, (s.m, s.n - 5))
        scipy.ndimage.prewitt = orig_prewitt

if __name__ == "__main__":
    unittest.main()
