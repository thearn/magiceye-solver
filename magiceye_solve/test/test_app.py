import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for tests

from app import (
    create_autocorrelation_plot,
    solve_and_display,
    process_image,
    reset_to_default_offset,
)

class DummySolver:
    def __init__(self):
        self.default_offset = 0
        self.autocorrelation_curve = np.zeros(10)
        self.autocorrelation_peak_diffs = np.zeros(3)
        self.autocorrelation_peak_indices = np.array([1, 2, 3], dtype=int)
    def solve(self, offset, channel_mode):
        return np.zeros((10, 10)), None
    def solve_with_offset(self, offset, channel_mode=None):
        return np.zeros((10, 10))

def test_create_autocorrelation_plot():
    arr = np.random.rand(100)
    fig = create_autocorrelation_plot(arr, 5, 10, np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert fig is not None

def test_solve_and_display():
    solver = DummySolver()
    result, fig = solve_and_display(solver, 0, "mono")
    assert result is not None

def test_process_image():
    img = np.zeros((10, 10))
    # Should not raise
    process_image(img)

def test_reset_to_default_offset():
    solver = DummySolver()
    result = reset_to_default_offset(solver, "mono")
    assert result is not None
