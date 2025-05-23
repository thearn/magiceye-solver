from argparse import ArgumentParser
from typing import Optional, Tuple, Any
import numpy as np
from scipy import ndimage
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
try:
    from skimage import filters as ski_filter
    from skimage import exposure
    _SKIMAGE_AVAILABLE = True
except ImportError:
    # skimage fallback
    ski_filter = None
    exposure = None
    _SKIMAGE_AVAILABLE = False

from scipy.signal import fftconvolve

def offset(img: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the offset that defines the stereoscopic effect.
    Now selects the offset corresponding to the highest peak in the autocorrelation curve.
    """
    # Handle empty or 1D arrays gracefully
    if img.size == 0 or img.ndim < 2:
        return 0, np.array([]), np.array([]), np.array([])
    img = img - img.mean()
    ac: np.ndarray = fftconvolve(img, np.flipud(np.fliplr(img)), mode='same')
    # check ac shape
    if ac.ndim < 1 or ac.shape[0] < 1:
         return img.shape[1], np.array([]), np.array([]), np.array([])
    ac_center_row = ac[int(ac.shape[0] / 2)]
    # check center row valid
    if ac_center_row.size == 0 or ac_center_row.std() == 0:
         return img.shape[1], np.array([]), np.array([]), np.array([])

    try:
        threshold = 3 * ac_center_row.std()
        median_val = np.median(ac_center_row)
        # threshold sanity check
        if not np.isfinite(threshold) or threshold <= 0:
             idx = np.array([])
        else:
             idx: np.ndarray = np.where(ac_center_row - median_val > threshold)[0]
    except (ValueError, FloatingPointError):
        idx = np.array([])

    # peak diffs for viz
    diffs: np.ndarray = np.ediff1d(idx)
    raw_offset: int = img.shape[1]

    if idx.size > 0:
        try:
            # skip center peak
            center_idx = len(ac_center_row) // 2
            valid_peaks_mask = np.abs(idx - center_idx) > 20
            if np.any(valid_peaks_mask):
                valid_peaks = idx[valid_peaks_mask]
                valid_peaks_values = ac_center_row[valid_peaks]
                if valid_peaks.size > 0:
                    # highest peak
                    highest_peak_idx = valid_peaks[np.argmax(valid_peaks_values)]
                    raw_offset = abs(highest_peak_idx - center_idx)
            # fallback to max diff if needed
            if diffs.size > 0:
                if raw_offset < 10 and np.max(diffs) >= 10:
                    raw_offset = np.max(diffs)
                elif raw_offset < 10 and np.max(diffs) < 10:
                    raw_offset = img.shape[1]
        except (ValueError, IndexError):
            raw_offset = img.shape[1]

    # min offset constraint
    if img.shape[1] < 10:
        final_offset = max(1, min(raw_offset, img.shape[1]))
    else:
        final_offset = max(10, min(raw_offset, img.shape[1]))

    return final_offset, ac_center_row, idx, diffs

def shift_pic(img: np.ndarray, gap: int) -> np.ndarray:
    """
    Shifts an image using numpy.roll based on the provided offset (gap).
    """
    m, n = img.shape
    shifted: np.ndarray = np.zeros((m, n))
    int_gap = int(gap)
    if int_gap <= 0:
        return img

    effective_gap = min(int_gap, n)
    for i in range(effective_gap):
        shifted += np.roll(img, -i, axis=1)
    return shifted[:, :max(0, n - effective_gap)]

def post_process(img: np.ndarray) -> np.ndarray:
    """
    Post-processes the results using skimage filters if available.
    """
    if not _SKIMAGE_AVAILABLE or img.size == 0:
        return img
    try:
        filt_1: np.ndarray = ski_filter.prewitt(img)
        if filt_1.size == 0 or np.all(filt_1 == filt_1[0, 0]):
            return filt_1
        filt_2: np.ndarray = exposure.equalize_hist(filt_1)
        return filt_2
    except Exception:
        return img

class InteractiveSolver:
    """
    Allows solving autostereograms with a user-defined offset.
    """
    def __init__(self, image: np.ndarray):
        """
        Initializes the solver with an image.

        Args:
            image: The input image as a NumPy array.
        """
        # normalize to float [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(float) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(float) / 65535.0

        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)

        self.image: np.ndarray = image
        self.shape: Tuple[int, ...] = image.shape
        self.c: int = 1

        # Additional check for >3 dimensions or unsupported 3D
        if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] in [1, 3, 4]):
            raise ValueError(f"Unsupported image shape: {self.shape}")

        if len(self.shape) >= 3 and self.shape[2] in [3, 4]:
            self.m, self.n, self.c = self.shape[0], self.shape[1], self.shape[2]
            self.color_image = True
            if self.c == 4:
                self.image = self.image[:, :, :3]
                self.c = 3
        elif len(self.shape) == 2:
            self.m, self.n = self.shape[0], self.shape[1]
            self.color_image = False
        else:
            raise ValueError(f"Unsupported image shape: {self.shape}")

        first_channel: np.ndarray = self.image[:, :, 0] if self.color_image else self.image
        if first_channel.size > 0 and first_channel.std() > 0:
            self.default_offset, self.autocorrelation_curve, self.autocorrelation_peak_indices, self.autocorrelation_peak_diffs = offset(first_channel)
        else:
            self.default_offset: int = self.n
            self.autocorrelation_curve: np.ndarray = np.array([])
            self.autocorrelation_peak_indices: np.ndarray = np.array([])
            self.autocorrelation_peak_diffs: np.ndarray = np.array([])

    def solve_with_offset(self, user_offset: int, channel_mode: str = 'separate') -> np.ndarray:
        """
        Solves the autostereogram using a specified offset and channel handling mode.

        Args:
            user_offset: The desired offset to use for shifting.
            channel_mode: How to handle color channels.
                          'separate' (default): Process each channel independently and concatenate results.
                          'average': Average color channels first, then process as grayscale.

        Returns:
            The solved image as a NumPy array.
        """
        if user_offset <= 0:
            if channel_mode == 'average':
                return np.zeros((self.m, self.n), dtype=float)
            else:
                return np.zeros((self.m, self.n * self.c), dtype=float)

        int_gap = int(user_offset)
        effective_gap = min(int_gap, self.n)
        final_width_per_channel = max(0, self.n - effective_gap)

        if channel_mode == 'average':
            img_to_process = np.mean(self.image, axis=2) if self.color_image else self.image
            if img_to_process.size == 0 or img_to_process.std() == 0.0:
                return np.zeros((self.m, final_width_per_channel), dtype=float)
            shifted: np.ndarray = shift_pic(img_to_process, effective_gap)
            if shifted.size == 0:
                return np.zeros((self.m, 0), dtype=float)
            try:
                filt_1: np.ndarray = ndimage.prewitt(shifted)
                filt_2: np.ndarray = ndimage.uniform_filter(filt_1, size=(5, 5))
                if _SKIMAGE_AVAILABLE:
                    filt_2 = post_process(filt_2)
                return filt_2
            except Exception:
                return np.zeros((self.m, final_width_per_channel), dtype=float)

        elif channel_mode == 'separate':
            solution: np.ndarray = np.zeros((self.m, final_width_per_channel * self.c), dtype=float)
            for i in range(self.c):
                color: np.ndarray = self.image[:, :, i] if self.color_image else self.image
                if not self.color_image and i > 0:
                    break
                if color.size == 0 or color.std() == 0.0:
                    continue
                shifted: np.ndarray = shift_pic(color, effective_gap)
                if shifted.size == 0:
                    continue
                try:
                    filt_1: np.ndarray = ndimage.prewitt(shifted)
                    filt_2: np.ndarray = ndimage.uniform_filter(filt_1, size=(5, 5))
                    if _SKIMAGE_AVAILABLE:
                        filt_2 = post_process(filt_2)
                    filt_m, filt_n = filt_2.shape
                    if filt_n != final_width_per_channel:
                        if filt_n < final_width_per_channel:
                            continue
                        filt_2 = filt_2[:, :final_width_per_channel]
                        filt_n = final_width_per_channel
                    start_col = i * final_width_per_channel
                    end_col = start_col + final_width_per_channel
                    rows_to_copy = min(self.m, filt_m)
                    solution[:rows_to_copy, start_col:end_col] = filt_2[:rows_to_copy, :]
                except Exception:
                    continue
            return solution

        else:
            return np.zeros((self.m, final_width_per_channel * self.c), dtype=float)
