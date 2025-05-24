from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

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

# constants
MIN_OFFSET_THRESHOLD = 10  # min offset for images wider than this
MIN_PEAK_DISTANCE_FROM_CENTER = 20 # min distance of a peak from the center to be considered valid
AUTOCORRELATION_STD_FACTOR = 3 # factor for threshold calculation in autocorrelation

def offset(img: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the offset that defines the stereoscopic effect.
    Selects the offset corresponding to the highest peak in the autocorrelation curve.
    """
    if img.size == 0 or img.ndim < 2:
        return 0, np.array([]), np.array([]), np.array([]) # return 0 for invalid input

    img_mean_subtracted = img - img.mean()
    ac: np.ndarray = fftconvolve(img_mean_subtracted, np.flipud(np.fliplr(img_mean_subtracted)), mode='same')

    if ac.ndim < 1 or ac.shape[0] < 1:
        return img.shape[1], np.array([]), np.array([]), np.array([]) # fallback to image width

    ac_center_row = ac[ac.shape[0] // 2]

    if ac_center_row.size == 0 or ac_center_row.std() == 0:
        return img.shape[1], np.array([]), np.array([]), np.array([]) # fallback if center row is invalid

    idx = np.array([])
    try:
        std_dev = ac_center_row.std()
        if std_dev > 0: # ensure std_dev is positive
            threshold = AUTOCORRELATION_STD_FACTOR * std_dev
            median_val = np.median(ac_center_row)
            if np.isfinite(threshold): # ensure threshold is a valid number
                idx = np.where(ac_center_row - median_val > threshold)[0]
    except (ValueError, FloatingPointError):
        pass # idx remains empty if calculation fails

    diffs: np.ndarray = np.ediff1d(idx)
    raw_offset: int = img.shape[1] # default to image width

    if idx.size > 0:
        center_idx = len(ac_center_row) // 2
        # consider peaks sufficiently far from the center
        valid_peaks_mask = np.abs(idx - center_idx) > MIN_PEAK_DISTANCE_FROM_CENTER
        
        if np.any(valid_peaks_mask):
            valid_peaks = idx[valid_peaks_mask]
            valid_peaks_values = ac_center_row[valid_peaks]
            if valid_peaks.size > 0:
                # offset from the highest valid peak
                highest_peak_idx = valid_peaks[np.argmax(valid_peaks_values)]
                raw_offset = abs(highest_peak_idx - center_idx)
        
        # fallback to max difference between peaks if current raw_offset is too small
        if diffs.size > 0 and raw_offset < MIN_OFFSET_THRESHOLD:
            max_diff : int = np.max(diffs)
            if max_diff >= MIN_OFFSET_THRESHOLD:
                raw_offset = max_diff
            # if max_diff is also small, it implies no strong periodic pattern, keep img.shape[1] as fallback

    # apply min offset constraint based on image width
    min_practical_offset = MIN_OFFSET_THRESHOLD if img.shape[1] >= MIN_OFFSET_THRESHOLD else 1
    final_offset = max(min_practical_offset, min(raw_offset, img.shape[1]))

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

    def _process_single_channel(self, channel_data: np.ndarray, gap: int, target_width: int) -> NDArray[np.float64]:
        """Helper to process a single image channel."""
        if channel_data.size == 0 or channel_data.std() == 0.0:
            return np.zeros((self.m, target_width), dtype=np.float64)

        shifted = shift_pic(channel_data, gap)
        if shifted.size == 0 or shifted.shape[1] == 0: # check if shifted result is empty or has zero width
             return np.zeros((self.m, target_width), dtype=np.float64)

        try:
            filt_1 = ndimage.prewitt(shifted)
            filt_2 = ndimage.uniform_filter(filt_1, size=(5, 5))
            if _SKIMAGE_AVAILABLE:
                filt_2 = post_process(filt_2)
            
            # ensure correct width
            if filt_2.shape[1] > target_width:
                filt_2 = filt_2[:, :target_width]
            elif filt_2.shape[1] < target_width:
                # pad if narrower, though shift_pic should handle this by design
                padding = np.zeros((filt_2.shape[0], target_width - filt_2.shape[1]), dtype=filt_2.dtype)
                filt_2 = np.concatenate((filt_2, padding), axis=1)
            return filt_2.astype(np.float64)
        except Exception:
            return np.zeros((self.m, target_width), dtype=np.float64)

    def solve_with_offset(self, user_offset: int, channel_mode: str = 'separate') -> NDArray[np.float64]:
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
            # determine expected shape for zero offset based on mode
            # for average mode, it's single channel, original width
            # for separate mode, it's multi-channel, original width per channel
            # however, shift_pic with 0 gap returns original image, so width isn't reduced
            # thus, n or n*c is appropriate.
            return np.zeros((self.m, self.n if channel_mode == 'average' else self.n * self.c), dtype=np.float64)

        effective_gap = min(int(user_offset), self.n)
        final_width_per_channel = max(0, self.n - effective_gap)

        if channel_mode == 'average':
            img_to_process = np.mean(self.image, axis=2) if self.color_image else self.image
            return self._process_single_channel(img_to_process, effective_gap, final_width_per_channel)

        elif channel_mode == 'separate':
            # prepare an empty array for the full solution
            # if final_width_per_channel is 0, the solution will also have 0 width for all channels
            solution_shape = (self.m, final_width_per_channel * self.c)
            solution: NDArray[np.float64] = np.zeros(solution_shape, dtype=np.float64)
            
            if final_width_per_channel == 0: # if no width, return empty solution early
                return solution

            for i in range(self.c):
                channel_data: np.ndarray = self.image[:, :, i] if self.color_image else self.image
                if not self.color_image and i > 0: # only process once for grayscale
                    break 
                
                processed_channel = self._process_single_channel(channel_data, effective_gap, final_width_per_channel)
                
                # ensure processed_channel has the correct dimensions before assignment
                if processed_channel.shape[0] == self.m and processed_channel.shape[1] == final_width_per_channel:
                    start_col = i * final_width_per_channel
                    end_col = start_col + final_width_per_channel
                    solution[:, start_col:end_col] = processed_channel
                # if dimensions mismatch (e.g. due to an error in _process_single_channel returning wrong shape),
                # that part of the solution remains zeros, which is a graceful fallback.
            return solution

        else: # unknown channel_mode
            # fallback to a zero array with expected dimensions if mode is invalid
            return np.zeros((self.m, final_width_per_channel * self.c), dtype=np.float64)
