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
    # Ensure these are None if import fails, so checks later don't raise NameError
    ski_filter = None
    exposure = None
    _SKIMAGE_AVAILABLE = False

from scipy.signal import fftconvolve

def offset(img: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the offset that defines the stereoscopic effect.
    Now selects the offset corresponding to the highest peak in the autocorrelation curve.
    """
    img = img - img.mean()
    ac: np.ndarray = fftconvolve(img, np.flipud(np.fliplr(img)), mode='same')
    # Ensure ac has the expected dimension before indexing
    if ac.ndim < 1 or ac.shape[0] < 1:
         # Handle case where convolution result is unexpected (e.g., empty image input)
         return img.shape[1], np.array([]), np.array([]), np.array([]) # Fallback
    ac_center_row = ac[int(ac.shape[0] / 2)]
    # Check if center row is valid before proceeding
    if ac_center_row.size == 0 or ac_center_row.std() == 0:
         return img.shape[1], np.array([]), np.array([]), np.array([]) # Fallback if row is empty or has no variation

    # Use try-except for division by zero or invalid value in std
    try:
        threshold = 3 * ac_center_row.std()
        median_val = np.median(ac_center_row)
        # Ensure threshold is a finite number
        if not np.isfinite(threshold) or threshold <= 0:
             idx = np.array([]) # No valid peaks if std dev is zero or invalid
        else:
             idx: np.ndarray = np.where(ac_center_row - median_val > threshold)[0]
    except (ValueError, FloatingPointError):
        idx = np.array([]) # Handle potential errors during calculation

    # Calculate differences between consecutive peaks for compatibility with visualization
    diffs: np.ndarray = np.ediff1d(idx)
    
    if idx.size == 0:
        # Fallback to image width if no significant peaks found
        return img.shape[1], ac_center_row, idx, diffs
    
    try:
        # Find the index of the highest peak (ignoring the central peak)
        center_idx = len(ac_center_row) // 2
        # Avoid the central peak by excluding a small region around the center
        valid_peaks_mask = np.abs(idx - center_idx) > 20
        
        if np.any(valid_peaks_mask):
            valid_peaks = idx[valid_peaks_mask]
            valid_peaks_values = ac_center_row[valid_peaks]
            
            if valid_peaks.size > 0:
                # Find the index of the highest peak among valid peaks
                highest_peak_idx = valid_peaks[np.argmax(valid_peaks_values)]
                
                # Calculate the offset as the distance from center to highest peak
                best_offset = abs(highest_peak_idx - center_idx)
                
                # Ensure best_offset is a reasonable value (e.g., not larger than image width)
                return min(best_offset, img.shape[1]), ac_center_row, idx, diffs
        
        # If no valid peaks or processing failed, fall back to original max_diff method
        if diffs.size > 0:
            max_diff = np.max(diffs)
            return min(max_diff, img.shape[1]), ac_center_row, idx, diffs
        else:
            return img.shape[1], ac_center_row, idx, diffs
            
    except (ValueError, IndexError) as e:
        # Fallback if any calculation fails
        print(f"Error finding highest peak: {e}")
        if diffs.size > 0:
            max_diff = np.max(diffs)
            return min(max_diff, img.shape[1]), ac_center_row, idx, diffs
        return img.shape[1], ac_center_row, idx, diffs

def shift_pic(img: np.ndarray, gap: int) -> np.ndarray:
    """
    Shifts an image using numpy.roll based on the provided offset (gap).
    """
    m, n = img.shape
    shifted: np.ndarray = np.zeros((m, n))
    # Ensure gap is a valid integer for range
    int_gap = int(gap)
    if int_gap <= 0: # Avoid issues if gap is non-positive
        return img # Or handle as appropriate

    # Ensure effective_gap doesn't exceed image width
    effective_gap = min(int_gap, n)

    for i in range(effective_gap):
        shifted += np.roll(img, -i, axis=1)
    # Adjust slicing based on the effective gap used
    return shifted[:, :max(0, n - effective_gap)]

def post_process(img: np.ndarray) -> np.ndarray:
    """
    Post-processes the results using skimage filters if available.
    """
    # Use the boolean flag for a clearer check
    if not _SKIMAGE_AVAILABLE:
        return img
    # We know ski_filter and exposure are available if _SKIMAGE_AVAILABLE is True
    # Add checks for empty or invalid input to filters
    if img.size == 0: return img
    try:
        filt_1: np.ndarray = ski_filter.prewitt(img)
        # Check if filt_1 is valid before histogram equalization
        if filt_1.size == 0 or np.all(filt_1 == filt_1[0,0]): # Avoid issues with constant images
             return filt_1 # Return prewitt result if equalization might fail
        filt_2: np.ndarray = exposure.equalize_hist(filt_1)
        return filt_2
    except Exception as e:
        print(f"Warning: Skimage post-processing failed: {e}")
        return img # Return original image on filter error

    return solution

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
         # Normalize image data to float [0, 1] if it's not already
        if image.dtype == np.uint8:
             image = image.astype(float) / 255.0
        elif image.dtype == np.uint16:
             image = image.astype(float) / 65535.0

        # Handle grayscale images loaded with an extra dimension
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(axis=2)

        self.image: np.ndarray = image
        self.shape: Tuple[int, ...] = image.shape
        self.m: int
        self.n: int
        self.c: int = 1
        self.color_image: bool

        if len(self.shape) >= 3 and self.shape[2] in [3, 4]:
            self.m, self.n, self.c = self.shape[0], self.shape[1], self.shape[2]
            self.color_image = True
            if self.c == 4: # Handle RGBA by ignoring alpha
                 self.image = self.image[:, :, :3]
                 self.c = 3
        elif len(self.shape) == 2:
            self.m, self.n = self.shape[0], self.shape[1]
            self.color_image = False
        else:
             # Handle unexpected shapes during init
             raise ValueError(f"Unsupported image shape: {self.shape}")

        # Calculate the default offset and autocorrelation curve using the first channel or grayscale
        first_channel: np.ndarray = self.image[:, :, 0] if self.color_image else self.image
        # Ensure the first channel is valid before calculating offset
        if first_channel.size > 0 and first_channel.std() > 0:
             self.default_offset, self.autocorrelation_curve, self.autocorrelation_peak_indices, self.autocorrelation_peak_diffs = offset(first_channel)
        else:
             # Handle cases like all-black images or invalid input
             self.default_offset: int = self.n # Fallback offset
             self.autocorrelation_curve: np.ndarray = np.array([]) # Initialize as empty if no curve
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
            print("Warning: User offset must be positive. Returning empty array.")
            # Determine return shape based on mode for zero offset
            if channel_mode == 'average':
                 return np.zeros((self.m, self.n), dtype=float)
            else: # 'separate' or invalid mode (defaults to separate shape)
                 return np.zeros((self.m, self.n * self.c), dtype=float)

        int_gap = int(user_offset)
        # Calculate effective_gap based on original image width (self.n)
        effective_gap = min(int_gap, self.n)
        final_width_per_channel = max(0, self.n - effective_gap)

        if channel_mode == 'average':
            if not self.color_image:
                img_to_process = self.image
            else:
                img_to_process = np.mean(self.image, axis=2)

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
            except Exception as e:
                 print(f"Error filtering averaged image: {e}")
                 return np.zeros((self.m, final_width_per_channel), dtype=float)

        elif channel_mode == 'separate':
             solution: np.ndarray = np.zeros((self.m, final_width_per_channel * self.c), dtype=float)

             for i in range(self.c):
                 color: np.ndarray
                 if self.color_image:
                      color = self.image[:, :, i]
                 else:
                      if i > 0: break
                      color = self.image

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
                           print(f"Warning: Filtered channel {i} width mismatch. Expected {final_width_per_channel}, got {filt_n}.")
                           if filt_n < final_width_per_channel: continue
                           filt_2 = filt_2[:, :final_width_per_channel]
                           filt_n = final_width_per_channel

                      start_col = i * final_width_per_channel
                      end_col = start_col + final_width_per_channel
                      rows_to_copy = min(self.m, filt_m)

                      solution[:rows_to_copy, start_col:end_col] = filt_2[:rows_to_copy, :]

                 except Exception as e:
                      print(f"Error processing channel {i}: {e}")
                      continue

             return solution

        else:
             print(f"Error: Invalid channel_mode '{channel_mode}'. Use 'separate' or 'average'.")
             return np.zeros((self.m, final_width_per_channel * self.c), dtype=float)
