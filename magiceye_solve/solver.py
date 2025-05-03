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

def offset(img: np.ndarray) -> int:
    """
    Calculates the offset that defines the stereoscopic effect.
    """
    img = img - img.mean()
    ac: np.ndarray = fftconvolve(img, np.flipud(np.fliplr(img)), mode='same')
    # Ensure ac has the expected dimension before indexing
    if ac.ndim < 1 or ac.shape[0] < 1:
         # Handle case where convolution result is unexpected (e.g., empty image input)
         return img.shape[1] # Fallback
    ac_center_row = ac[int(ac.shape[0] / 2)]
    # Check if center row is valid before proceeding
    if ac_center_row.size == 0 or ac_center_row.std() == 0:
         return img.shape[1] # Fallback if row is empty or has no variation

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

    diffs: np.ndarray = np.ediff1d(idx)
    if diffs.size == 0:
        # Fallback to image width if no significant peaks found
        return img.shape[1]
    # Use try-except for potential empty diffs if idx had <= 1 element
    try:
        max_diff = np.max(diffs)
        # Ensure max_diff is a reasonable value (e.g., not larger than image width)
        return min(max_diff, img.shape[1])
    except ValueError:
        return img.shape[1] # Fallback if diffs is empty

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


def magiceye_solver(x: np.ndarray) -> np.ndarray:
    """
    Solves the autostereogram image represented by the ndarray, x.
    """
    shape: Tuple[int, ...] = x.shape
    m: int
    n: int
    c: int = 1 # Default for grayscale
    color_image: bool
    if len(shape) >= 3 and shape[2] in [3, 4]: # Check for 3 (RGB) or 4 (RGBA) channels
        m, n, c = shape[0], shape[1], shape[2]
        color_image = True
        if c == 4: # Handle RGBA by ignoring alpha
             x = x[:, :, :3]
             c = 3
    elif len(shape) == 2:
        m, n = shape[0], shape[1]
        color_image = False
    else:
        # Handle unexpected shapes
        print(f"Warning: Unexpected image shape {shape}. Attempting to process as grayscale.")
        try:
             # Attempt to reshape or take the first 2D slice
             if x.ndim >= 2:
                  m, n = x.shape[0], x.shape[1]
                  if x.ndim > 2:
                       x = x[:, :, 0] # Take first channel if more than 2D
             else:
                  raise ValueError("Image dimensions are less than 2.")
             color_image = False
        except Exception as e:
             print(f"Error: Could not interpret image shape {shape}. Returning empty array. Details: {e}")
             return np.array([]) # Return empty if shape is unusable

    # Calculate default offset based on the first channel or grayscale
    first_channel = x[:, :, 0] if color_image else x
    default_offset = offset(first_channel) if first_channel.size > 0 and first_channel.std() > 0 else n

    # Allocate solution array - size depends on whether we process channels separately
    # For the standard solver, we assume separate processing.
    solution: np.ndarray = np.zeros((m, max(0, n - default_offset) * c), dtype=float)
    final_width_per_channel = max(0, n - default_offset)

    for i in range(c):
        color: np.ndarray = x[:, :, i] if color_image else x

        if color.size == 0 or color.std() == 0.0:
            continue # Skip empty or blank channels

        # Use the pre-calculated default_offset for shifting
        shifted: np.ndarray = shift_pic(color, default_offset)

        if shifted.size == 0: # Check if shift_pic returned an empty array
             continue

        # Apply filters
        try:
             filt_1: np.ndarray = ndimage.prewitt(shifted)
             filt_2: np.ndarray = ndimage.uniform_filter(filt_1, size=(5, 5))

             if _SKIMAGE_AVAILABLE:
                  filt_2 = post_process(filt_2)
             # else: # Removed redundant warning print
             #     print("Warning: Skimage filters not available, skipping post-processing.")

             # Ensure filt_2 has the expected width before placing in solution
             filt_m, filt_n = filt_2.shape
             if filt_n != final_width_per_channel:
                  # This shouldn't happen if shift_pic is correct, but handle defensively
                  print(f"Warning: Filtered channel {i} width mismatch. Expected {final_width_per_channel}, got {filt_n}.")
                  # Adjust filt_n or skip if dimensions are wrong
                  if filt_n < final_width_per_channel:
                       # Pad if smaller? Or skip? Let's skip for now.
                       continue
                  else:
                       # Truncate if larger
                       filt_2 = filt_2[:, :final_width_per_channel]
                       filt_n = final_width_per_channel


             # Place the processed channel into the solution array
             start_col = i * final_width_per_channel
             end_col = start_col + final_width_per_channel
             # Ensure row count doesn't exceed original 'm'
             rows_to_copy = min(m, filt_m)

             solution[:rows_to_copy, start_col:end_col] = filt_2[:rows_to_copy, :]

        except Exception as e:
             print(f"Error processing channel {i}: {e}")
             continue # Skip channel on error

    return solution


def magiceye_solve_file(fname: str, output_name: Optional[str] = None) -> None:
    """
    Loads an image from disk, solves it, and saves it.
    """
    if not output_name:
        # Ensure fname has an extension before splitting
        parts = fname.rsplit('.', 1)
        base_name = parts[0] if len(parts) > 1 else fname
        # Use png for output as it's lossless and common
        output_name = f'{base_name}-solution.png'

    try:
        # Use matplotlib.pyplot.imread which handles various formats
        image: np.ndarray = plt.imread(fname)
        # Normalize image data to float [0, 1] if it's not already
        if image.dtype == np.uint8:
             image = image.astype(float) / 255.0
        elif image.dtype == np.uint16:
             image = image.astype(float) / 65535.0

    except FileNotFoundError:
        print(f"Error: File not found at {fname}")
        return
    except Exception as e:
        print(f"Error reading image file {fname}: {e}")
        return

    # Handle grayscale images loaded with an extra dimension by plt.imread
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)

    solved: np.ndarray = magiceye_solver(image)

    # Check if solved image is empty or invalid before saving
    if solved.size == 0:
        print("Error: Solver returned an empty or invalid image.")
        return

    # Create figure for saving
    f = plt.figure(frameon=False)
    # Ensure figure size matches image aspect ratio
    dpi = f.get_dpi()
    f.set_size_inches(solved.shape[1]/dpi, solved.shape[0]/dpi)
    ax = plt.Axes(f, [0., 0., 1., 1.]) # Fill the entire figure
    ax.set_axis_off()
    f.add_axes(ax)

    try:
        # Display using gray colormap, assuming output is intensity
        ax.imshow(solved, cmap=plt.cm.gray, aspect='auto')
    except Exception as e:
        print(f"Error preparing solved image for display: {e}")
        plt.close(f)
        return

    try:
        # Save with tight bounding box and no padding
        f.savefig(output_name, dpi=dpi, pad_inches=0)
        print(f"Solution saved to {output_name}")
    except Exception as e:
        print(f"Error saving image to {output_name}: {e}")
    finally:
        plt.close(f) # Ensure figure is closed


def magiceye_solve_cli() -> None:
    """
    Provides a command-line interface to magiceye_solver.
    """
    parser = ArgumentParser()
    parser.add_argument("f", help='Source image filename')
    parser.add_argument("-o", default=None, help='Output filename (optional)') # Use None default
    args = parser.parse_args() # Use parse_args directly
    src: str = args.f
    fn: Optional[str] = args.o

    magiceye_solve_file(src, output_name=fn)

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


        # Calculate the default offset using the first channel or grayscale
        first_channel: np.ndarray = self.image[:, :, 0] if self.color_image else self.image
        # Ensure the first channel is valid before calculating offset
        if first_channel.size > 0 and first_channel.std() > 0:
             self.default_offset: int = offset(first_channel)
        else:
             # Handle cases like all-black images or invalid input
             self.default_offset: int = self.n # Fallback offset

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
                 # Average mode returns single channel, width n
                 return np.zeros((self.m, self.n), dtype=float)
            else: # 'separate' or invalid mode (defaults to separate shape)
                 # Separate mode returns concatenated channels, width n*c
                 return np.zeros((self.m, self.n * self.c), dtype=float)

        int_gap = int(user_offset)
        # Calculate effective_gap based on original image width (self.n)
        effective_gap = min(int_gap, self.n)
        final_width_per_channel = max(0, self.n - effective_gap)

        if channel_mode == 'average':
            if not self.color_image:
                # If not color, treat as grayscale
                img_to_process = self.image
            else:
                # Average the channels
                img_to_process = np.mean(self.image, axis=2)

            if img_to_process.size == 0 or img_to_process.std() == 0.0:
                # Return shape for average mode (single channel result)
                return np.zeros((self.m, final_width_per_channel), dtype=float)

            # --- Shift using user_offset ---
            shifted: np.ndarray = shift_pic(img_to_process, effective_gap)

            if shifted.size == 0:
                 return np.zeros((self.m, 0), dtype=float) # Empty result if width becomes 0

            # --- Apply Filters ---
            try:
                 filt_1: np.ndarray = ndimage.prewitt(shifted)
                 filt_2: np.ndarray = ndimage.uniform_filter(filt_1, size=(5, 5))
                 if _SKIMAGE_AVAILABLE:
                      filt_2 = post_process(filt_2)
                 return filt_2 # Return the single processed channel
            except Exception as e:
                 print(f"Error filtering averaged image: {e}")
                 # Return empty array with expected shape on error
                 return np.zeros((self.m, final_width_per_channel), dtype=float)


        elif channel_mode == 'separate':
             # --- Separate channel processing ---
             # Allocate solution array for concatenated results
             solution: np.ndarray = np.zeros((self.m, final_width_per_channel * self.c), dtype=float)

             for i in range(self.c):
                 color: np.ndarray
                 if self.color_image:
                      color = self.image[:, :, i]
                 else:
                      # If not color image, process only once
                      if i > 0: break
                      color = self.image

                 if color.size == 0 or color.std() == 0.0:
                      continue # Skip this channel if it's blank or empty

                 # --- Shift using user_offset ---
                 shifted: np.ndarray = shift_pic(color, effective_gap)

                 if shifted.size == 0:
                      continue # Skip if shifting results in empty array

                 # --- Apply Filters ---
                 try:
                      filt_1: np.ndarray = ndimage.prewitt(shifted)
                      filt_2: np.ndarray = ndimage.uniform_filter(filt_1, size=(5, 5))
                      if _SKIMAGE_AVAILABLE:
                           filt_2 = post_process(filt_2)

                      # --- Place into solution array ---
                      filt_m, filt_n = filt_2.shape
                      if filt_n != final_width_per_channel:
                           # Handle potential width mismatch (defensive)
                           print(f"Warning: Filtered channel {i} width mismatch. Expected {final_width_per_channel}, got {filt_n}.")
                           if filt_n < final_width_per_channel: continue # Skip if too small
                           filt_2 = filt_2[:, :final_width_per_channel] # Truncate if too large
                           filt_n = final_width_per_channel

                      start_col = i * final_width_per_channel
                      end_col = start_col + final_width_per_channel
                      rows_to_copy = min(self.m, filt_m)

                      solution[:rows_to_copy, start_col:end_col] = filt_2[:rows_to_copy, :]

                 except Exception as e:
                      print(f"Error processing channel {i}: {e}")
                      continue # Skip channel on error

             return solution

        else:
             print(f"Error: Invalid channel_mode '{channel_mode}'. Use 'separate' or 'average'.")
             # Return empty array matching default 'separate' shape
             return np.zeros((self.m, final_width_per_channel * self.c), dtype=float)


# Example of how to run from command line if needed:
if __name__ == "__main__":
    magiceye_solve_cli()
