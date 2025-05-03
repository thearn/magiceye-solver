from argparse import ArgumentParser
from typing import Optional, Tuple, Any
import numpy as np
from scipy import ndimage
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
try:
    from skimage import filter as ski_filter
    from skimage import exposure
except ImportError:
    ski_filter = None

from scipy.signal import fftconvolve

def offset(img: np.ndarray) -> int:
    """
    Calculates the offset that defines the stereoscopic effect.
    """
    img = img - img.mean()
    ac: np.ndarray = fftconvolve(img, np.flipud(np.fliplr(img)), mode='same')
    ac = ac[int(len(ac) / 2)]
    idx: np.ndarray = np.where((ac - np.median(ac)) / ac.std() > 3)[0]
    # diffs = [] # This line seems unused
    diffs: np.ndarray = np.ediff1d(idx)
    if diffs.size == 0:
        # Handle cases where no significant differences are found, maybe return a default?
        # Returning image width as a fallback, though this might not be ideal.
        return img.shape[1]
    return np.max(diffs)

def shift_pic(img: np.ndarray) -> np.ndarray:
    """
    Shifts an image using numpy.roll based on the calculated offset.
    """
    gap: int = offset(img)
    m, n = img.shape
    shifted: np.ndarray = np.zeros((m, n))
    # Ensure gap is a valid integer for range
    int_gap = int(gap)
    if int_gap <= 0: # Avoid issues if gap is non-positive
        return img # Or handle as appropriate
    for i in range(int_gap):
        shifted += np.roll(img, -i, axis=1)
    # Adjust slicing if gap is larger than image width
    return shifted[:, :max(0, n - int_gap)]

def post_process(img: np.ndarray) -> np.ndarray:
    """
    Post-processes the results using skimage filters if available.
    """
    if ski_filter is None:
        # Return input if skimage is not available or processing is skipped
        return img
    # Assuming ski_filter.hprewitt and exposure.equalize_hist return ndarray
    filt_1: np.ndarray = ski_filter.hprewitt(img)
    filt_2: np.ndarray = exposure.equalize_hist(filt_1)
    return filt_2

def magiceye_solver(x: np.ndarray) -> np.ndarray:
    """
    Solves the autostereogram image represented by the ndarray, x.
    """
    shape: Tuple[int, ...] = x.shape
    m: int
    n: int
    c: int
    color_image: bool
    if len(shape) >= 3:
        m, n, c = shape[0], shape[1], shape[2]
        color_image = True
    else:
        m, n = shape[0], shape[1]
        c = 1
        color_image = False

    # Ensure c is at least 1 for calculation
    c = max(1, c)
    solution: np.ndarray = np.zeros((m, c * n), dtype=float)

    for i in range(c):
        color: np.ndarray
        if color_image:
            color = x[:, :, i]
        else:
            # Ensure x is treated as 2D if not color
            color = x if x.ndim == 2 else x.reshape(m, n)

        if color.std() == 0.0:
            continue

        shifted: np.ndarray = shift_pic(color)
        if shifted.size == 0: # Check if shift_pic returned an empty array
             continue

        filt_1: np.ndarray = ndimage.prewitt(shifted)
        filt_2: np.ndarray = ndimage.uniform_filter(filt_1, size=(5, 5))

        if ski_filter:
            filt_2 = post_process(filt_2)

        filt_m, filt_n = filt_2.shape
        # Ensure indices are within bounds
        start_col = i * n
        end_col = start_col + filt_n
        if end_col > solution.shape[1]:
             # Adjust if filt_n makes it exceed solution bounds (can happen if n != filt_n)
             end_col = solution.shape[1]
             filt_n = end_col - start_col # Recalculate filt_n based on available space

        if filt_m > m: # Ensure row count doesn't exceed original
             filt_m = m

        solution[:filt_m, start_col:end_col] = filt_2[:filt_m, :filt_n]

    # Adjust final slice if necessary
    final_cols = c * n
    return solution[:, :final_cols]


def magiceye_solve_file(fname: str, output_name: Optional[str] = None) -> None:
    """
    Loads an image from disk, solves it, and saves it.
    """
    if not output_name:
        # Ensure fname has an extension before splitting
        parts = fname.rsplit('.', 1)
        base_name = parts[0] if len(parts) > 1 else fname
        output_name = f'{base_name}-solution.png'

    try:
        image: np.ndarray = plt.imread(fname)
    except FileNotFoundError:
        print(f"Error: File not found at {fname}")
        return
    except Exception as e:
        print(f"Error reading image file {fname}: {e}")
        return

    solved: np.ndarray = magiceye_solver(image)

    # Check if solved image is empty or invalid before saving
    if solved.size == 0:
        print("Error: Solver returned an empty image.")
        return

    f = plt.figure(frameon=False)
    ax = f.add_subplot(111)
    try:
        plt.imshow(solved, cmap=plt.cm.gray)
    except Exception as e:
        print(f"Error displaying solved image: {e}")
        plt.close(f)
        return

    ax.set_axis_off()
    ax.autoscale_view(True, True, True)
    try:
        f.savefig(output_name, bbox_inches='tight')
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

# Example of how to run from command line if needed:
# if __name__ == "__main__":
#     magiceye_solve_cli()
