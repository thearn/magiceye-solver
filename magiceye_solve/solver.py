from argparse import ArgumentParser
import numpy as np
from scipy.ndimage import filters
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
try:
    from skimage import filter as ski_filter
    from skimage import exposure
except:
    ski_filter = None

from scipy.signal import fftconvolve


def offset(img):
    """
    calculates the offset that defines the stereoscopic effect
    """
    img = img - img.mean()
    ac = fftconvolve(img, np.flipud(np.fliplr(img)), mode='same')
    ac = ac[int(len(ac) / 2)]
    idx = np.where((ac - np.median(ac)) / ac.std() > 3)[0]
    diffs = []
    diffs = np.ediff1d(idx)
    return np.max(diffs)


def shift_pic(img):
    """
    Shifts an image using numpy.roll
    """
    gap = offset(img)
    m, n = img.shape
    shifted = np.zeros((m, n))
    for i in range(int(gap)):
        shifted += np.roll(img, -i, axis=1)
    return shifted[:, :-gap]


def post_process(img):
    """
    Post-processes the results using skimage filters
    """
    filt_1 = ski_filter.hprewitt(img)
    filt_2 = exposure.equalize_hist(filt_1)

    return filt_2


def magiceye_solver(x):
    """
    Solves the autostereogram image represented by the ndarray, x
    """
    shape = x.shape
    if len(shape) >= 3:
        m, n, c = shape
        color_image = True
    else:
        m, n, c = shape[0], shape[1], 1
        color_image = False
    solution = np.zeros((m, c * n),  dtype=float)
    for i in range(c):
        if color_image:
            color = x[:, :, i]
        else:
            color = x
        if color.std() == 0.0:
            continue
        shifted = shift_pic(color)
        filt_1 = filters.prewitt(shifted)
        filt_2 = filters.uniform_filter(filt_1, size=(5, 5))
        if ski_filter:
            filt_2 = post_process(filt_2)
        m, n = filt_2.shape
        solution[:m, i * n:i * n + n] = filt_2

    return solution[:, :c * n]


def magiceye_solve_file(fname, output_name=None):
    """
    Loads an image from disk, solves it, and saves it.
    """

    if not output_name:
        output_name = ''.join([fname.split(".")[0], '-solution.png'])

    image = plt.imread(fname)
    solved = magiceye_solver(image)

    f = plt.figure(frameon=False)
    ax = f.add_subplot(111)
    plt.imshow(solved, cmap=plt.cm.gray)
    ax.set_axis_off()
    ax.autoscale_view(True, True, True)
    f.savefig(output_name, bbox_inches='tight')
    plt.close()


def magiceye_solve_cli():
    """
    Provides a command-line interface to magiceye_solver
    """

    parser = ArgumentParser()
    parser.add_argument("f", help='Source image filename')
    parser.add_argument("-o", default="", help='Output filename')
    args = vars(parser.parse_args())
    src = args['f']
    fn = args['o']

    magiceye_solve_file(src, output_name=fn)
