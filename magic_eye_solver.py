import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import filters
from scipy.stats import kurtosis
from scipy.misc import imread, imsave
try:
    from skimage import filter as ski_filter
except:
    ski_filter = None

def fft_2d_autocorr(x):
    """ 2D autocorrelation, using FFT"""
    fr = fft2(x)
    fr2 = fft2(np.flipud(np.fliplr(x)))
    m,n = fr.shape
    cc = np.real(ifft2(fr*fr2))
    cc = np.roll(cc, -m/2+1,axis=0)
    cc = np.roll(cc, -n/2+1,axis=1)
    return cc

def offset(mx):
    ac = fft_2d_autocorr(mx)
    ac=ac[len(ac)/2]
    idx= np.where((ac-np.median(ac))/ac.std() > 3)[0]
    diffs=[]
    diffs = np.ediff1d(idx)
    return np.max( diffs)

def shift_pic(mx):
    gap = offset(mx)
    m,n = mx.shape
    mx2 = np.zeros((m,n))
    for i in xrange(int(gap)):
        mx2 += np.roll(mx,-i, axis = 1)
    return mx2[:,:-gap]

def post_process(mx2):
    mx2 = ski_filter.hprewitt(mx2)
    return mx2

def solve_magiceye(x):
    m,n,c = x.shape
    k_ = -3
    for i in xrange(3):
        color = x[:,:,i]
        mx2 = shift_pic(color)
        mx2 = filters.prewitt(mx2)
        mx2 = filters.uniform_filter(mx2, size = (5,5))
        if ski_filter:
            mx2 = post_process(mx2)
        k = kurtosis(mx2.flatten())
        if k > k_:
            solution = mx2
            k_ = k
    return solution
        
if __name__ == "__main__":
    """ 
    Generates solutions either from from command line arguments
    or by selection of suitable images generated from a list
    """
    import os
    import sys
    if len(sys.argv) < 2:
        pngs = []
        for root, dirs, files in os.walk(os.getcwd()):
            dirname = root.split(os.path.sep)[-1]
            for fn in files:
                typ = fn.split('.')[-1]
                if typ == 'png' or typ == "jpg":
                    if dirname != os.getcwd().split('/')[-1]:
                        pngs.append('/'.join([dirname,fn]))
                    else:
                        pngs.append(fn)
        print
        print "Please select from the following images:"
        print "========================================"
        print "(selection  filename)"
        i = 0
        for fn in pngs:
            print i, fn
            i+=1
        print
        print "Make selections (separate by commas):",
        select = raw_input()
        fns = [pngs[int(sub.strip())] for sub in select.split(',')]
    else:
        fns = [sys.argv[1]]
    print
    for fn in fns:
        x=imread(fn)
        m,n,_ = x.shape
        print "Solving image %s..." % fn
        solution = solve_magiceye(x)
        sfn = fn.split('.')[0]+'_solution.png'
        print "Saving solution to %s..." % sfn
        imsave(sfn,solution)
        
        solution=imread(sfn)
        n2=solution.shape[1]
        joined = np.zeros((m,n+n2,3))
        joined[:,:n,:] = x
        joined[:,n:,0] = solution
        joined[:,n:,1] = solution
        joined[:,n:,2] = solution
        
        sfn = fn.split('.')[0]+'_joined.png'
        print "Saving joined to %s..." % sfn
        imsave(sfn,joined)
        print
        