![Alt text](http://i.imgur.com/AUmpOSr.png "Example" )

# magiceye-solver
[![Build Status](https://travis-ci.org/thearn/magiceye-solver.png?branch=master)](https://travis-ci.org/thearn/magiceye-solver)

This is a  short python code that demonstrates how to automatically "solve" a magic eye autostereogram by estimating a
projection of the underlying image. This provides a decent contour outline of the hidden object, though most finer detail
tends to be lost.

Requirements:
--------------

- Python 2.7+
- Numpy 1.5+
- Scipy 0.12+

Optional:

- scikit-image 0.8+ (code will attempt to import filtering functions for additional post processing, but will not raise an error if
library is not available)

Example usages:
----------
This code can be used in two different ways.
### Run directly from the command line:

Run `magiceye_solver` with the filename of the image that you would like to
process passed as an argument:

For example:
```bash
$ magiceye_solver image.jpg
```

This will produce an output file named `image-solution.png` which displays the
solution of the autostereogram.
The output filename can also be specified directly:

```bash
$ magiceye_solver image.jpg -o image_output.png
```

### Imported and used as a library:
The `magiceye_solver()` and 'The `magiceye_solver_file()` functions can also be imported and
used in your own application like any other image/array processing function:

```python
from magiceye_solver import magiceye_solver, magiceye_solver_file
import pylab #matplotlib plotting

image = pylab.imread("image.jpg") #load magiceye image

solution = magiceye_solver(image) #solve it

pylab.imshow(solution, cmap = pylab.cm.gray) #plot the solution

pylab.show() #show the plot

"""
Or, we can process the file and produce a corresponding output directly:
"""

magiceye_solver_file("image_2.jpg")

magiceye_solver_file("image_3.jpg", output_name="image_3_output.png")

```


How it works:
-------------
- For each of the R, G, and B channels of a magic eye image:
    1. An autocorrelation is computed (via FFT) to find strong horizontal periodicities in the inputted image
    2. The sum of all horizontal translative shifts of the image up to the peak autocorrelation
    shift is computed. That is, the entire image is "smeared" horizontally by the distance determined in step 1.
    3. An edge detection and uniform filter is applied to clean up the resulting sum and help separate the cumulated noise
from useful objective information.

The processed R, G and B channels are then concatenated into a single grayscale
image, as output.

Notes / todo list:
---------
- The post-process filtering should be improved to clean up the output a bit more. The solutions are kind of grainy.
- This certainly seems to work better for some autostereogram images than others, but still seems to give generally
useful output for the test images I've been able to collect so far.
- Good alternative solution methods are likely to exist, so there is still plenty of
experimentation left to do with this.
- I experimented with PCA and ICA (both as pre-processing the R, G, B channels and as post-processing of the results),
but this didn't improve the results very much.

Example results
----------------

![Alt text](http://i.imgur.com/AUmpOSr.png "Solution 1")

![Alt text](http://i.imgur.com/77qq4xY.jpg "Solution 2")

![Alt text](http://i.imgur.com/WZVGvkX.jpg "Solution 3")

![Alt text](http://i.imgur.com/3H9zeCJ.jpg "Solution 4")

![Alt text](http://i.imgur.com/Xru4K0v.jpg "Solution 5")

![Alt text](http://i.imgur.com/fAuwqXZ.jpg "Solution 6")

![Alt text](http://i.imgur.com/WmVzQdv.jpg "Solution 7")
