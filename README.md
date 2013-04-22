![Alt text](http://i.imgur.com/AUmpOSr.png "Example" )

# magiceye-solver
Are you as frustrated by not being able to see those magic eye optical illusions as I am? Well, now you're in luck.

This is a  short python code that demonstrates how to automatically "solve" a magic eye autostereogram by estimating a 
projection of the underlying image. This provides a decent contour outline of the hidden object, though most finer detail
tends to be lost.

Requirements:
---------------

- Python 2.7+
- Numpy 1.5+
- Scipy 0.12+

Optional:

- scikit-image 0.8+ (code will attempt to import filtering functions for additional post processing, but will not raise an error if 
library is not available)

Example usages:
----------
This code can be used in three different ways.
### Run directly, with a command line argument:

Run `magic_eye_solver.py` with the filename of the image that you would like to
process passed as an argument:

```bash
$ python magic_eye_solver.py "example_images/example1.png"
```
This generates text output:
```
Solving image example_images/example1.png...
Saving solution to example_images/example1_solution.png...
Saving joined to example_images/example1_joined.png...
```

This will generate `example1_solution.png` and `example1_joined.png` in the same
directory as the original files, showing the computed solution and a side-by-side
comparison of the original image and the solution, respectively.

### Run directly, without a command line argument:
If a filename is not specified, a list of png and jpg image files will be presented
to the user, who will then be prompted to select from the file names shown.

For example, we run the file directly, with no argument:
```bash
$ python magic_eye_solver.py
```
Select all example files (example1.png through example7.png) from a generated list (notice that banner.png is not 
selected by the user):
```
Please select from the following images:
========================================
(selection  filename)
0 example_images/banner.png
1 example_images/example1.png
2 example_images/example2.png
3 example_images/example3.png
4 example_images/example4.png
5 example_images/example5.png
6 example_images/example6.png
7 example_images/example7.png

Make selections (separate by commas): 1,2,3,4,5,6,7

Solving image example_images/example1.png...
Saving solution to example_images/example1_solution.png...
Saving joined to example_images/example1_joined.png...

Solving image example_images/example2.png...
Saving solution to example_images/example2_solution.png...
Saving joined to example_images/example2_joined.png...

Solving image example_images/example3.png...
Saving solution to example_images/example3_solution.png...
Saving joined to example_images/example3_joined.png...

Solving image example_images/example4.png...
Saving solution to example_images/example4_solution.png...
Saving joined to example_images/example4_joined.png...

Solving image example_images/example5.png...
Saving solution to example_images/example5_solution.png...
Saving joined to example_images/example5_joined.png...

Solving image example_images/example6.png...
Saving solution to example_images/example6_solution.png...
Saving joined to example_images/example6_joined.png...

Solving image example_images/example7.png...
Saving solution to example_images/example7_solution.png...
Saving joined to example_images/example7_joined.png...
```

### Imported and used as a library:
The `solve_magiceye()` method can also be imported from `magic_eye_solver.py` and
used in your own application like any other image/array processing function:

```python
from magic_eye_solver import solve_magiceye
import pylab #matplotlib plotting

image = pylab.imread("example_images/example1.png") #load magiceye image

solution = solve_magiceye(image) #solve it

pylab.imshow(solution, cmap = pylab.cm.gray) #plot the solution

pylab.show() #show the plot

```

How it works:
-------------
- For each of the R, G, and B channels of a magic eye image:
    1. An autocorrelation is computed (via FFT) to find strong horizontal periodicities in the inputted image
    2. The sum of all horizontal translative shifts of the image up to the peak autocorrelation
    shift is computed. That is, the entire image is "smeared" horizontally by the distance determined in step 1.
    3. An edge detection and uniform filter is applied to clean up the resulting sum and help separate the cumulated noise 
from useful objective information. 
- The most leptokurtotic (high in sample kurtosis) of three channels processed as above is returned as the solution
that is most likely to have the clearest 2D grayscale projection of the underlying image. 

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
The following examples show the computed solutions of the magiceye images 
found in the `example_images` directory.

![Alt text](http://i.imgur.com/AUmpOSr.png "Solution 1")

![Alt text](http://i.imgur.com/77qq4xY.jpg "Solution 2")

![Alt text](http://i.imgur.com/WZVGvkX.jpg "Solution 3")

![Alt text](http://i.imgur.com/3H9zeCJ.jpg "Solution 4")

![Alt text](http://i.imgur.com/Xru4K0v.jpg "Solution 5")

![Alt text](http://i.imgur.com/fAuwqXZ.jpg "Solution 6")

![Alt text](http://i.imgur.com/WmVzQdv.jpg "Solution 7")
