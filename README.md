
---
title: Magic Eye Solver
emoji: 👀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

> 🚀 **Try the Magic Eye Solver instantly on [Hugging Face Spaces](https://huggingface.co/spaces/thearn/magiceye-solver)!**


[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20Spaces-Magic%20Eye%20Solver-blue)](https://huggingface.co/spaces/thearn/magiceye-solver)
![Build Status](https://github.com/thearn/magiceye-solver/actions/workflows/ci.yml/badge.svg)
[![Codecov](https://codecov.io/gh/thearn/magiceye-solver/branch/main/graph/badge.svg)](https://codecov.io/gh/thearn/magiceye-solver)

![Alt text](http://i.imgur.com/AUmpOSr.png "Example" )

UPDATE: fixed for Python 3.5

A python code that demonstrates how to automatically "solve" a magic eye autostereogram by estimating a
projection of the underlying image. This provides a decent contour outline of the hidden object, though most finer detail
tends to be lost.

Requirements:
--------------

- Python 3.13+
- Poetry (for dependency management)

## Installation:

This project uses Poetry for dependency management. Make sure you have Poetry installed first.

1.  Clone the repository.
2.  Install dependencies using Poetry:
    ```bash
    poetry install
    ```

### Development:

Tests can be run using Poetry:
```bash
poetry run pytest
```

Type checking can be run with:
```bash
poetry run mypy magiceye_solve
```

### For Hugging Face Deployment:

The project automatically generates a `requirements.txt` file for Hugging Face compatibility using pip-tools:
```bash
poetry run pip-compile requirements.in
```


## Examples

There are two main ways to use this code:

### Running the Gradio Web App:

The easiest way to try out the solver is by running the included Gradio web application.

1.  Make sure you have installed the dependencies (see Installation section).
2.  Run the app from the project's root directory:
    ```bash
    poetry run python app.py
    ```
3.  This will launch a web interface (usually at http://127.0.0.1:7860) where you can upload your own Magic Eye images or use the provided samples to see the solver in action.

### Imported and used as a library:
The `magiceye_solver()` and `magiceye_solver_file()` functions can also be imported and used in your own Python application like any other image/array processing function:

```python
from magiceye_solve import magiceye_solver, magiceye_solver_file
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


## How it works:

- For each of the R, G, and B channels of a magic eye image:
    1. An autocorrelation is computed (via FFT) to find strong horizontal periodicities in the inputted image
    2. The sum of all horizontal translative shifts of the image up to the peak autocorrelation
    shift is computed. That is, the entire image is "smeared" horizontally by the distance determined in step 1.
    3. An edge detection and uniform filter is applied to clean up the resulting sum and help separate the cumulated noise
from useful objective information.

The processed R, G and B channels are then concatenated into a single grayscale
image, as output.

## Notes / todo list:

- The post-process filtering should be improved to clean up the output a bit more. The solutions are kind of grainy.
- This certainly seems to work better for some autostereogram images than others, but still seems to give generally
useful output for the test images I've been able to collect so far.
- Good alternative solution methods are likely to exist, so there is still plenty of
experimentation left to do with this.
- I experimented with PCA and ICA (both as pre-processing the R, G, B channels and as post-processing of the results),
but this didn't improve the results very much.

## Example results

![Alt text](http://i.imgur.com/AUmpOSr.png "Solution 1")

![Alt text](http://i.imgur.com/77qq4xY.jpg "Solution 2")

![Alt text](http://i.imgur.com/WZVGvkX.jpg "Solution 3")

![Alt text](http://i.imgur.com/3H9zeCJ.jpg "Solution 4")

![Alt text](http://i.imgur.com/Xru4K0v.jpg "Solution 5")

![Alt text](http://i.imgur.com/fAuwqXZ.jpg "Solution 6")

![Alt text](http://i.imgur.com/WmVzQdv.jpg "Solution 7")
