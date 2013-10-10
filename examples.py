import os
import glob

from magiceye_solver import magiceye_solve_file

"""
Runs magiceye_solver on the images in the samples directory
"""

os.chdir("samples")
types = ('*.jpg', '*.gif', '*.png')
files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob(files))

for fname in sorted(files_grabbed):
    if "-solution.png" not in fname:
        magiceye_solve_file(fname)
