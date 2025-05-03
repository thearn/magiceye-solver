import os
import glob
from typing import List, Tuple

from magiceye_solve import magiceye_solve_file

"""
Runs magiceye_solver on the images in the samples directory
"""

os.chdir("samples")
types: Tuple[str, str, str] = ('*.jpg', '*.gif', '*.png')
files_grabbed: List[str] = []
for files_pattern in types:
    files_grabbed.extend(glob.glob(files_pattern))

for fname in sorted(files_grabbed):
    if "-solution.png" not in fname:
        magiceye_solve_file(fname)
