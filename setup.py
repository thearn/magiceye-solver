from setuptools import setup, find_packages

setup(name='magiceye_solve',
      version='0.1',
      install_requires=['numpy', 'scipy', 'matplotlib'],
      description="Program that automatically solves magic eye autostereograms",
      author='Tristan Hearn',
      author_email='tristanhearn@gmail.com',
      url='https://github.com/thearn/magiceye-solver',
      license='Apache 2.0',
      packages=['magiceye_solve'],
      entry_points={
          'console_scripts':
          ['magiceye_solver=magiceye_solve.solver:magiceye_solve_cli']
      }

      )
