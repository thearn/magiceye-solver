from setuptools import setup, find_packages

# Function to read requirements from requirements.txt
def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

# Read requirements
install_reqs = parse_requirements('requirements.txt')

setup(name='magiceye_solve',
      version='0.1',
      install_requires=install_reqs,
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
