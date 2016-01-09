from distutils.core import setup
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy as np
from os.path import dirname, join

import pyhsmm

pyhsmm_path = dirname(dirname(pyhsmm.__file__))
eigen_include = join(pyhsmm_path, 'deps')
pyhsmm_include = join(pyhsmm_path, 'pyhsmm', 'internals')

print eigen_include

setup(
    name='autoregressive',
    version='0.0.6',
    description='Extension for switching vector autoregressive models with pyhsmm',
    author='Matthew James Johnson',
    author_email='mattjj@csail.mit.edu',
    url='https://github.com/mattjj/pyhsmm-autoregressive',
    license='GPL',
    packages=['autoregressive'],
    keywords=[
        'bayesian', 'inference', 'mcmc', 'time-series',
        'autoregressive', 'var', 'svar'],
    install_requires=[
        'Cython >= 0.20.1',
        'numpy', 'scipy', 'matplotlib', 'pybasicbayes' 'pyhsmm']
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++'],
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[np.get_include(), eigen_include, pyhsmm_include]
)
