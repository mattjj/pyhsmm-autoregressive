from distutils.core import setup
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy as np
import os
from warnings import warn

import pyhsmm
eigen_include_path = os.path.join(
    os.path.dirname(os.path.dirname(pyhsmm.__file__)),'deps')
pyhsmm_include_path = os.path.join(
    os.path.dirname(pyhsmm.__file__),'internals')

# wrap the build_ext command to handle and compilation errors
class build_ext(_build_ext):
    # if extension modules fail to build, keep going anyway
    def run(self):
        try:
            _build_ext.run(self)
        except CompileError:
            warn('Failed to build optional extension modules')

try:
    ext_modules = cythonize('**/*.pyx')
except:
    warn('Failed to generate optional extension module code from Cython files')
    ext_modules = []

setup(
    name='autoregressive',
    version='0.0.3',
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
        'numpy',
        'scipy',
        'matplotlib',
        'pybasicbayes >= 0.1.3',
        'pyhsmm >= 0.1.4'],
    ext_modules=ext_modules,
    include_dirs=[np.get_include(),eigen_include_path,pyhsmm_include_path],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++'],
    classifiers=
        ['Intended Audience :: Science/Research',
         'Programming Language :: Python',
         'Programming Language :: C++'],
    cmdclass={'build_ext': build_ext},
)
