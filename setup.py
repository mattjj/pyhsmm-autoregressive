from distutils.core import setup
import numpy as np
import os
import sys
from Cython.Build import cythonize

import pyhsmm
eigen_include_path = os.path.join(
    os.path.dirname(pyhsmm.__file__),'deps','Eigen3')
pyhsmm_include_path = os.path.join(
    os.path.dirname(pyhsmm.__file__),'internals')

if '--compile-stuff' in sys.argv:
    extra_compile_args = ['-w','-DNDEBUG']
    extra_link_args =[]

    ext_modules = cythonize('**/*.pyx')
    for e in ext_modules:
        e.extra_compile_args.extend(extra_compile_args)
        e.extra_link_args.extend(extra_link_args)
else:
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
)
