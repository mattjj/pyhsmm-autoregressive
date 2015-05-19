from distutils.core import setup
import numpy as np
import os
from Cython.Build import cythonize

import pyhsmm
eigen_include_path = os.path.join(
    os.path.dirname(pyhsmm.__file__),'deps','Eigen3')
pyhsmm_include_path = os.path.join(
    os.path.dirname(pyhsmm.__file__),'internals')

extra_compile_args = ['-w','-DNDEBUG']
extra_link_args =[]

ext_modules = cythonize('**/*.pyx')
for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)

setup(
    name='autoregressive',
    author='Matthew James Johnson',
    author_email='mattjj@csail.mit.edu',
    url='https://github.com/mattjj/pyhsmm-autoregressive',
    keywords=
        ['bayesian', 'inference', 'mcmc', 'time-series',
         'autoregressive', 'var', 'svar'],
    install_requires=
        ['Cython >= 0.20.1', 'numpy', 'scipy',
         'matplotlib', 'pybasicbayes', 'pyhsmm'],
    ext_modules=ext_modules,
    include_dirs=[np.get_include(),eigen_include_path,pyhsmm_include_path],
    classifiers=
        ['Intended Audience :: Science/Research',
         'Programming Language :: Python',
         'Programming Language :: C++'],
)

