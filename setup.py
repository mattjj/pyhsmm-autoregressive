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
    ext_modules=ext_modules,
    include_dirs=[np.get_include(),eigen_include_path,pyhsmm_include_path],
)

