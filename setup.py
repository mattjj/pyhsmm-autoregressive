from distutils.core import setup
import numpy as np
import os

import pyhsmm
from pyhsmm.util.cyutil import cythonize # my version of Cython.Build.cythonize
eigen_include_path = os.path.join(
        os.path.dirname(pyhsmm.__file__),'deps/Eigen3/')
pyhsmm_include_path = os.path.join(
        os.path.dirname(pyhsmm.__file__),'internals/')


setup(
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[np.get_include(),pyhsmm_include_path,eigen_include_path],
)

