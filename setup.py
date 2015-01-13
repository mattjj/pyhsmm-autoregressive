from distutils.core import setup
import numpy as np
import os, sys

import pyhsmm
from pyhsmm.util.cyutil import cythonize # my version of Cython.Build.cythonize
eigen_include_path = os.path.join(
        os.path.dirname(pyhsmm.__file__),'deps/Eigen3/')
pyhsmm_include_path = os.path.join(
        os.path.dirname(pyhsmm.__file__),'internals/')

extra_compile_args = ['-w','-DNDEBUG']
extra_link_args =[]

if '--with-mkl' in sys.argv:
    sys.argv.remove('--with-mkl')
    extra_compile_args.extend(['-m64','-I' + os.environ['MKLROOT'] + '/include','-DEIGEN_USE_MKL_ALL'])
    extra_link_args.extend(('-Wl,--start-group %(MKLROOT)s/lib/intel64/libmkl_intel_lp64.a %(MKLROOT)s/lib/intel64/libmkl_core.a %(MKLROOT)s/lib/intel64/libmkl_sequential.a -Wl,--end-group -lm' % {'MKLROOT':os.environ['MKLROOT']}).split(' '))

ext_modules = cythonize('**/*.pyx')
for e in ext_modules:
    e.extra_compile_args.extend(extra_compile_args)
    e.extra_link_args.extend(extra_link_args)

setup(
    ext_modules=ext_modules,
    include_dirs=[np.get_include(),eigen_include_path,pyhsmm_include_path],
)

