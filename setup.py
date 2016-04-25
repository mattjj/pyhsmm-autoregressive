from distutils.core import setup
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import numpy as np
from os.path import dirname, join, exists
from os import mkdir
from shutil import move
import tarfile
from urllib import urlretrieve
from glob import glob

# make dependency directory
if not exists('deps'):
    mkdir('deps')

# download Eigen if we don't have it in deps
eigenurl = 'http://bitbucket.org/eigen/eigen/get/3.2.6.tar.gz'
eigentarpath = join('deps', 'Eigen.tar.gz')
eigenpath = join('deps', 'Eigen')
if not exists(eigenpath):
    print('Downloading Eigen...')
    urlretrieve(eigenurl, eigentarpath)
    with tarfile.open(eigentarpath, 'r') as tar:
        tar.extractall('deps')
    thedir = glob(join('deps', 'eigen-eigen-*'))[0]
    move(join(thedir, 'Eigen'), eigenpath)
    print('...done!')

setup(
    name='autoregressive',
    version='0.1.1',
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
        'numpy', 'scipy', 'matplotlib', 'pybasicbayes >= 0.2.1', 'pyhsmm'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++'],
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[np.get_include(), 'deps']
)
