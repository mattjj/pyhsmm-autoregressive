from distutils.core import setup

setup(
    name='autoregressive',
    version='0.0.4',
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
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++'],
)
