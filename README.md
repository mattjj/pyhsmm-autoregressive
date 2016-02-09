Clone this with something like
```
git clone git://github.com/mattjj/pyhsmm-autoregressive.git autoregressive
```

The compiled code component is optional, but currently requires GCC (for its OpenMP support). Recent versions of Clang (3.7.0 and later) also support OpenMP but require the flag `-fopenmp=libomp`. See #4.
