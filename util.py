from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def AR_striding(data,nlags):
    # I had some trouble with views and as_strided, so copy if not contiguous
    data = np.asarray(data)
    if not data.flags.c_contiguous:
        data = data.copy(order='C')
    if data.ndim == 1:
        data = np.reshape(data,(-1,1))
    sz = data.dtype.itemsize
    return ast(data,shape=(data.shape[0]-nlags,data.shape[1]*(nlags+1)),strides=(data.shape[1]*sz,sz))

def undo_AR_striding(data,nlags):
    sz = data.dtype.itemsize
    return ast(data,shape=(data.shape[0]+nlags,data.shape[1]/(nlags+1)),strides=(data.shape[1]/(nlags+1)*sz,sz))

