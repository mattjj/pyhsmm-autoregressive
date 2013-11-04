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
    return ast(
            data,
            shape=(data.shape[0]-nlags,data.shape[1]*(nlags+1)),
            strides=(data.shape[1]*sz,sz))

def undo_AR_striding(strided_data,nlags):
    sz = strided_data.dtype.itemsize
    return ast(
            strided_data,
            shape=(strided_data.shape[0]+nlags,strided_data.shape[1]/(nlags+1)),
            strides=(strided_data.shape[1]/(nlags+1)*sz,sz))

def getardatadimension(strided_data):
    # TODO doesn't work when data is copied
    if isinstance(strided_data,np.ndarray):
        return strided_data.strides[0] // strided_data.strides[1]
    else:
        return getardatadimension(strided_data[0])


def getardatanlags(strided_data):
    if isinstance(strided_data,np.ndarray):
        return strided_data.shape[1] * strided_data.dtype.itemsize // strided_data.strides[0] - 1
    else:
        return getardatanlags(strided_data[0])

def is_strided(data):
    # TODO doesn't work when data is copied
    if isinstance(data,list):
        return all(is_strided(d) for d in data)
    return data.ndim == 2 and \
            data.strides[0] != data.dtype.itemsize * data.shape[1]

