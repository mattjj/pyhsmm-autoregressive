from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


### striding data for efficient AR computations

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


### data properties of AR coefficient matrices

def is_affine(A):
    return bool(A.shape[1] % A.shape[0])


def dimensions(A):
    if is_affine(A):
        A = A[:,:-1]
    D, nlags = A.shape[0], A.shape[1] // A.shape[0]
    return D, nlags, is_affine(A)


### analyzing AR coefficient matrices

def canonical_matrix(A):
    assert not is_affine(A), 'this function ignores affine parts of A'
    D, nlags, _ = dimensions(A)
    mat = np.zeros((D*nlags,D*nlags))
    mat[:-D,D:] = np.eye(D*(nlags-1))
    mat[-D:,:] = A[:,:D*nlags]
    return mat


def siso_transfer_function(A,i,j):
    D, nlags, _ = dimensions(A)
    assert 0 <= i < nlags*D and 0 <= j < D
    bigA = canonical_matrix(A)
    I = np.eye(bigA.shape[0])

    def H(freqs):
        zs = np.exp(1j*np.array(freqs))
        return np.array(
            [np.linalg.inv(z*I-bigA)[-D:,-2*D:-D][j,i] for z in zs])

    return H


def is_stable(A):
    bigA = canonical_matrix(A)
    return np.all(np.abs(np.linalg.eigvals(bigA)) < 1.)
