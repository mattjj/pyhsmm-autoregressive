from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


### misc

def zero_pad(a, shape):
    assert a.ndim == len(shape)
    return np.pad(a, [(0, d1-d2) for d1, d2 in zip(shape,a.shape)], 'constant')


### striding data for efficient AR computations

def atleast_2d_col(data):
    return data.reshape((-1,1)) if data.ndim == 1 else data


def AR_striding(data,nlags):
    data = atleast_2d_col(np.require(data, requirements='C'))
    sz = data.dtype.itemsize
    shape = (data.shape[0]-nlags,data.shape[1]*(nlags+1))
    strides = (data.shape[1]*sz,sz)
    return ast(data, shape=shape, strides=strides)


def undo_AR_striding(strided_data,nlags):
    sz = strided_data.dtype.itemsize
    shape = (strided_data.shape[0]+nlags,strided_data.shape[1]/(nlags+1))
    strides = (strided_data.shape[1]/(nlags+1)*sz,sz)
    return ast(strided_data, shape=shape, strides=strides)


### data properties of AR coefficient matrices

def is_affine(A):
    A = np.atleast_2d(A)
    return bool(A.shape[1] % A.shape[0])


def unpack_affine(A):
    A = np.atleast_2d(A)
    if is_affine(A):
        A, b = A[:,:-1], A[:,-1]
    else:
        A, b = A, np.zeros(A.shape[0])
    return A, b


def dimensions(A):
    A = np.atleast_2d(A)
    if is_affine(A):
        A = A[:,:-1]
    D, nlags = A.shape[0], A.shape[1] // A.shape[0]
    return D, nlags, is_affine(A)


### analyzing AR coefficient matrices

def canonical_AR1(A, Sigma=None):
    A = np.atleast_2d(A)
    D, nlags, _ = dimensions(A)
    A, b = unpack_affine(A)

    bigA = zero_pad(A, (D*nlags, D*nlags))
    bigA[D:,:-D] = np.eye(D*(nlags-1))

    if Sigma is None:
        return bigA, zero_pad(b, (D*nlags,))
    else:
        return bigA, zero_pad(b, (D*nlags,)), zero_pad(Sigma, (D*nlags, D*nlags))


def canonical_matrix(A):
    # NOTE: throws away affine part
    D, nlags, _ = dimensions(A)
    mat = np.zeros((D*nlags,D*nlags))
    mat[:-D,D:] = np.eye(D*(nlags-1))
    mat[-D:,:] = A[:,:D*nlags]
    return mat


def siso_transfer_function(A,i,j):
    D, nlags, _ = dimensions(A)

    assert 0 <= i < nlags*D and 0 <= j < D
    assert not is_affine(A), 'this function ignores the affine part'

    bigA, _ = canonical_AR1(A)
    I = np.eye(bigA.shape[0])

    def H(freqs):
        zs = np.exp(1j*np.array(freqs))
        return np.array(
            [np.linalg.inv(z*I-bigA)[-D:,-2*D:-D][j,i] for z in zs])

    return H


def is_stable(A):
    bigA, _ = canonical_AR1(A)
    return np.all(np.abs(np.linalg.eigvals(bigA)) < 1.)


### AR process utilities

def predict_nsteps(nsteps, A, Sigma, mu_0, Sigma_0=None):
    # NOTE could use matrix exponentiation here to be faster for large nsteps
    return predict_sequence([A]*nsteps, [Sigma]*nsteps, mu_0, Sigma_0)


def predict_sequence(As, Sigmas, mu_0, Sigma_0=None):
    assert len(set(A.shape for A in As)) == 1, 'must all have same lags'
    Sigma_0 = Sigma_0 if Sigma_0 is not None else np.zeros(2*(mu_0.shape[0],))

    if len(As) == 0:
        return mu_0, Sigma_0
    D, nlags, _ = dimensions(As[0])

    def predict_onestep(A, Sigma, mu_0, Sigma_0):
        A, b, Sigma = canonical_AR1(A, Sigma)
        return A.dot(mu_0) + b, Sigma_0 + Sigma

    predictions = [(mu_0, Sigma_0)]
    for A, Sigma in zip(As, Sigmas):
        predictions.append(predict_onestep(A, Sigma, *predictions[-1]))

    return [(mu[:D], Sigma[:D,:D]) for mu, Sigma in predictions]
