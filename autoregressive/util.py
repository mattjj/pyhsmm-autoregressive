from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

import scipy.stats as stats


### misc


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
    return D, nlags


### analyzing AR coefficient matrices

def canonical_AR1(A, Sigma=None):
    A = np.atleast_2d(A)
    D, nlags = dimensions(A)
    A, b = unpack_affine(A)

    def zero_pad(a):
        return np.pad(a, [(D*nlags - d, 0) for d in a.shape], 'constant')

    bigA = zero_pad(A) + np.eye(D*nlags, k=D)

    if Sigma is None:
        return bigA, zero_pad(b)
    else:
        return bigA, zero_pad(b), zero_pad(Sigma)


def siso_transfer_function(A,i,j):
    D, nlags = dimensions(A)

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

def score_kstep_predictions(A, Sigma, data, k):
    D, nlags = dimensions(A)
    strided_data = AR_striding(data, nlags-1)

    def propagator(k):
        bigA, bigb = canonical_AR1(A)
        A_k, b_k = np.linalg.matrix_power(bigA, k)[-D:], k*bigb[-D:]

        def propagate(strided_data):
            return A_k.dot(strided_data.T).T + b_k

        return propagate

    propagate = propagator(k)
    return stats.multivariate_normal(np.zeros(D), Sigma*k)\
        .logpdf(data[nlags-1+k:] - propagate(strided_data[:-k]))


### switching AR process utilities

def predict_sequence(As, Sigmas, mu_0, Sigma_0):
    (nlags, D), T = mu_0.shape, len(As)
    out_Sigmas = np.cumsum(Sigmas, axis=0)
    out_mus = np.vstack((mu_0, np.zeros((T, D))))

    strided_mus = AR_striding(out_mus, nlags)
    for (A, b), xy in zip(map(unpack_affine, As), strided_mus):
        xy[-D:] = A.dot(xy[:-D]) + b

    return out_mus[nlags:], out_Sigmas


def score_switching_predictions(As, Sigmas, data):
    if len(As) == 0:
        return np.array([], dtype=np.float64)
    nlags, D = dimensions(As[0])
    t = data.shape[0] - len(As)
    mus, sigmas = predict_sequence(As, Sigmas, data[t-nlags:t], np.zeros((D,D)))
    return [stats.multivariate_normal(mu, sigma).logpdf(d)
            for mu, sigma, d in zip(mus, sigmas, data[t:])]
