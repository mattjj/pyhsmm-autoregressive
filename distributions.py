from __future__ import division
import numpy as np
from numpy import newaxis as na

from pyhsmm.basic.abstractions import GibbsSampling, MaxLikelihood, Distribution
from pyhsmm.basic.distributions import Regression, ARDRegression
from pyhsmm.util.stats import sample_mniw, sample_invwishart, sample_mn, \
        getdatasize

from pyhsmm.basic.pybasicbayes.util.general import blockarray

from util import AR_striding, undo_AR_striding

class _ARMixin(object):
    @property
    def nlags(self):
        if not self.affine:
            return self.D_in // self.D_out
        else:
            return (self.D_in - 1) // self.D_out

    @property
    def D(self):
        return self.D_out

    def rvs(self,lagged_data):
        return super(_ARMixin,self).rvs(
                x=np.atleast_2d(lagged_data.ravel()),return_xy=False)

    ### some AR analysis

    def eval_transfer_function(self,from_idx,to_idx,freqs):
        assert 0 <= from_idx < self.D and 0 <= to_idx < self.D
        D = self.D
        bigA = self.canonical_matrix
        I = np.eye(bigA.shape[0])
        zs = np.exp(1j*np.array(freqs))
        return np.array(
                [np.linalg.inv(z*I-bigA)[-D:,-2*D:-D][to_idx,from_idx]
                    for z in zs])

    @property
    def is_stable(self):
        bigA = self.canonical_matrix
        return np.all(np.abs(np.linalg.eigvals(bigA)) < 1.)

    @property
    def canonical_matrix(self):
        D, nlags = self.D, self.nlags
        mat = np.zeros((D*nlags,D*nlags))
        mat[:-D,D:] = np.eye(D*(nlags-1))
        mat[-D:,:] = self.A[:,:D*nlags]
        return mat

    ### for low-level code

    @property
    def _param_matrix(self):
        D, A, sigma = self.D, self.A, self.sigma
        sigma_inv = np.linalg.inv(sigma)
        parammat =  -1./2 * blockarray([
            [A.T.dot(sigma_inv).dot(A), -A.T.dot(sigma_inv)],
            [-sigma_inv.dot(A), sigma_inv]
            ])
        normalizer = D/2*np.log(2*np.pi) + np.log(np.diag(np.linalg.cholesky(sigma))).sum()
        return parammat, normalizer

class AutoRegression(_ARMixin,Regression):
    pass

class ARDAutoRegression(_ARMixin,ARDRegression):
    def __init__(self,M_0,**kwargs):
        blocksizes = [M_0.shape[0]]*(M_0.shape[1] // M_0.shape[0]) \
                + ([1] if M_0.shape[1] % M_0.shape[0] else [])
        super(ARDAutoRegression,self).__init__(
                M_0=M_0,blocksizes=blocksizes,**kwargs)

