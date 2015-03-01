from __future__ import division
import numpy as np
from numpy import newaxis as na

from pyhsmm.basic.abstractions import GibbsSampling, MaxLikelihood, Distribution
from pyhsmm.basic.distributions import Regression, ARDRegression
from pyhsmm.util.stats import sample_mniw, sample_invwishart, sample_mn, \
        getdatasize

from util import AR_striding

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

    def _get_statistics(self,data):
        return super(_ARMixin,self)._get_statistics(
                data=self._ensure_strided(data))

    def _get_weighted_statistics(self,data,weights):
        return super(_ARMixin,self)._get_weighted_statistics(
                data=self._ensure_strided(data),weights=weights)

    def log_likelihood(self,xy):
        return super(_ARMixin,self).log_likelihood(self._ensure_strided(xy))

    def _ensure_strided(self,data):
        if isinstance(data,np.ndarray):
            if data.shape[1] != self.D*(self.nlags+1):
                data = AR_striding(data,self.nlags)
            return data
        else:
            return [self._ensure_strided(d) for d in data]

class AutoRegression(_ARMixin,Regression):
    pass

class ARDAutoRegression(_ARMixin,ARDRegression):
    def __init__(self,M_0,**kwargs):
        blocksizes = [M_0.shape[0]]*(M_0.shape[1] // M_0.shape[0]) \
                + ([1] if M_0.shape[1] % M_0.shape[0] and M_0.shape[0] != 1 else [])
        super(ARDAutoRegression,self).__init__(
                M_0=M_0,blocksizes=blocksizes,**kwargs)

