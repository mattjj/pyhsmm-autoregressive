from __future__ import division
import numpy as np
from numpy import newaxis as na

from pyhsmm.basic.abstractions import GibbsSampling, MaxLikelihood, Distribution
from pyhsmm.basic.distributions import Regression
from pyhsmm.util.stats import sample_mniw, sample_invwishart, sample_mn, \
        getdatasize

from util import AR_striding, undo_AR_striding

class AutoRegression(Regression):
    @property
    def nlags(self):
        return self.D_in // self.D_out

    @property
    def D(self):
        return self.D_out

    @property
    def is_stable(self):
        D, nlags = self.D, self.nlags
        mat = np.zeros((D*nlags,D*nlags)),
        mat[:-D,D:] = np.eye(D*(nlags-1))
        mat[-D:,:] = self.A
        return np.all(np.abs(np.linalg.eigvals(mat)) < 1.)

    def rvs(self,lagged_data):
        return super(AutoRegression,self).rvs(
                x=np.atleast_2d(lagged_data.flatten()),return_xy=False)

