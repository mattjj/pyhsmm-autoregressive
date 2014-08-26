from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import pyhsmm
from pyhsmm.util.general import rle
from pyhsmm.basic.distributions import Gaussian

from util import AR_striding, undo_AR_striding
from autoregressive.states import ARHMMStates, ARHSMMStates, \
        ARHMMStatesEigen, ARHSMMStatesEigen, ARHSMMStatesGeo

class _ARMixin(object):
    def __init__(self,init_emission_distns=None,**kwargs):
        super(_ARMixin,self).__init__(**kwargs)
        if init_emission_distns is None:
            init_emission_distns = \
                    [Gaussian(nu_0=self.P+10,sigma_0=np.eye(self.P),
                        mu_0=np.zeros(self.P),kappa_0=1.)
                        for _ in range(self.num_states)]
        self.init_emission_distns = init_emission_distns

    def add_data(self,data,strided=False,**kwargs):
        strided_data = AR_striding(data,self.nlags) if not strided else data
        super(_ARMixin,self).add_data(data=strided_data,**kwargs)

    def resample_parameters(self):
        super(_ARMixin,self).resample_parameters()
        self.resample_init_emission_distn()

    def resample_init_emission_distn(self):
        for state, d in enumerate(self.init_emission_distns):
            d.resample([s.data[:self.nlags].ravel()
                for s in self.states_list if s.stateseq[0] == state])
        self._clear_caches()

    @property
    def nlags(self):
        return self.obs_distns[0].nlags

    @property
    def D(self):
        return self.obs_distns[0].D

    @property
    def P(self):
        return self.D*self.nlags

    def plot_observations(self,colors=None,states_objs=None):
        if colors is None:
            colors = self._get_colors()
        if states_objs is None:
            states_objs = self.states_list

        cmap = cm.get_cmap()

        for s in states_objs:
            data = undo_AR_striding(s.data,self.nlags)

            stateseq_norep, durs = rle(s.stateseq)
            starts = np.concatenate(((0,),durs.cumsum()))
            for state,start,dur in zip(stateseq_norep,starts,durs):
                plt.plot(
                        np.arange(start,start+data[start:start+dur].shape[0]),
                        data[start:start+dur],
                        color=cmap(colors[state]))
            plt.xlim(0,s.T-1)

class ARHMM(_ARMixin,pyhsmm.models.HMM):
    _states_class = ARHMMStatesEigen

class ARWeakLimitHDPHMM(_ARMixin,pyhsmm.models.WeakLimitHDPHMM):
    _states_class = ARHMMStatesEigen


class ARHSMM(_ARMixin,pyhsmm.models.HSMM):
    _states_class = ARHSMMStatesEigen

class ARWeakLimitHDPHSMM(_ARMixin,pyhsmm.models.WeakLimitHDPHSMM):
    _states_class = ARHSMMStatesEigen

class ARWeakLimitStickyHDPHMM(_ARMixin,pyhsmm.models.WeakLimitStickyHDPHMM):
    _states_class = ARHMMStatesEigen

# class ARWeakLimitHDPHSMMIntNegBin(_ARMixin,pyhsmm.models.WeakLimitHDPHSMMIntNegBin):
#     _states_class = ARHSMMStatesIntegerNegativeBinomial

class ARWeakLimitGeoHDPHSMM(_ARMixin,pyhsmm.models.WeakLimitGeoHDPHSMM):
    _states_class = ARHSMMStatesGeo

