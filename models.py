from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import pyhsmm

from util import AR_striding, undo_AR_striding
from autoregressive.states import ARHMMStates, ARHSMMStates, \
        ARHMMStatesEigen, ARHSMMStatesEigen, ARHSMMStatesGeo, \
        ARHMMStatesPossibleChangepoints
from pyhsmm.util.general import rle

# TODO model first observations?

class _ARMixin(object):
    @property
    def nlags(self):
        return self.obs_distns[0].nlags

    def add_data(self,data,already_strided=False,**kwargs):
        strided_data = AR_striding(data,self.nlags) if not already_strided else data
        super(_ARMixin,self).add_data(data=strided_data,**kwargs)

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


class ARHMMPossibleChangepoints(_ARMixin,pyhsmm.models.HMMPossibleChangepoints):
    _states_class = ARHMMStatesPossibleChangepoints

