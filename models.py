from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import pyhsmm

from util import AR_striding, undo_AR_striding
from pyhsmm.plugins.autoregressive.states import ARHMMStates, ARHSMMStates, ARHMMStatesEigen
from pyhsmm.util.general import rle

# these exist only to play appropriate stride tricks on the data
# and to instantiate the appropriate states classes for forward generation

class ARHMM(pyhsmm.models.HMM):
    _states_class = ARHMMStates

    def __init__(self,nlags,*args,**kwargs):
        super(ARHMM,self).__init__(*args,**kwargs)
        self.nlags = nlags

    def add_data(self,data,stateseq=None,initialize_from_prior=True):
        strided_data = AR_striding(data,self.nlags)
        self.states_list.append(
                self._states_class(
                    nlags=self.nlags,
                    model=self,
                    data=strided_data,
                    stateseq=stateseq,
                    initialize_from_prior=initialize_from_prior))

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


class ARHMMEigen(ARHMM):
    _states_class = ARHMMStatesEigen


class ARStickyHMMEigen(ARHMMEigen,pyhsmm.models.StickyHMMEigen): # TODO test
    _states_class = ARHMMStatesEigen

    def __init__(self,nlags,*args,**kwargs):
        pyhsmm.models.StickyHMMEigen.__init__(self,*args,**kwargs)
        self.nlags = nlags


class ARHSMM(pyhsmm.models.HSMM):
    _states_class = ARHSMMStates

    def __init__(self,nlags,*args,**kwargs):
        super(ARHSMM,self).__init__(*args,**kwargs)
        self.nlags = nlags

    def add_data(self,data,stateseq=None,censoring=None,initialize_from_prior=True):
        strided_data = AR_striding(data,self.nlags)
        self.states_list.append(
                ARHSMMStates(
                    nlags=self.nlags,
                    model=self,
                    data=strided_data,
                    stateseq=stateseq,
                    censoring=censoring,
                    initialize_from_prior=initialize_from_prior))

    # TODO repeated code from ARHMM class
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

