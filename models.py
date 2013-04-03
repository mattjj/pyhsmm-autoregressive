from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import pyhsmm

from util import AR_striding, undo_AR_striding
from pyhsmm.plugins.autoregressive.states import ARHMMStates, ARHSMMStates, ARHMMStatesEigen

# these exist only to play appropriate stride tricks on the data
# and to instantiate the appropriate states classes for forward generation

class ARHMM(pyhsmm.models.HMM):
    def __init__(self,nlags,*args,**kwargs):
        super(ARHMM,self).__init__(*args,**kwargs)
        self.nlags = nlags

    def add_data(self,data,stateseq=None,initialize_from_prior=True):
        strided_data = AR_striding(data.copy(),self.nlags)
        self.states_list.append(ARHMMStates(strided_data.shape[0],self.state_dim,self.obs_distns,self.dur_distns,self.trans_distn,self.init_state_distn,trunc=self.trunc,data=strided_data,stateseq=stateseq,nlags=self.nlags,initialize_from_prior=initialize_from_prior))

    def plot_observations(self,*args,**kwargs):
        pass

class ARHMMEigen(ARHMM):
    def add_data(self,data,stateseq=None,initialize_from_prior=True):
        strided_data = AR_striding(data.copy(),self.nlags)
        self.states_list.append(ARHMMStatesEigen(strided_data.shape[0],self.state_dim,self.obs_distns,self.dur_distns,self.trans_distn,self.init_state_distn,trunc=self.trunc,data=strided_data,stateseq=stateseq,nlags=self.nlags,initialize_from_prior=initialize_from_prior))


class ARHSMM(pyhsmm.models.HSMM):
    def __init__(self,nlags,*args,**kwargs):
        super(ARHSMM,self).__init__(*args,**kwargs)
        self.nlags = nlags

    def add_data(self,data,stateseq=None,censoring=None,initialize_from_prior=True):
        strided_data = AR_striding(data.copy(),self.nlags)
        self.states_list.append(ARHSMMStates(strided_data.shape[0],self.state_dim,self.obs_distns,self.dur_distns,self.trans_distn,self.init_state_distn,trunc=self.trunc,data=strided_data,stateseq=stateseq,censoring=censoring,nlags=self.nlags,initialize_from_prior=initialize_from_prior))

    def plot_observations(self,colors=None,states_objs=None):
        # TODO makethis pcolor background to keep track colors
        if colors is None:
            colors = self._get_colors()
        if states_objs is None:
            states_objs = self.states_list

        cmap = cm.get_cmap()

        for s in states_objs:
            data = undo_AR_striding(s.data,self.nlags)

            for state,start,dur in zip(s.stateseq_norep,np.concatenate(((0,),s.durations.cumsum()))[:-1],s.durations):
                plt.plot(np.arange(start,start+data[start:start+dur].shape[0]),data[start:start+dur],color=cmap(colors[state]))
