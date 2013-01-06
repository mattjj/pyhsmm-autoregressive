from __future__ import division

import pyhsmm

from util import AR_striding
from pyhsmm.plugins.autoregressive.states import ARHMMStates, ARHSMMStates

# these exist only to play appropriate stride tricks on the data
# and to instantiate the appropriate states classes for forward generation

class ARHMM(pyhsmm.models.HMM):
    def __init__(self,nlags,*args,**kwargs):
        super(ARHMM,self).__init__(*args,**kwargs)
        self.nlags = nlags

    def add_data(self,data,stateseq=None):
        strided_data = AR_striding(data,self.nlags)
        self.states_list.append(ARHMMStates(data.shape[0]-self.nlags,self.state_dim,self.obs_distns,self.dur_distns,self.trans_distn,self.init_state_distn,trunc=self.trunc,data=strided_data,stateseq=stateseq,nlags=self.nlags))

    def plot_observations(self,*args,**kwargs):
        pass


class ARHSMM(pyhsmm.models.HSMM):
    def __init__(self,nlags,*args,**kwargs):
        super(ARHSMM,self).__init__(*args,**kwargs)
        self.nlags = nlags

    def add_data(self,data,stateseq=None,censoring=None):
        strided_data = AR_striding(data,self.nlags)
        self.states_list.append(ARHSMMStates(data.shape[0]-self.nlags,self.state_dim,self.obs_distns,self.dur_distns,self.trans_distn,self.init_state_distn,trunc=self.trunc,data=strided_data,stateseq=stateseq,censoring=censoring,nlags=self.nlags))

    def plot_observations(self,*args,**kwargs):
        pass
