from __future__ import division
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

import pyhsmm

from pyhsmm.plugins.autoregressive.states import ARHMMStates, ARHSMMStates

# these exist only to play appropriate stride tricks on the data
# and to instantiate the appropriate states classes for forward generation

def AR_striding(data,nlags):
    if data.ndim == 1:
        data = np.reshape(data,(-1,1))
    sz = data.dtype.itemsize
    return ast(data,shape=(data.shape[0]-nlags,data.shape[1]*(nlags+1)),strides=(data.shape[1]*sz,sz))


class ARHMM(pyhsmm.models.HMM):
    def __init__(self,nlags,*args,**kwargs):
        super(ARHMM,self).__init__(*args,**kwargs)
        self.nlags = nlags

    def add_data(self,data,stateseq=None):
        strided_data = AR_striding(data,self.nlags)
        self.states_list.append(ARHMMStates(data.shape[0],self.state_dim,self.obs_distns,self.dur_distns,self.trans_distn,self.init_state_distn,trunc=self.trunc,data=strided_data,stateseq=stateseq))


class ARHSMM(pyhsmm.models.HSMM):
    def __init__(self,nlags,*args,**kwargs):
        super(ARHSMM,self).__init__(*args,**kwargs)
        self.nlags = nlags

    def add_data(self,data,stateseq=None,censoring=None):
        strided_data = AR_striding(data,self.nlags)
        self.states_list.append(ARHMMStates(data.shape[0],self.state_dim,self.obs_distns,self.dur_distns,self.trans_distn,self.init_state_distn,trunc=self.trunc,data=strided_data,stateseq=stateseq,censoring=censoring))
