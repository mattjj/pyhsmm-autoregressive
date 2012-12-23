from __future__ import division
import numpy as np

import pyhsmm

# TODO
# * possible changepoints version

class ARHMM(pyhsmm.models.HMM):
    def resample_model(self):
        for state, distn in enumerate(self.obs_distns):
            distn.resample([(s.data,np.where(s.stateseq == state)[0]) for s in self.states_list])

        self.trans_distn.resample([s.stateseq for s in self.states_list])

        self.init_state_distn.resample([s.stateseq[0] for s in self.states_list])

        for s in self.states_list:
            s.resample()

class ARHSMM(ARHMM, pyhsmm.models.HSMM):
    def resample_model(self):
        pyhsmm.models.HSMM.resample_model(self)
