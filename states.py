from __future__ import division
import numpy as np

from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen
from pyhsmm.internals.hsmm_states import HSMMStatesPython, HSMMStatesEigen, \
        GeoHSMMStates

from util import AR_striding

class _ARStatesMixin(object):
    @property
    def D(self):
        return self.obs_distns[0].D

    @property
    def nlags(self):
        return self.obs_distns[0].nlags

    @property
    def init_emission_distns(self):
        return self.model.init_emission_distns

    def generate_obs(self):
        data = np.zeros((self.T+self.nlags,self.D))
        data[:self.nlags] = self.init_emission_distns[self.stateseq[0]]\
                .rvs().reshape(data[:self.nlags].shape)
        for idx, state in enumerate(self.stateseq):
            data[idx+self.nlags] = \
                self.obs_distns[state].rvs(lagged_data=data[idx:idx+self.nlags])
        self.data = AR_striding(data,self.nlags)

        return data

class ARHMMStates(_ARStatesMixin,HMMStatesPython):
    pass

class ARHMMStatesEigen(_ARStatesMixin,HMMStatesEigen):
    pass

class ARHSMMStates(_ARStatesMixin,HSMMStatesPython):
    pass

class ARHSMMStatesEigen(_ARStatesMixin,HSMMStatesEigen):
    pass

# class ARHSMMStatesIntegerNegativeBinomial(_ARStatesMixin,HSMMStatesIntegerNegativeBinomial):
#     pass

class ARHSMMStatesGeo(_ARStatesMixin,GeoHSMMStates):
    pass

