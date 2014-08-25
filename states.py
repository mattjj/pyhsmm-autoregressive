from __future__ import division
import numpy as np

from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen, \
        HMMStatesPossibleChangepoints
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

    def generate_obs(self):
        data = np.zeros((self.T+self.nlags,self.D))
        # TODO: first observations aren't modeled at the moment
        data[:self.nlags] = 10*np.random.normal(size=(self.nlags,self.D))
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

class ARHMMStatesPossibleChangepoints(_ARStatesMixin,HMMStatesPossibleChangepoints):
    def generate_obs(self):
        raise NotImplementedError

