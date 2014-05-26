from __future__ import division
from pyhsmm.internals.hmm_states import HMMStatesPython, HMMStatesEigen
from pyhsmm.internals.hsmm_states import HSMMStatesPython, HSMMStatesEigen, \
        GeoHSMMStates

class _ARStatesMixin(object):
    def generate_obs(self):
        raise NotImplementedError # TODO

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

