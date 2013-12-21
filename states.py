from __future__ import division

from pyhsmm.internals.states import HMMStatesPython, HMMStatesEigen, \
        HSMMStatesPython, HSMMStatesEigen

class _ARStatesMixin(object):
    def generate_obs(self):
        raise NotImplementedError # TODO

    # TODO what else does this class need?

class ARHMMStates(_ARStatesMixin,HMMStatesPython):
    pass

class ARHMMStatesEigen(_ARStatesMixin,HMMStatesEigen):
    pass

class ARHSMMStates(_ARStatesMixin,HSMMStatesPython):
    pass

class ARHSMMStatesEigen(_ARStatesMixin,HSMMStatesEigen):
    pass
