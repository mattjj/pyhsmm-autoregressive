from __future__ import division
import numpy as np

import pyhsmm

# there are really just here to override generation; otherwise they are the same
# as the iid case

class ARHMMStates(pyhsmm.internals.states.HMMStatesPython):
    def generate(self):
        raise NotImplementedError

class ARHSMMStates(pyhsmm.internals.states.HSMMStatesPython):
    def generate(self):
        raise NotImplementedError

