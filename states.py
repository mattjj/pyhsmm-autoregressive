from __future__ import division
import numpy as np

import pyhsmm

# there are really just here to override generation; otherwise they are the same
# as the iid case

class ARHMMStates(pyhsmm.internals.states.HMMStatesPython):
    def __init__(self,*args,**kwargs):
        self.nlags = kwargs['nlags']
        del kwargs['nlags']
        super(ARHMMStates,self).__init__(*args,**kwargs)

    def generate_obs(self):
        # NOTE needs heuristic to generate the prefix
        raise NotImplementedError

class ARHMMStatesEigen(pyhsmm.internals.states.ARHMMStatesEigen):
    def __init__(self,*args,**kwargs):
        self.nlags = kwargs['nlags']
        del kwargs['nlags']
        super(ARHMMStates,self).__init__(*args,**kwargs)

    def generate_obs(self):
        # NOTE needs heuristic to generate the prefix
        raise NotImplementedError

class ARHSMMStates(pyhsmm.internals.states.HSMMStatesPython, ARHMMStates):
    def __init__(self,*args,**kwargs):
        self.nlags = kwargs['nlags']
        del kwargs['nlags']
        super(ARHSMMStates,self).__init__(*args,**kwargs)

    def generate_obs(self,*args,**kwargs):
        return ARHMMStates.generate_obs(self,*args,**kwargs)

