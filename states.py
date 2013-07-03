from __future__ import division

import pyhsmm

# there are really just here to override generation; otherwise they are the same
# as the iid case

class ARHMMStates(pyhsmm.internals.states.HMMStatesPython):
    def __init__(self,nlags,*args,**kwargs):
        self.nlags = nlags
        super(ARHMMStates,self).__init__(*args,**kwargs)

    def generate_obs(self):
        raise NotImplementedError


class ARHMMStatesEigen(pyhsmm.internals.states.HMMStatesEigen):
    def __init__(self,nlags,*args,**kwargs):
        self.nlags = nlags
        super(ARHMMStatesEigen,self).__init__(*args,**kwargs)


class ARHSMMStates(pyhsmm.internals.states.HSMMStatesPython):
    def __init__(self,nlags,*args,**kwargs):
        self.nlags = nlags
        super(ARHSMMStates,self).__init__(*args,**kwargs)

    def generate_obs(self,*args,**kwargs):
        raise NotImplementedError

