from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.basic.distributions import NegativeBinomialIntegerR2Duration

import autoregressive.models as m
import autoregressive.distributions as d

np.random.seed(1)

###################
#  generate data  #
###################

As = [np.hstack((-np.eye(2),2*np.eye(2))),
        np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2))),
        np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))]

truemodel = m.ARHSMM(
        alpha=10.,init_state_concentration=4.,
        obs_distns=[d.AutoRegression(A=A,sigma=0.1*np.eye(2)) for A in As],
        dur_distns=
        [pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4)]+[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*6,beta_0=4)]*2
        )

data = truemodel.rvs(500)

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')

##################
#  create model  #
##################

Nmax = 20
affine = True

obs_distns=[d.AutoRegression(
    nu_0=3, S_0=np.eye(2), M_0=np.zeros((2,4+affine)),
    K_0=np.eye(4+affine), affine=affine) for state in range(Nmax)]

dur_distns=[NegativeBinomialIntegerR2Duration(
    r_discrete_distn=np.ones(10.),alpha_0=1.,beta_0=1.) for state in range(Nmax)]

model = m.FastARWeakLimitHDPHSMMDelayedIntNegBin(
        alpha=4.,gamma=4.,init_state_concentration=10.,
        obs_distns=obs_distns,
        dur_distns=dur_distns,
        delay=2, # minimum duration is 3
        dtype='float64',
        )

model.add_data(data)

###############
#  inference  #
###############

for itr in progprint_xrange(100):
    model.resample_model()

plt.figure()
model.plot()

plt.figure()
colors = ['b','r','y','k','g']
stateseq = model.states_list[0].stateseq
for i,s in enumerate(np.unique(stateseq)):
    plt.plot(data[s==stateseq,0],data[s==stateseq,1],colors[i % len(colors)] + 'o')

