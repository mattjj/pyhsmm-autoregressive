from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.stats import whiten, cov

import autoregressive.models as m
import autoregressive.distributions as d

###################
#  generate data  #
###################

As = [0.99*np.hstack((-np.eye(2),2*np.eye(2))),
        np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2))),
        np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))]

truemodel = m.ARHSMM(
        alpha=4.,init_state_concentration=4.,
        obs_distns=[d.AutoRegression(A=A,sigma=np.eye(2)) for A in As],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4)
            for state in range(len(As))],
        )

data, labels = truemodel.generate(500)

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')
plt.gcf().suptitle('data')

plt.figure()
truemodel.plot()
plt.gcf().suptitle('truth')

##################
#  create model  #
##################

Nmax = 10
affine = True
nlags = 3
model = m.FastARHMM(
        alpha=4.,
        init_state_distn='uniform',
        obs_distns=[
            d.ARDAutoRegression(
                nu_0=3,
                S_0=np.eye(2),
                M_0=np.zeros((2,2*nlags+affine)),
                a=10.,b=0.1,blocksizes=[2]*nlags + ([1] if affine else []),
                niter=10,
                affine=affine)
            for state in range(Nmax)],
        dtype='float32',
        )

model.add_data(data)

###############
#  inference  #
###############

for itr in progprint_xrange(100):
    model.resample_model()

plt.figure()
model.plot()
plt.gcf().suptitle('sampled')

plt.figure()
colors = model._get_colors()
cmap = plt.get_cmap()
stateseq = model.states_list[0].stateseq
for i,s in enumerate(np.unique(stateseq)):
    plt.plot(data[model.nlags:][s==stateseq,0],data[model.nlags:][s==stateseq,1],
            color=cmap(colors[s]),linestyle='',marker='o')

