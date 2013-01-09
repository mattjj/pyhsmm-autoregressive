from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

import pyhsmm
from pyhsmm.util.text import progprint_xrange
import pyhsmm.plugins.autoregressive.models as m
import pyhsmm.plugins.autoregressive.distributions as d

###############
#  make data  #
###############

a = d.MNIW(3,10*np.eye(2),np.zeros((2,4)),100*np.eye(4))
# a.A = np.hstack((-np.eye(2),2*np.eye(2)))

b = d.MNIW(3,10*np.eye(2),np.zeros((2,4)),100*np.eye(4))
# b.A = np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))

c = d.MNIW(3,10*np.eye(2),np.zeros((2,4)),100*np.eye(4))
# c.A = np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))


data = np.array([0,1]).repeat(2).reshape((2,2))
distns = [a,b,c]
for i in range(4):
    data = np.concatenate((data,distns[i % len(distns)].rvs(prefix=data[-2:],length=25)))

# plt.figure()
# plt.plot(data[:,0],data[:,1],'bx-')

### empirical bayes!

# ddata = np.diff(data,axis=0)
# dmu = ddata.mean(0)
# dcov = ddata.T.dot(ddata)/ddata.shape[0] - np.outer(dmu,dmu)

# K = *np.eye(2*data.shape[1])
# S_0 = 5*dcov

### construct dat HSMM

####################
#  make DAT MODEL  #
####################

Nmax = 10
model = m.ARHSMM(
        nlags=2,
        alpha=4.,gamma=4.,init_state_concentration=2.,
        # obs_distns=[d.MNIW(data.shape[1]+2,S_0,np.zeros((S_0.shape[0],K.shape[0])),K) for state in range(Nmax)],
        obs_distns=[d.MNIW(3,50*np.eye(2),np.zeros((2,4)),np.eye(4)) for state in range(Nmax)],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4) for state in range(Nmax)],
        trunc=50
        )

model.add_data(data)

# ######################
# #  do DAT INFERENCE  #
# ######################

for itr in progprint_xrange(50):
    model.resample_model()

plt.figure()
model.plot()

plt.figure()
colors = ['b','r','y','k','g']
stateseq = model.states_list[0].stateseq
# TODO improve plotting
for i,s in enumerate(np.unique(stateseq)):
    plt.plot(data[s==stateseq,0],data[s==stateseq,1],colors[i % len(colors)] + 'o')

