from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

import pyhsmm
from pyhsmm.util.text import progprint_xrange
import autoregressive.models as m
import autoregressive.distributions as d

np.seterr(over='raise')

###################
#  generate data  #
###################

a = d.MNIW(dof=3,S=np.eye(2),M=np.zeros((2,4)),K=np.eye(4))
a.A = np.hstack((-np.eye(2),2*np.eye(2)))

b = d.MNIW(dof=3,S=np.eye(2),M=np.zeros((2,4)),K=np.eye(4))
b.A = np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))

c = d.MNIW(dof=3,S=np.eye(2),M=np.zeros((2,4)),K=np.eye(4))
c.A = np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))


data = np.array([0,2]).repeat(2).reshape((2,2))
distns = [a,b,c]
for i in range(9):
    data = np.concatenate((data,distns[i % len(distns)].rvs(prefix=data[-2:],length=30)))

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')

##################
#  create model  #
##################

Nmax = 20
model = m.ARHMM(
        nlags=2,
        alpha=4.,gamma=4.,init_state_concentration=10.,
        obs_distns=[d.MNIW(dof=3,S=np.eye(2),M=np.zeros((2,4)),K=np.eye(4)) for state in range(Nmax)],
        )

model.add_data(data)

###############
#  inference  #
###############

print 'Gibbs sampling initialization'
for itr in progprint_xrange(10):
    model.resample_model()

print 'EM'
for itr in progprint_xrange(50):
    model.EM_step()

plt.figure()
model.plot()

plt.figure()
colors = ['b','r','y','k','g']
stateseq = model.states_list[0].stateseq
for i,s in enumerate(np.unique(stateseq)):
    plt.plot(data[s==stateseq,0],data[s==stateseq,1],colors[i % len(colors)] + 'o')

