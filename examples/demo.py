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

####################
#  make DAT MODEL  #
####################

Nmax = 20
model = m.ARHSMM(
        nlags=2,
        alpha=4.,gamma=4.,init_state_concentration=10.,
        obs_distns=[d.MNIW(dof=3,S=np.eye(2),M=np.zeros((2,4)),K=np.eye(4)) for state in range(Nmax)],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4) for state in range(Nmax)],
        )

model.add_data(data,trunc=50)

######################
#  do DAT INFERENCE  #
######################

for itr in progprint_xrange(100):
    model.resample_model()

plt.figure()
model.plot()

plt.figure()
colors = ['b','r','y','k','g']
stateseq = model.states_list[0].stateseq
for i,s in enumerate(np.unique(stateseq)):
    plt.plot(data[s==stateseq,0],data[s==stateseq,1],colors[i % len(colors)] + 'o')

