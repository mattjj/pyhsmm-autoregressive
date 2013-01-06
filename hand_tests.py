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

a = d.MNIW(3,0.1*np.eye(2),np.zeros((2,4)),10*np.eye(4))
a.A = np.hstack((-np.eye(2),2*np.eye(2)))
data1 = a.rvs(prefix=np.array([0,1]).repeat(2).reshape((2,2)),length=50)

b = d.MNIW(3,0.1*np.eye(2),np.zeros((2,4)),10*np.eye(4))
b.A = np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))
data2 = b.rvs(prefix=data1[-2:],length=50)

data3 = a.rvs(prefix=data2[-2:],length=50)

data = np.vstack((data1,data2,data3))

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')

### empirical bayes!

# ddata = np.diff(data,axis=0)
# dmu = ddata.mean(0)
# dcov = ddata.T.dot(ddata)/ddata.shape[0] - np.outer(dmu,dmu)

# K = 0.1*np.eye(2*data.shape[1])
# S_0 = 5*dcov

### construct dat HSMM

####################
#  make DAT MODEL  #
####################

Nmax = 5
model = m.ARHSMM(
        nlags=2,
        alpha=2.,gamma=2.,init_state_concentration=2.,
        # obs_distns=[d.MNIW(data.shape[1]+2,S_0,np.zeros((S_0.shape[0],K.shape[0])),K) for state in range(Nmax)],
        obs_distns=[d.MNIW(3,0.1*np.eye(2),np.zeros((2,4)),10*np.eye(4)) for state in range(Nmax)],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*50,beta_0=4) for state in range(Nmax)]
        )

model.add_data(data)

######################
#  do DAT INFERENCE  #
######################

for itr in progprint_xrange(25):
    model.resample_model()

plt.figure()
model.plot()
