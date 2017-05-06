from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
np.random.seed(0)

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.stats import whiten, cov

import autoregressive.models as m
import autoregressive.distributions as d

###################
#  generate data  #
###################
T = 5000
D_obs = 2
As = [0.99*np.hstack((-np.eye(2),2*np.eye(2))),
        np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2))),
        np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))]

truemodel = m.ARHSMM(
        alpha=4.,init_state_concentration=4.,
        obs_distns=[d.AutoRegression(A=A,sigma=np.eye(2)) for A in As],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4)
            for state in range(len(As))],
        )

datas = []
labels = []
for t in range(0,T,500):
    data, label = truemodel.generate(500, keep=True)
    datas.append(data)
    labels.append(label)

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')
plt.gcf().suptitle('data')

truemodel.plot()
plt.gcf().suptitle('truth')

##################
#  create model  #
##################

Nmax = 10
affine = False
nlags = 3

# Construct a standard AR-HMM for fitting
model = m.ARHMM(
        alpha=4.,
        init_state_distn='uniform',
        obs_distns=[
            d.AutoRegression(
                nu_0=3,
                S_0=np.eye(D_obs),
                M_0=np.hstack((np.eye(D_obs), np.zeros((D_obs, D_obs*(nlags-1)+affine)))),
                K_0=np.eye(D_obs*nlags+affine),
                affine=affine)
            for state in range(Nmax)],
        )

# Or construct a sticky AR-HMM with a Bayesian nonparametric
# prior on the number of states, i.e. an HDP-HMM. We'll do
# inference with a "weak-limit" approximation of the HDP.
model = m.ARWeakLimitStickyHDPHMM(
        alpha=4., gamma=4., kappa=10., 
        init_state_distn='uniform',
        obs_distns=[
            d.AutoRegression(
                nu_0=3,
                S_0=np.eye(D_obs),
                M_0=np.hstack((np.eye(D_obs), np.zeros((D_obs, D_obs*(nlags-1)+affine)))),
                K_0=np.eye(D_obs*nlags+affine),
                affine=affine)
            for state in range(Nmax)],
        )

for data in datas:
    model.add_data(data)

###############
#  inference  #
###############

for itr in progprint_xrange(100):
    model.resample_model()

model.plot()
plt.gcf().suptitle('sampled')

