from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.general import labels_to_changepoints

import autoregressive.models as am
import autoregressive.distributions as ad

Nmax = 10
affine = True
nlags = 3

###################
#  generate data  #
###################

As = [0.99*np.hstack((-np.eye(2),2*np.eye(2))),
        np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2))),
        np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))]

truemodel = am.ARHSMM(
        alpha=4.,init_state_concentration=4.,
        obs_distns=[ad.AutoRegression(A=A,sigma=np.eye(2)) for A in As],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4)
            for state in range(len(As))],
        )

datas, labels = zip(*[truemodel.generate(500) for _ in range(2)])

##########################
#  get the changepoints  #
##########################

changepoints = [[(a+2,b+2) for a,b in labels_to_changepoints(l)] for l in labels]
for c in changepoints:
    c[0] = (0,c[0][1])

#########################
#  model and inference  #
#########################

model = am.ARWeakLimitHDPHSMMPossibleChangepointsSeparateTrans(
        alpha_a_0=1.,alpha_b_0=1./10,
        gamma_a_0=1.,gamma_b_0=1./10,
        init_state_distn='uniform',
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4)
            for state in range(Nmax)],
        obs_distns=[
            ad.ARDAutoRegression(
                nu_0=3,
                S_0=np.eye(2),
                M_0=np.zeros((2,2*nlags+affine)),
                a=10.,b=0.1,niter=10, # instead of K_0
                affine=affine)
            for state in range(Nmax)],
        )

for data, c in zip(datas,changepoints):
    model.add_data(data=data,changepoints=c,group_id=0)

###############
#  inference  #
###############

for itr in progprint_xrange(25):
    model.resample_model(joblib_jobs=2)

