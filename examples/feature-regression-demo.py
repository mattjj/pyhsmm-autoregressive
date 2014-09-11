from __future__ import division

import numpy as np
from matplotlib import pyplot as plt

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
data_dim = data.shape[1]

# returns a 1D feature vector given the recent data in data_window
# data_window is windowsize x data_dim
def featurefn(data_window):
    return data_window.flatten() # linear features gives us regular AR

windowsize = 3
featuresize = featurefn(data[:windowsize]).shape[0]
affine = True

model = m.FeatureARHMM(
        windowsize=windowsize,
        featurefn=featurefn,
        alpha=4.,
        init_state_distn='uniform',
        obs_distns=[
            d.ARDRegression( # or just Regression
                nu_0=data_dim+3,
                S_0=np.eye(data_dim),
                M_0=np.zeros((data_dim,featuresize+affine)),
                a=10.,b=0.1,niter=10, # instead of K_0
                affine=affine)
            for state in range(Nmax)],
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
    plt.plot(data[windowsize:][s==stateseq,0],data[windowisze:][s==stateseq,1],
            color=cmap(colors[s]),linestyle='',marker='o')

plt.show()

