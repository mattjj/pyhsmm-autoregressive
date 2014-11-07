from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

np.random.seed(1)

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.basic.distributions import NegativeBinomialIntegerR2Duration

import autoregressive.models as m
import autoregressive.distributions as d

###################
#  generate data  #
###################

As = [np.hstack((-np.eye(2),2*np.eye(2))),
        np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2))),
        np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))]

truemodel = m.ARHSMM(
        alpha=4.,init_state_concentration=4.,
        obs_distns=[d.AutoRegression(A=A,sigma=0.1*np.eye(2)) for A in As],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=4*25,beta_0=4)
            for state in range(len(As))],
        )

data = truemodel.rvs(1000)

plt.figure()
plt.plot(data[:,0],data[:,1],'bx-')

##################
#  create model  #
##################

Nmax = 3
affine = True

obs_distns=[d.AutoRegression(
    nu_0=3, S_0=np.eye(2), M_0=np.zeros((2,4+affine)),
    K_0=np.eye(4+affine), affine=affine) for state in range(Nmax)]

dur_distns=[NegativeBinomialIntegerR2Duration(
    r_discrete_distn=np.r_[0,0,1],alpha_0=1.,beta_0=1.) for state in range(Nmax)]

model = m.FastARWeakLimitHDPHSMMTruncatedIntNegBin(
        alpha=4.,gamma=4.,init_state_concentration=10.,
        obs_distns=obs_distns,
        dur_distns=dur_distns,
        delay=3,
        dtype='float64',
        )

model.add_data(data)

###############
#  inference  #
###############

for itr in progprint_xrange(25):
    model.resample_model()

plt.figure()
model.plot()

plt.figure()
colors = ['b','r','y','k','g']
stateseq = model.states_list[0].stateseq
for i,s in enumerate(np.unique(stateseq)):
    plt.plot(data[s==stateseq,0],data[s==stateseq,1],colors[i % len(colors)] + 'o')

##########
#  blah  #
##########

# plt.close('all')

# s = model.states_list[0]

# # new fast code
# s.clear_caches()
# model.resample_states()
# ll1 = s._normalizer

# # very generic code
# s.clear_caches()
# model.resample_states_slow()
# ll2 = s._normalizer

# # old fast code
# s.clear_caches()
# model.resample_states_old()
# ll3 = model.log_likelihood()

# print ll1
# print ll2
# print ll3

# # A = s.hmm_trans_matrix_switched
# # smallA = s.trans_matrix

# # # let's zero out everything except the block diagonal!


# # from pyhsmm.util.general import cumsum

# # def mult(v):
# #     out = np.zeros(A.shape[1])
# #     outs = np.zeros(smallA.shape[0])
# #     delay = s.delays[0]
# #     rs, ps = s.rs, s.ps

# #     starts, ends = cumsum(s.rs+s.delays,strict=True), cumsum(s.rs+s.delays,strict=False)

# #     for m, (start,end) in enumerate(zip(starts,ends)):
# #         out[start] = 0
# #         out[start+1:start+delay] = v[start:start+delay-1]
# #         out[start+delay:start+delay+rs[m]] = ps[m] * v[start+delay:start+delay+rs[m]]
# #         out[start+delay+1:start+delay+rs[m]] += (1-ps[m])*v[start+delay:start+delay+rs[m]-1]
# #         out[start+delay] += v[start+delay-1]

# #         outs[m] = v[start+delay+rs[m]-1]

# #     for m, (start,end) in enumerate(zip(starts,ends)):
# #         out[start:start+rs[m]] += outs.dot(smallA[:,m] * (1-s.ps)) * s.bwd_enter_rows[m]

# #     return out

# # v = np.random.randn(A.shape[0])

# # print v.dot(A)
# # print mult(v)
# # print np.allclose(v.dot(A),mult(v))
