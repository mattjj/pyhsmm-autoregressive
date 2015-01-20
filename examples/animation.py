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

As = [0.99*np.hstack((-np.eye(2),2*np.eye(2))),
        np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2))),
        np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2)))]

truemodel = m.ARHSMM(
        alpha=2.,init_state_distn='uniform',
        obs_distns=[d.AutoRegression(A=A,sigma=np.eye(2)) for A in As],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=1*25,beta_0=1)
            for state in range(len(As))],
        )

data, labels = truemodel.generate(1000)
data += np.random.normal(size=data.shape) # some extra noise

fig, spa = plt.subplots(2,1)
spa[0].plot(data[:,0],data[:,1],'bx-')
spa[1].plot(data,'bx-')
spa[1].set_xlim(0,data.shape[0])
fig.suptitle('data')

truemodel.plot()
plt.gcf().suptitle('truth')

##################
#  create model  #
##################

Nmax = 10
affine = True
nlags = 2
model = m.ARWeakLimitStickyHDPHMM(
        alpha=4.,gamma=4.,kappa=50.,
        init_state_distn='uniform',
        obs_distns=[
            d.AutoRegression(
                nu_0=2.5,
                S_0=2.5*np.eye(2),
                M_0=np.zeros((2,2*nlags+affine)),
                K_0=10*np.eye(2*nlags+affine),
                affine=affine)
            for state in range(Nmax)],
        )

model.add_data(data)

###############
#  inference  #
###############

from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip

fig = model.make_figure()
model.plot(fig=fig,draw=False,plot_slice=slice(0,200))

def make_frame_mpl(t):
    model.resample_model()
    model.plot(fig=fig,update=True,draw=False,plot_slice=slice(0,200))
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame_mpl, duration=3)
animation.write_videofile('gibbs.mp4',fps=100)

