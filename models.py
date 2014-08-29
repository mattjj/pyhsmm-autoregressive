from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import pyhsmm
from pyhsmm.util.general import rle, cumsum
from pyhsmm.basic.distributions import Gaussian

from util import AR_striding, undo_AR_striding
from autoregressive.states import ARHMMStates, ARHSMMStates, \
        ARHMMStatesEigen, ARHSMMStatesEigen, ARHSMMStatesGeo, \
        ARHSMMStatesIntegerNegativeBinomialStates

class _ARMixin(object):
    def __init__(self,init_emission_distn=None,**kwargs):
        super(_ARMixin,self).__init__(**kwargs)
        if init_emission_distn is None:
            init_emission_distn = \
                    Gaussian(nu_0=self.P+10,sigma_0=np.eye(self.P),
                        mu_0=np.zeros(self.P),kappa_0=1.)
        self.init_emission_distn = init_emission_distn

    def add_data(self,data,strided=False,**kwargs):
        strided_data = AR_striding(data,self.nlags) if not strided else data
        super(_ARMixin,self).add_data(data=strided_data,**kwargs)

    ### Gibbs

    def resample_parameters(self):
        super(_ARMixin,self).resample_parameters()
        self.resample_init_emission_distn()

    def resample_init_emission_distn(self):
        self.init_emission_distn.resample(
                [s.data[:self.nlags].ravel() for s in self.states_list])

    ### prediction

    def predict(self,seed_data,timesteps,with_noise=False):
        assert seed_data.shape[0] >= self.nlags

        full_data = np.vstack((seed_data,np.nan*np.ones((timesteps,self.D))))
        self.add_data(full_data)
        s = self.states_list.pop()
        s.resample() # fills in extra states

        if with_noise:
            for state, row in zip(s.stateseq[-timesteps:],s.data[-timesteps:]):
                row[-self.D:] = self.obs_distns[state].rvs(lagged_data=row[:-self.D])
        else:
            for state, row in zip(s.stateseq[-timesteps:],s.data[-timesteps:]):
                o = self.obs_distns[state]
                if not o.affine:
                    row[-self.D:] = o.A.dot(row[:-self.D])
                else:
                    row[-self.D:] = o.A[:,:-1].dot(row[:-self.D]) + np.squeeze(o.A[:,-1])

        return full_data

    def fill_in(self,data):
        raise NotImplementedError

    ### convenient properties

    @property
    def nlags(self):
        return self.obs_distns[0].nlags

    @property
    def D(self):
        return self.obs_distns[0].D

    @property
    def P(self):
        return self.D*self.nlags

    ### plotting

    def plot_observations(self,colors=None,states_objs=None):
        if colors is None:
            colors = self._get_colors()
        if states_objs is None:
            states_objs = self.states_list

        cmap = cm.get_cmap()

        for s in states_objs:
            data = undo_AR_striding(s.data,self.nlags)

            stateseq_norep, durs = rle(s.stateseq)
            starts = np.concatenate(((0,),durs.cumsum()))
            for state,start,dur in zip(stateseq_norep,starts,durs):
                plt.plot(
                        np.arange(start,start+data[start:start+dur].shape[0]),
                        data[start:start+dur],
                        color=cmap(colors[state]))
            plt.xlim(0,s.T-1)

class _HMMFastResamplingMixin(_ARMixin):
    _obs_stats = None

    def resample_states(self,**kwargs):
        from messages import resample_arhmm
        if len(self.states_list) > 0:
            stateseqs = [np.empty(s.T,dtype='int32') for s in self.states_list]
            params, normalizers = map(np.array,zip(*[o._param_matrix for o in self.obs_distns]))
            stats, loglikes = resample_arhmm(
                    s.pi_0,s.trans_matrix,
                    params,normalizers,
                    [undo_AR_striding(s.data,self.nlags) for s in self.states_list],
                    stateseqs,
                    [np.random.uniform(size=s.T) for s in self.states_list],
                    self.alphans)
            for s, stateseq, loglike in zip(self.states_list,stateseqs,loglikes):
                s.stateseq = stateseq
                s._normalizer = loglike

            self._obs_stats = stats
        else:
            self._obs_stats = None

    def resample_obs_distns(self):
        if self._obs_stats is not None:
            for o, statmat in zip(self.obs_distns,self._obs_stats):
                o.resample(stats=statmat)
        else:
            for o in self.obs_distns:
                o.resample()

    @property
    def alphans(self):
        if not hasattr(self,'_alphans'):
            self._alphans = [np.empty((s.T,self.num_states)) for s in self.states_list]
        return self._alphans

class _INBHSMMFastResamplingMixin(_ARMixin):
    _obs_stats = None

    def resample_states(self,**kwargs):
        from messages import resample_arhmm
        from itertools import repeat, chain
        if len(self.states_list) > 0:
            stateseqs = [np.empty(s.T,dtype='int32') for s in self.states_list]
            params, normalizers = \
                map(np.array,zip(*[o._param_matrix
                    for o in chain(*[repeat(o,r) for o, r in zip(self.obs_distns,s.rs)])]))
            assert params.shape[0] == s.hmm_bwd_pi_0.shape[0] == normalizers.shape[0] \
                    == s.hmm_bwd_trans_matrix.shape[0]
            stats, loglikes = resample_arhmm(
                    s.hmm_bwd_pi_0,s.hmm_bwd_trans_matrix,
                    params,normalizers,
                    [undo_AR_striding(s.data,self.nlags) for s in self.states_list],
                    stateseqs,
                    [np.random.uniform(size=s.T) for s in self.states_list],
                    self.alphans)
            for s, stateseq, loglike in zip(self.states_list,stateseqs,loglikes):
                s.stateseq = stateseq
                s._map_states()
                s._normalizer = loglike

            starts, ends = cumsum(s.rs,strict=True), cumsum(s.rs,strict=False)
            stats = map(np.array,stats)
            stats = [sum(stats[start:end]) for start, end in zip(starts,ends)]
            assert len(stats) == len(self.obs_distns)

            self._obs_stats = stats
        else:
            self._obs_stats = None

    def resample_obs_distns(self):
        if self._obs_stats is not None:
            for o, statmat in zip(self.obs_distns,self._obs_stats):
                o.resample(stats=statmat)
        else:
            for o in self.obs_distns:
                o.resample()

    @property
    def alphans(self):
        return [np.empty((s.T,sum(s.rs))) for s in self.states_list]


class ARHMM(_HMMFastResamplingMixin,pyhsmm.models.HMM):
    _states_class = ARHMMStatesEigen

class ARWeakLimitHDPHMM(_HMMFastResamplingMixin,pyhsmm.models.WeakLimitHDPHMM):
    _states_class = ARHMMStatesEigen


class ARHSMM(_ARMixin,pyhsmm.models.HSMM):
    _states_class = ARHSMMStatesEigen

class ARWeakLimitHDPHSMM(_ARMixin,pyhsmm.models.WeakLimitHDPHSMM):
    _states_class = ARHSMMStatesEigen

class ARWeakLimitStickyHDPHMM(_ARMixin,pyhsmm.models.WeakLimitStickyHDPHMM):
    _states_class = ARHMMStatesEigen

class ARWeakLimitHDPHSMMIntNegBin(
        _INBHSMMFastResamplingMixin,
        pyhsmm.models.WeakLimitHDPHSMMIntNegBin):
    _states_class = ARHSMMStatesIntegerNegativeBinomialStates

class ARWeakLimitGeoHDPHSMM(_ARMixin,pyhsmm.models.WeakLimitGeoHDPHSMM):
    _states_class = ARHSMMStatesGeo

