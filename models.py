from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import pyhsmm
from pyhsmm.models import _SeparateTransMixin
from pyhsmm.util.general import rle, cumsum
from pyhsmm.basic.distributions import Gaussian

from util import AR_striding, undo_AR_striding

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

    def generate(self,T,keep=True):
        s = self._states_class(model=self,T=T,initialize_from_prior=True)
        data = self._generate_obs(s)
        if keep:
            self.states_list.append(s)
        else:
            return data, s.stateseq

    def _generate_obs(self,states_obj):
        data = np.zeros((states_obj.T+self.nlags,self.D))

        if hasattr(self,'prefix'):
            data[:self.nlags] = self.prefix
        else:
            data[:self.nlags] = self.init_emission_distn\
                    .rvs().reshape(data[:self.nlags].shape)

        for idx, state in enumerate(states_obj.stateseq):
            data[idx+self.nlags] = \
                self.obs_distns[state].rvs(lagged_data=data[idx:idx+self.nlags])

        states_obj.data = AR_striding(data,self.nlags)

        return data

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

        return undo_AR_striding(s.data,nlags=self.nlags)

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
                        np.arange(start,start+data[start:start+dur+1].shape[0]),
                        data[start:start+dur+1],
                        color=cmap(colors[state]))
            plt.xlim(0,s.T-1)

###################
#  model classes  #
###################

class ARHMM(_ARMixin,pyhsmm.models.HMM):
    pass

class ARWeakLimitHDPHMM(_ARMixin,pyhsmm.models.WeakLimitHDPHMM):
    pass


class ARHSMM(_ARMixin,pyhsmm.models.HSMM):
    pass

class ARWeakLimitHDPHSMM(_ARMixin,pyhsmm.models.WeakLimitHDPHSMM):
    pass


class ARWeakLimitStickyHDPHMM(_ARMixin,pyhsmm.models.WeakLimitStickyHDPHMM):
    pass


class ARWeakLimitHDPHSMMIntNegBin(_ARMixin,pyhsmm.models.WeakLimitHDPHSMMIntNegBin):
    pass


class ARWeakLimitGeoHDPHSMM(_ARMixin,pyhsmm.models.WeakLimitGeoHDPHSMM):
    pass


class ARHMMSeparateTrans(_ARMixin,pyhsmm.models.HMMSeparateTrans):
    pass

class ARWeakLimitHDPHMMSeparateTrans(_ARMixin,pyhsmm.models.WeakLimitHDPHMMSeparateTrans):
    pass

class ARWeakLimitHDPHSMMIntNegBinSeparateTrans(_ARMixin,pyhsmm.models.WeakLimitHDPHSMMIntNegBin):
    pass


class ARWeakLimitHDPHSMMDelayedIntNegBin(
        _ARMixin,
        pyhsmm.models.WeakLimitHDPHSMMDelayedIntNegBin):
    pass

class ARWeakLimitHDPHSMMDelayedIntNegBinSeparateTrans(
        _ARMixin,
        pyhsmm.models.WeakLimitHDPHSMMDelayedIntNegBinSeparateTrans):
    pass

###########################
#  low-level code mixins  #
###########################

class _HMMFastResamplingMixin(_ARMixin):
    _obs_stats = None
    _transcounts = []

    def __init__(self,dtype='float32',**kwargs):
        self.dtype = dtype
        super(_HMMFastResamplingMixin,self).__init__(**kwargs)

    def add_data(self,data,**kwargs):
        super(_HMMFastResamplingMixin,self).add_data(data.astype(self.dtype),**kwargs)

    def resample_states(self,**kwargs):
        from messages import resample_arhmm
        if len(self.states_list) > 0:
            stateseqs = [np.empty(s.T,dtype='int32') for s in self.states_list]
            params, normalizers = map(np.array,zip(*[o._param_matrix for o in self.obs_distns]))
            stats, transcounts, loglikes = resample_arhmm(
                    [s.pi_0.astype(self.dtype) for s in self.states_list],
                    [s.trans_matrix.astype(self.dtype) for s in self.states_list],
                    params.astype(self.dtype), normalizers.astype(self.dtype),
                    [undo_AR_striding(s.data,self.nlags) for s in self.states_list],
                    stateseqs,
                    [np.random.uniform(size=s.T).astype(self.dtype) for s in self.states_list],
                    self.alphans)
            for s, stateseq, loglike in zip(self.states_list,stateseqs,loglikes):
                s.stateseq = stateseq
                s._normalizer = loglike

            self._obs_stats = stats
            self._transcounts = transcounts
        else:
            self._obs_stats = None
            self._transcounts = []

    def resample_obs_distns(self):
        if self._obs_stats is not None:
            for o, statmat in zip(self.obs_distns,self._obs_stats):
                o.resample(stats=statmat)
        else:
            for o in self.obs_distns:
                o.resample()

    # def resample_trans_distn(self):
    #     self.trans_distn.resample(trans_counts=self._transcounts)

    @property
    def alphans(self):
        if not hasattr(self,'_alphans'):
            self._alphans = [np.empty((s.T,self.num_states),
                dtype=self.dtype) for s in self.states_list]
        return self._alphans

class _INBHSMMFastResamplingMixin(_ARMixin):
    _obs_stats = None

    def __init__(self,dtype='float32',**kwargs):
        self.dtype = dtype
        super(_INBHSMMFastResamplingMixin,self).__init__(**kwargs)

    def add_data(self,data,**kwargs):
        super(_INBHSMMFastResamplingMixin,self).add_data(data.astype(self.dtype),**kwargs)

    def resample_states(self,**kwargs):
        # TODO only use this when the number/size of sequences warrant it
        from messages import resample_arhmm
        if len(self.states_list) > 0:
            stateseqs = [np.empty(s.T,dtype='int32') for s in self.states_list]
            params, normalizers = map(np.array,zip(*[o._param_matrix for o in self.obs_distns]))
            params, normalizers = params.repeat(s.rs,axis=0), normalizers.repeat(s.rs,axis=0)
            stats, _, loglikes = resample_arhmm(
                    [s.hmm_pi_0.astype(self.dtype) for s in self.states_list],
                    [s.hmm_trans_matrix.astype(self.dtype) for s in self.states_list],
                    params.astype(self.dtype), normalizers.astype(self.dtype),
                    [undo_AR_striding(s.data,self.nlags) for s in self.states_list],
                    stateseqs,
                    [np.random.uniform(size=s.T).astype(self.dtype) for s in self.states_list],
                    self.alphans)
            for s, stateseq, loglike in zip(self.states_list,stateseqs,loglikes):
                s.stateseq = stateseq
                s._map_states()
                s._normalizer = loglike

            starts, ends = cumsum(s.rs,strict=True), cumsum(s.rs,strict=False)
            stats = map(np.array,stats)
            stats = [sum(stats[start:end]) for start, end in zip(starts,ends)]

            self._obs_stats = stats
        else:
            self._obs_stats = None

    # TODO IIRC this is behaving erratically; may be numerical, needs auto-tests
    # def resample_obs_distns(self):
    #     if self._obs_stats is not None:
    #         for o, statmat in zip(self.obs_distns,self._obs_stats):
    #             o.resample(stats=statmat)
    #     else:
    #         for o in self.obs_distns:
    #             o.resample()

    @property
    def alphans(self):
        return [np.empty((s.T,sum(s.rs)), dtype=self.dtype) for s in self.states_list]


class FastARHMM(_HMMFastResamplingMixin,pyhsmm.models.HMM):
    pass

class FastARWeakLimitHDPHMM(_HMMFastResamplingMixin,pyhsmm.models.WeakLimitHDPHMM):
    pass

class FastARWeakLimitStickyHDPHMM(_HMMFastResamplingMixin,pyhsmm.models.WeakLimitStickyHDPHMM):
    pass

class FastARWeakLimitHDPHSMMIntNegBin(
        _INBHSMMFastResamplingMixin,
        pyhsmm.models.WeakLimitHDPHSMMIntNegBin):
    pass



class _FastDelayedMixin(_INBHSMMFastResamplingMixin):
    # NOTE: basically uses s.rs+s.delays instead of just s.rs

    def resample_states(self,**kwargs):
        from messages import resample_arhmm
        if len(self.states_list) > 0:
            stateseqs = [np.empty(s.T,dtype='int32') for s in self.states_list]
            params, normalizers = map(np.array,zip(*[o._param_matrix for o in self.obs_distns]))
            params, normalizers = \
                    params.repeat(s.rs+s.delays,axis=0), normalizers.repeat(s.rs+s.delays,axis=0)
            stats, _, loglikes = resample_arhmm(
                    [s.hmm_pi_0.astype(self.dtype) for s in self.states_list],
                    [s.hmm_trans_matrix.astype(self.dtype) for s in self.states_list],
                    params.astype(self.dtype), normalizers.astype(self.dtype),
                    [undo_AR_striding(s.data,self.nlags) for s in self.states_list],
                    stateseqs,
                    [np.random.uniform(size=s.T).astype(self.dtype) for s in self.states_list],
                    self.alphans)
            for s, stateseq, loglike in zip(self.states_list,stateseqs,loglikes):
                s.stateseq = stateseq
                s._map_states()
                s._normalizer = loglike

            starts, ends = cumsum(s.rs+s.delays,strict=True), cumsum(s.rs+s.delays,strict=False)
            stats = map(np.array,stats)
            stats = [sum(stats[start:end]) for start, end in zip(starts,ends)]

            self._obs_stats = stats
        else:
            self._obs_stats = None

    @property
    def alphans(self):
        return [np.empty((s.T,sum(s.rs+s.delays)), dtype=self.dtype) for s in self.states_list]


class FastARWeakLimitHDPHSMMDelayedIntNegBin(
        _FastDelayedMixin,
        pyhsmm.models.WeakLimitHDPHSMMDelayedIntNegBin):
    pass

class FastARWeakLimitHDPHSMMDelayedIntNegBinSeparateTrans(
        _FastDelayedMixin,
        pyhsmm.models.WeakLimitHDPHSMMDelayedIntNegBinSeparateTrans):
    pass

