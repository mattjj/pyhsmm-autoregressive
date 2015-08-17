from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import pyhsmm
from pyhsmm.basic.distributions import Gaussian
from pyhsmm.util.general import cumsum
from pybasicbayes.util.general import blockarray

from util import AR_striding, undo_AR_striding


class _ARMixin(object):
    def __init__(self,init_emission_distn=None,**kwargs):
        super(_ARMixin,self).__init__(**kwargs)
        if init_emission_distn is None:
            init_emission_distn = \
                    Gaussian(nu_0=self.P+1,sigma_0=10*self.P*np.eye(self.P),
                        mu_0=np.zeros(self.P),kappa_0=1.)
        self.init_emission_distn = init_emission_distn

    def add_data(self,data,strided=False,**kwargs):
        strided_data = AR_striding(data,self.nlags) if not strided else data
        super(_ARMixin,self).add_data(data=strided_data,**kwargs)

    def _generate_obs(self,s):
        if s.data is None:
            # generating brand new data sequence
            data = np.zeros((s.T+self.nlags,self.D))

            if hasattr(self,'prefix'):
                data[:self.nlags] = self.prefix
            else:
                data[:self.nlags] = self.init_emission_distn\
                    .rvs().reshape(data[:self.nlags].shape)

            for idx, state in enumerate(s.stateseq):
                data[idx+self.nlags] = \
                    self.obs_distns[state].rvs(lagged_data=data[idx:idx+self.nlags])

            s.data = AR_striding(data,self.nlags)

        else:
            # filling in missing data
            data = undo_AR_striding(s.data,self.nlags)

            # TODO should sample from init_emission_distn if there are nans in
            # data[:self.nlags]
            assert not np.isnan(data[:self.nlags]).any(), "can't have missing data (nans) in prefix"

            nan_idx, = np.where(np.isnan(data[self.nlags:]).any(1))
            for idx, state in zip(nan_idx,s.stateseq[nan_idx]):
                data[idx+self.nlags] = \
                    self.obs_distns[state].rvs(lagged_data=data[idx:idx+self.nlags])

        return data

    ### Gibbs

    def resample_parameters(self):
        super(_ARMixin,self).resample_parameters()
        self.resample_init_emission_distn()

    def resample_init_emission_distn(self):
        self.init_emission_distn.resample(
                [s.data[:self.nlags].ravel() for s in self.states_list])

    def _get_joblib_pair(self,s):
        return (undo_AR_striding(s.data,self.nlags),s._kwargs)

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

    @property
    def datas(self):
        return [undo_AR_striding(s.data,self.nlags) for s in self.states_list]

    ### plotting

    def _plot_2d_data_scatter(self,ax=None,state_colors=None,plot_slice=slice(None),update=False):
        # TODO doesn't play very nicely with plot_slice
        ax = ax if ax else plt.gca()

        artists = []
        for s, data in zip(self.states_list,self.datas):
            data = data[plot_slice]

            # scatter, like _HMMBase._plot_2d_data_scatter but pads the stateseq
            state_colors = state_colors if state_colors else self._get_colors()
            stateseq = np.concatenate((np.repeat(s.stateseq[0],self.nlags),s.stateseq))[plot_slice]
            colorseq = np.array([state_colors[state] for state in stateseq])
            if update and hasattr(s,'_data_scatter'):
                s._data_scatter.set_offsets(data[:,:2])
                s._data_scatter.set_color(colorseq)
            else:
                s._data_scatter = ax.scatter(data[:,0],data[:,1],c=colorseq,s=5)
            artists.append(s._data_scatter)

            # connect scatter points with line segments
            state_colors = self._get_colors(scalars=True)
            stateseq = np.concatenate((np.repeat(s.stateseq[0],self.nlags),s.stateseq[:-1]))[plot_slice]
            colorseq = np.array([state_colors[state] for state in stateseq])
            if update and hasattr(s,'_data_linesegments'):
                s._data_linesegments.set_array(colorseq)
            else:
                lc = s._data_linesegments = LineCollection(AR_striding(data,1).reshape(-1,2,2))
                lc.set_array(colorseq)
                lc.set_linewidth(0.5)
                ax.add_collection(lc)
            artists.append(s._data_linesegments)

        return artists

    def _plot_stateseq_data_values(self,s,ax,state_colors,plot_slice,update,data=None):
        data = undo_AR_striding(s.data,self.nlags)[plot_slice]
        stateseq = np.concatenate((np.repeat(s.stateseq[0],self.nlags),s.stateseq[:-1]))[plot_slice]
        colorseq = np.tile(np.array([state_colors[state] for state in stateseq]),data.shape[1])

        if update and hasattr(s,'_data_lc'):
            s._data_lc.set_array(colorseq)
        else:
            ts = np.arange(data.shape[0])
            segments = np.vstack(
                [AR_striding(np.hstack((ts[:,None], scalarseq[:,None])),1).reshape(-1,2,2)
                    for scalarseq in data.T])
            lc = s._data_lc = LineCollection(segments)
            lc.set_array(colorseq)
            lc.set_linewidth(0.5)
            ax.add_collection(lc)

        return s._data_lc


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
    def resample_dur_distns(self):
        for state, distn in enumerate(self.dur_distns):
            distn.resample_with_censoring_and_truncation(
            data=
            [s.durations_censored[s.untrunc_slice][s.stateseq_norep[s.untrunc_slice] == state]
                - s.delays[state] for s in self.states_list],
            censored_data=
            [s.durations_censored[s.trunc_slice][s.stateseq_norep[s.trunc_slice] == state]
                - s.delays[state] for s in self.states_list])
        self._clear_caches()


class ARWeakLimitHDPHSMMDelayedIntNegBinSeparateTrans(
        _ARMixin,
        pyhsmm.models.WeakLimitHDPHSMMDelayedIntNegBinSeparateTrans):
    def resample_dur_distns(self):
        for state, distn in enumerate(self.dur_distns):
            distn.resample_with_censoring_and_truncation(
            data=
            [s.durations_censored[s.untrunc_slice][s.stateseq_norep[s.untrunc_slice] == state]
                - s.delays[state] for s in self.states_list],
            censored_data=
            [s.durations_censored[s.trunc_slice][s.stateseq_norep[s.trunc_slice] == state]
                - s.delays[state] for s in self.states_list])
        self._clear_caches()


### low-level code

class _HMMFastResamplingMixin(_ARMixin):
    _obs_stats = None
    _transcounts = []

    def __init__(self,dtype='float32',**kwargs):
        self.dtype = dtype
        super(_HMMFastResamplingMixin,self).__init__(**kwargs)

    def add_data(self,data,**kwargs):
        super(_HMMFastResamplingMixin,self).add_data(data.astype(self.dtype),**kwargs)

    def resample_states_slow(self,**kwargs):
        super(_HMMFastResamplingMixin,self).resample_states(**kwargs)

    def resample_states(self,**kwargs):
        from messages import resample_arhmm
        if len(self.states_list) > 0:
            stateseqs = [np.empty(s.T,dtype='int32') for s in self.states_list]
            params, normalizers = map(np.array,zip(*[self._param_matrix(o) for o in self.obs_distns]))
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
            super(_HMMFastResamplingMixin,self).resample_obs_distns()

    # TODO transcounts being lumped together in low-level code, needs separate
    # trans treatment. and negbin / hsmm treatment for that class.
    # def resample_trans_distn(self):
    #     self.trans_distn.resample(trans_counts=self._transcounts)

    @property
    def alphans(self):
        if not hasattr(self,'_alphans'):
            self._alphans = [np.empty((s.T,self.num_states),
                dtype=self.dtype) for s in self.states_list]
        return self._alphans

    @staticmethod
    def _param_matrix(o):
        D, A, sigma = o.D, o.A, o.sigma
        sigma_inv = np.linalg.inv(sigma)
        parammat =  -1./2 * blockarray([
            [A.T.dot(sigma_inv).dot(A), -A.T.dot(sigma_inv)],
            [-sigma_inv.dot(A), sigma_inv]
            ])
        normalizer = D/2*np.log(2*np.pi) + np.log(np.diag(np.linalg.cholesky(sigma))).sum()
        return parammat, normalizer


class _INBHSMMFastResamplingMixin(_ARMixin):
    _obs_stats = None

    def __init__(self,dtype='float32',**kwargs):
        self.dtype = dtype
        super(_INBHSMMFastResamplingMixin,self).__init__(**kwargs)

    def add_data(self,data,**kwargs):
        super(_INBHSMMFastResamplingMixin,self).add_data(data.astype(self.dtype),**kwargs)

    def resample_states_slow(self,**kwargs):
        super(_INBHSMMFastResamplingMixin,self).resample_states(**kwargs)

    def resample_states(self,**kwargs):
        # NOTE: kwargs is just to absorb any multiprocessing stuff
        # TODO only use this when the number/size of sequences warrant it
        from messages import resample_arhmm
        assert self.obs_distns[0].D_out > 1
        if len(self.states_list) > 0:
            stateseqs = [np.empty(s.T,dtype='int32') for s in self.states_list]
            params, normalizers = map(np.array,zip(*[self._param_matrix(o) for o in self.obs_distns]))
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

    def resample_obs_distns(self):
        if self._obs_stats is not None:
            for state, (o, statmat) in enumerate(zip(self.obs_distns,self._obs_stats)):
                # # NOTE: next two lines are for testing correctness
                # statmat2 = o._get_statistics([s.data[s.stateseq == state] for s in self.states_list])
                # assert all(np.allclose(a,b) for a,b in zip(statmat,statmat2))
                o.resample(stats=statmat)
        else:
            super(_INBHSMMFastResamplingMixin,self).resample_obs_distns()

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


########################
#  feature regression  #
########################

class _FeatureRegressionMixin(object):
    def __init__(self,windowsize=None,featurefn=None,**kwargs):
        self.windowsize, self.featurefn = windowsize, featurefn
        super(_FeatureRegressionMixin,self).__init__(**kwargs)

    def add_data(self,data,featureseq=None,features_and_data=None,**kwargs):
        if features_and_data is not None:
            super(_FeatureRegressionMixin,self).add_data(data=features_and_data,**kwargs)
        else:
            if featureseq is None:
                data, featureseq = self._get_featureseq(data)
            super(_FeatureRegressionMixin,self).add_data(data=np.hstack((featureseq,data)),**kwargs)

    def generate(self,T=None,featureseq=None,keep=True):
        assert (T is None) ^ (featureseq is None)

        if featureseq is not None:
            s = self._states_class(model=self,T=featureseq.shape[0],initialize_from_prior=True)
            data = self._generate_obs(s)
        else:
            raise NotImplementedError

        if keep:
            self.add_data(data=data,featureseq=featureseq)

        return data, s.stateseq

    def _generate_obs(self,states_obj):
        data = np.empty((featureseq.shape[0], self.D))
        for t, state in enumerate(s.stateseq):
            data[t] = self.obs_distns[state].rvs(np.atleast_2d(featureseq[t]))
        return data

    def _get_featureseq(self,data):
        assert None not in (self.featurefn, self.windowsize)
        assert data.ndim == 2 and data.shape[0] > self.windowsize
        featuredim = self.featurefn(data[:self.windowsize]).shape[0]
        out = np.empty((data.shape[0]-self.windowsize,featuredim),dtype=data.dtype)
        for t in xrange(out.shape[0]):
            out[t] = self.featurefn(data[t:t+self.windowsize])
        return data[self.windowsize:], out

    def _get_joblib_pair(self,s):
        return (None, dict(s._kwargs, features_and_data=s.data))

    ### prediction

    def predict(self,seed_data,timesteps,with_noise=False):
        assert None not in (self.featurefn,self.windowsize)
        raise NotImplementedError


class FeatureARHMM(_FeatureRegressionMixin,pyhsmm.models.HMM):
    pass


class FeatureARWeakLimitStickyHDPHMM(
        _FeatureRegressionMixin,
        pyhsmm.models.WeakLimitStickyHDPHMM):
    pass


class FeatureARWeakLimitStickyHDPHMMSeparateTrans(
    _FeatureRegressionMixin,
    pyhsmm.models.WeakLimitStickyHDPHMMSeparateTrans):
    pass

### low-level code


class _FastFeatureRegressionMixin(_FeatureRegressionMixin):
    _obs_stats = None
    _transcounts = []

    def __init__(self,dtype='float64',**kwargs):
        self.dtype = dtype
        super(_FastFeatureRegressionMixin,self).__init__(**kwargs)

    def add_data(self,data,**kwargs):
        super(_FastFeatureRegressionMixin,self).add_data(data=data.astype(self.dtype),**kwargs)

    def resample_states(self,**kwargs):
        # NOTE: kwargs is just to absorb any multiprocessing stuff
        from messages import resample_featureregressionhmm
        assert self.obs_distns[0].D_out > 1
        if len(self.states_list) > 0:
            stateseqs = [np.empty(s.T,dtype='int32') for s in self.states_list]
            params, normalizers = map(np.array,zip(*[self._param_matrix(o) for o in self.obs_distns]))
            stats, transcounts, loglikes = resample_featureregressionhmm(
                    self.obs_distns[0].D_out,
                    [s.pi_0.astype(self.dtype) for s in self.states_list],
                    [s.trans_matrix.astype(self.dtype) for s in self.states_list],
                    params.astype(self.dtype), normalizers.astype(self.dtype),
                    [s.data for s in self.states_list],
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
            self._alphans = [np.empty((s.T,self.num_states),
                dtype=self.dtype) for s in self.states_list]
        return self._alphans


class FastFeatureARHMM(_FastFeatureRegressionMixin,pyhsmm.models.HMM):
    pass


class FastFeatureARWeakLimitStickyHDPHMM(
        _FastFeatureRegressionMixin,
        pyhsmm.models.WeakLimitStickyHDPHMM):
    pass


#########################
#  changepoints models  #
#########################

class _ARChangepointsMixin(object):
    def add_data(self,data,changepoints,strided=False,**kwargs):
        nlags = self.nlags
        if sum(b-a for a,b in changepoints) != data.shape[0] - nlags:
            changepoints = [(max(a-nlags,0),b-nlags) for a,b in changepoints if b > nlags]
        super(_ARChangepointsMixin,self).add_data(
                data=data,changepoints=changepoints,**kwargs)


class ARWeakLimitHDPHSMMPossibleChangepoints(
        _ARChangepointsMixin,
        _ARMixin,
        pyhsmm.models.WeakLimitHDPHSMMPossibleChangepoints):
    pass


class ARWeakLimitHDPHSMMPossibleChangepointsSeparateTrans(
        _ARChangepointsMixin,
        _ARMixin,
        pyhsmm.models.WeakLimitHDPHSMMPossibleChangepointsSeparateTrans):
    pass
