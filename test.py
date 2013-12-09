from __future__ import division
import numpy as np
import abc

from nose.plugins.attrib import attr

from pyhsmm.basic.pybasicbayes.testing.mixins \
        import BigDataGibbsTester, GewekeGibbsTester, mkdir
from pyhsmm.util import testing
import distributions, util

# TODO geweke tests aren't playing nicely. instability or broken?

class ARBigDataGibbsTester(BigDataGibbsTester):
    @abc.abstractproperty
    def prefixes(self):
        pass

    def check_big_data(self,setting_idx,hypparam_dict):
        d1 = self.distribution_class(**hypparam_dict)
        d2 = self.distribution_class(**hypparam_dict)

        # NOTE: these lines are different
        data = d1.rvs(self.prefixes[setting_idx],self.big_data_size)
        strided_data = util.AR_striding(data,d1.nlags)
        d2.resample(strided_data)

        assert self.params_close(d1,d2)

class ARGewekeGibbsTester(GewekeGibbsTester):
    def check_geweke(self,setting_idx,hypparam_dict):
            import os
            from matplotlib import pyplot as plt
            plt.ioff()
            fig = plt.figure()
            figpath = self.geweke_figure_filepath(setting_idx)
            mkdir(os.path.dirname(figpath))

            nsamples, data_size, ntrials = self.geweke_nsamples, \
                    self.geweke_data_size, self.geweke_ntrials

            d = self.distribution_class(**hypparam_dict)
            sample_dim = np.atleast_1d(self.geweke_statistics(d,d.rvs(self.prefixes[setting_idx],10))).shape[0]

            num_statistic_fails = 0
            for trial in xrange(ntrials):
                # collect forward-generated statistics
                forward_statistics = np.squeeze(np.empty((nsamples,sample_dim)))
                for i in xrange(nsamples):
                    d = self.distribution_class(**hypparam_dict)
                    data = d.rvs(self.prefixes[setting_idx],data_size)
                    forward_statistics[i] = self.geweke_statistics(d,data)

                # collect gibbs-generated statistics
                gibbs_statistics = np.squeeze(np.empty((nsamples,sample_dim)))
                d = self.distribution_class(**hypparam_dict)
                data = d.rvs(self.prefixes[setting_idx],data_size)
                for i in xrange(nsamples):
                    d.resample(util.AR_striding(data,d.nlags),**self.resample_kwargs)
                    data = d.rvs(self.prefixes[setting_idx],data_size)
                    gibbs_statistics[i] = self.geweke_statistics(d,data)

                testing.populations_eq_quantile_plot(forward_statistics,gibbs_statistics,fig=fig)
                try:
                    sl = self.geweke_numerical_slice(d,setting_idx)
                    testing.assert_populations_eq_moments(
                            forward_statistics[...,sl],gibbs_statistics[...,sl],
                            pval=self.geweke_pval)
                except AssertionError:
                    datapath = os.path.join(os.path.dirname(__file__),'figures',
                            self.__class__.__name__,'setting_%d_trial_%d.pdf' % (setting_idx,trial))
                    np.savez(datapath,fwd=forward_statistics,gibbs=gibbs_statistics)
                    example_violating_means = forward_statistics.mean(0), gibbs_statistics.mean(0)
                    num_statistic_fails += 1

            plt.savefig(figpath)

            assert num_statistic_fails <= self.geweke_num_statistic_fails_to_tolerate, \
                    'Geweke MAY have failed, check FIGURES in %s (e.g. %s vs %s)' \
                    % ((os.path.dirname(figpath),) + example_violating_means)

@attr('mniw')
class TestMNIW(ARBigDataGibbsTester):
    @property
    def distribution_class(self):
        return distributions.MNIW

    @property
    def prefixes(self):
        return (
                np.array([[0.],[0.1],[0.2]]),
                np.zeros((2,2)),
               )

    @property
    def big_data_size(self):
        return 5000

    @property
    def hyperparameter_settings(self):
        return (
                dict(
                    dof=1.5,S=3*np.eye(1),M=np.zeros((1,3)),K=10*np.eye(3),
                    A=np.array([[0.3,0.2,0.1]]),Sigma=0.7*np.eye(1),
                    ),
                dict(
                    dof=3,S=np.eye(2),M=np.zeros((2,4)),K=np.eye(4),
                    A=np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]]).dot(np.hstack((-np.eye(2),np.eye(2)))) + np.hstack((np.zeros((2,2)),np.eye(2))),
                    Sigma=np.diag((0.7,0.3)),
                    ),
               )

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.A - d2.A) < 0.1 and np.linalg.norm(d1.Sigma - d2.Sigma) < 0.1


    @property
    def geweke_hyperparameter_settings(self):
        return (
                dict(
                    dof=1.5,S=3*np.eye(1),M=np.zeros((1,3)),K=10*np.eye(3),
                    A=np.array([[0.3,0.2,0.1]]),Sigma=0.7*np.eye(1),
                    ),
               ) # TODO ensure stability


    def geweke_statistics(self,d,data):
        return d.A.ravel()

    @property
    def geweke_data_size(self):
        return 5

    @property
    def geweke_nsamples(self):
        return 2500

