from __future__ import division
import numpy as np
import abc

from nose.plugins.attrib import attr

from pyhsmm.basic.pybasicbayes.testing.mixins \
        import BigDataGibbsTester, GewekeGibbsTester
import distributions, util

class ARBigDataGibbsTester(BigDataGibbsTester):
    @abc.abstractproperty
    def big_data_prefixes(self):
        pass

    def check_big_data(self,setting_idx,hypparam_dict):
        d1 = self.distribution_class(**hypparam_dict)
        d2 = self.distribution_class(**hypparam_dict)

        # NOTE: these lines are different
        data = d1.rvs(self.big_data_prefixes[setting_idx],self.big_data_size)
        strided_data = util.AR_striding(data,d1.nlags)
        d2.resample(strided_data)

        assert self.params_close(d1,d2)

@attr('mniw')
class TestMNIW(ARBigDataGibbsTester):
    @property
    def distribution_class(self):
        return distributions.MNIW

    @property
    def big_data_prefixes(self):
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

