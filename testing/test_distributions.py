from __future__ import division
import numpy as np
import abc
from matplotlib import pyplot as plt

from nose.plugins.attrib import attr
from pyhsmm.basic.pybasicbayes.testing.mixins import BigDataGibbsTester

from .. import distributions as d
from ..util import AR_striding

class ARBigDataGibbsTester(BigDataGibbsTester):
    def check_big_data(self,setting_idx,hypparam_dict):
        d1 = self.distribution_class(**hypparam_dict)
        d2 = self.distribution_class(**hypparam_dict)

        data = d1.rvs(prefix=self.prefixes[setting_idx],length=self.big_data_size)
        d2.resample(AR_striding(data,self.nlagss[setting_idx]))

        assert self.params_close(d1,d2)

    @abc.abstractproperty
    def prefixes(self):
        pass

    @abc.abstractproperty
    def nlagss(self):
        pass


@attr('AR_MNIW')
class Test_AR_MNIW(ARBigDataGibbsTester):
    @property
    def distribution_class(self):
        return d.AR_MNIW

    @property
    def hyperparameter_settings(self):
        return (
            dict(nu_0=25,S_0=25*np.eye(2),M_0=np.zeros((2,4)),Kinv_0=np.eye(4),
                A=np.hstack((-0.2*np.eye(2),1.2*np.eye(2))),sigma=np.eye(2)),
            dict(nu_0=25,S_0=2*25*np.eye(2),M_0=np.zeros((2,4)),Kinv_0=1./3*np.eye(4),
                A=np.hstack((-0.2*np.eye(2),1.2*np.eye(2))),sigma=np.eye(2)),
            )

    @property
    def prefixes(self):
        return (np.zeros((2,2)),np.zeros((2,2)))

    @property
    def nlagss(self):
        return (2,2)

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.fullA-d2.fullA) < 0.1 \
                and np.linalg.norm(d1.sigma-d2.sigma) < 0.1

    @property
    def big_data_size(self):
        return 10000

    @property
    def big_data_repeats_per_setting(self):
        return 3

@attr('AR_MNFixedSigma')
class Test_AR_MNFixedSigma(ARBigDataGibbsTester):
    @property
    def distribution_class(self):
        return d.AR_MNFixedSigma

    @property
    def hyperparameter_settings(self):
        return (
            dict(sigma=np.diag([1.,2.]),M_0=np.zeros((2,4)),Uinv_0=1e-2*np.eye(2),
                Vinv_0=1e-2*np.eye(4),A=np.hstack((-0.2*np.eye(2),1.2*np.eye(2)))),
            )

    @property
    def prefixes(self):
        return (np.zeros((2,2)),)

    @property
    def nlagss(self):
        return (2,)

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.fullA-d2.fullA) < 0.1

    @property
    def big_data_size(self):
        return 10000

    @property
    def big_data_repeats_per_setting(self):
        return 3

