from __future__ import division
import numpy as np
import scipy.linalg

import pyhsmm

from util.stats import sample_mniw, getdatasize

class MNIW(pyhsmm.basic.abstractions.GibbsSampling):
    '''
    conjugate Matrix-Normal Inverse Wishart prior for (vector) autoregressive
    processes
    '''

    def __init__(self,dof,S,M,K):
        self.dof = dof
        self.S = S
        self.M = M
        self.K = K

        self.nlags = M.shape[1]/M.shape[0]
        self._dotter = np.ones(M.shape[0])

        self.resample()

    def log_likelihood(self,x):
        D = self.A.shape[0]
        x = np.reshape(x,(-1,D))
        Sigmachol = self.Sigmachol = self.Sigmachol \
                if self.Sigmachol is not None else np.linalg.cholesky(self.Sigma)
        return -1./2. * \
                self._dotter.dot(scipy.linalg.solve_triangular(
                    Sigmachol,
                    self.A.dot(x[:,:-D]) - x[:,-D:],
                    lower=True)**2) \
                - D/2*np.log(2*np.pi * np.diag(Sigmachol).prod())

    def rvs(self,size=[]):
        raise NotImplementedError

    def resample(self,data):
        self.A, self.Sigma = sample_mniw(*self._posterior_hypparams(*self._get_statistics(data)))
        self.Sigmachol = None

    def _get_statistics(self,data):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all(isinstance(d,np.ndarray) for d in data))

        nlags = self.nlags
        D = self.M.shape[0]
        assert data.shape[1]/D == (nlags+1)

        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                Syy = data[:,-D:].T.dot(data[:,-D:])
                Sytyt = data[:,:-D].T.dot(data[:,:-D])
                Syyt = data[:,-D:].T.dot(data[:,:-D])
            else:
                Syy = sum(d[:,-D:].T.dot(d[:,-D:]) for d in data)
                Sytyt = sum(d[:,:-D].T.dot(data[:,:-D]) for d in data)
                Syyt = sum(d[:,-D:].T.dot(d[:,:-D]) for d in data)
        else:
            Syy = Sytyt = Syyt = None

        return Syy,Sytyt,Syyt,n

    def _posterior_hypparams(self,Syy,Sytyt,Syyt,n):
        if n > 0:
            K_n = Sytyt + self.K
            M_n = np.linalg.solve(K_n,(Syyt + self.M.dot(self.K)).T).T
            S_n = (Syy - M_n.dot((Syyt + self.M.dot(self.K)).T)) + self.S_0
            dof_n = n + self.dof
        else:
            K_n = self.K
            M_n = self.M
            S_n = self.S_0
            dof_n = self.dof

        return dof_n, S_n, M_n, np.linalg.inv(K_n)

