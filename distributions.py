from __future__ import division
import numpy as np
from numpy import newaxis as na
import scipy.linalg

import pyhsmm
from pyhsmm.util.stats import sample_mniw, getdatasize

from util import AR_striding

class MNIW(pyhsmm.basic.abstractions.GibbsSampling):
    '''
    conjugate Matrix-Normal Inverse Wishart prior for (vector) autoregressive
    processes

    e.g. MNIW(3,3*np.eye(2),np.zeros((2,4)),10*np.eye(4))
    '''

    def __init__(self,dof,S,M,K,affine=False):
        assert S.shape[0] == S.shape[1] == M.shape[0] and M.shape[1] == K.shape[0] == K.shape[1]
        assert ((not affine) and M.shape[1] % M.shape[0] == 0) or (affine and (M.shape[1]-1) % M.shape[0] == 0)

        self.dof = dof
        self.S = S
        self.M = M
        self.K = K

        self.affine = affine

        self.nlags = M.shape[1]//M.shape[0] if not affine else (M.shape[1]-1)//M.shape[0]
        self._dotter = np.ones(M.shape[0])

        self.resample()

    def __repr__(self):
        return 'MNIW(\nA=\n%s,\nSigma=\n%s\n)' % (self.A,self.Sigma)

    def _get_sigma_chol(self):
        if self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(self.Sigma)
        return self._sigma_chol

    def log_likelihood(self,x):
        D = self.A.shape[0]
        chol = self._get_sigma_chol()
        return -1./2. * self._dotter.dot(scipy.linalg.solve_triangular(
                    chol,
                    (x[:,:-D].dot(self.A[:,(1 if self.affine else 0):].T) - x[:,-D:]).T \
                            + (self.A[:,0,na] if self.affine else 0),
                    lower=True)**2) \
                - D/2*np.log(2*np.pi) - np.log(chol.diagonal()).sum()

    def rvs(self,prefix,length):
        assert prefix.ndim == 2 and prefix.shape[0] == self.nlags and prefix.shape[1] == self.M.shape[0]

        out = np.zeros((length+self.nlags,self.M.shape[0]))
        out[:self.nlags] = prefix
        strided_out = AR_striding(out,self.nlags-1)

        chol = self._get_sigma_chol()

        randomness = np.random.normal(size=(length,self.M.shape[0])).dot(chol.T)

        for itr in range(length):
            out[itr+self.nlags] = self.A[:,(1 if self.affine else 0):].dot(strided_out[itr]) \
                    + randomness[itr] + (self.A[:,0] if self.affine else 0)

        return out[self.nlags:]

    def resample(self,data=[]):
        self.A, self.Sigma = sample_mniw(*self._posterior_hypparams(*self._get_statistics(data)))
        self._sigmachol = None

    def _get_statistics(self,data):
        # NOTE data must be passed in strided!!!
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all(isinstance(d,np.ndarray) for d in data))

        D = self.M.shape[0]

        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                Syy = data[:,-D:].T.dot(data[:,-D:])
                Sytyt = data[:,:-D].T.dot(data[:,:-D])
                Syyt = data[:,-D:].T.dot(data[:,:-D])

                if self.affine:
                    Syyt = np.hstack((data[:,-D:].sum(0)[:,na],Syyt))
                    Sytytsum = data[:,:-D].sum(0)
                    Sytyt = np.vstack((
                            np.hstack(((n,),Sytytsum)),
                            np.hstack((Sytytsum[:,na],Sytyt))
                        ))
            else:
                Syy = sum(d[:,-D:].T.dot(d[:,-D:]) for d in data)
                Sytyt = sum(d[:,:-D].T.dot(d[:,:-D]) for d in data)
                Syyt = sum(d[:,-D:].T.dot(d[:,:-D]) for d in data)

                if self.affine:
                    Syyt = np.hstack((sum(d[:,-D:].sum(0) for d in data)[:,na],Syyt))
                    Sytytsum = sum(d[:,:-D].sum(0) for d in data)
                    Sytyt = np.vstack((
                            np.hstack(((n,),Sytytsum)),
                            np.hstack((Sytytsum[:,na],Sytyt))
                        ))
        else:
            Syy = Sytyt = Syyt = None

        return Syy,Sytyt,Syyt,n

    def _posterior_hypparams(self,Syy,Sytyt,Syyt,n):
        if n > 0:
            K_n = Sytyt + self.K
            M_n = np.linalg.solve(K_n,(Syyt + self.M.dot(self.K)).T).T
            S_n = (Syy - M_n.dot((Syyt + self.M.dot(self.K)).T)) + self.S
            dof_n = n + self.dof
        else:
            K_n = self.K
            M_n = self.M
            S_n = self.S
            dof_n = self.dof

        return dof_n, S_n, M_n, np.linalg.inv(K_n)

