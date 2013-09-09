from __future__ import division
import numpy as np
from numpy import newaxis as na
import scipy.linalg

from pyhsmm.basic.abstractions import GibbsSampling, MaxLikelihood, MAP
from pyhsmm.util.stats import sample_mniw, getdatasize

from util import AR_striding

class MNIW(GibbsSampling, MaxLikelihood):
    '''
    conjugate Matrix-Normal Inverse Wishart prior for (vector) autoregressive
    processes

    e.g. MNIW(3,3*np.eye(2),np.zeros((2,4)),10*np.eye(4))
    '''

    def __init__(self,dof,S,M,K,affine=False):
        assert S.shape[0] == S.shape[1] == M.shape[0] and M.shape[1] == K.shape[0] == K.shape[1]
        assert (M.shape[1] + (-1 if affine else 0)) % M.shape[0] == 0

        self.dof = dof
        self.S = S
        self.M = M
        self.K = K

        self.affine = affine

        self.nlags = (M.shape[1] + (-1 if affine else 0))//M.shape[0]
        self._dotter = np.ones(M.shape[0])

        self.resample()

    @property
    def params(self):
        if self.affine:
            return dict(A=self.A,b=self.b,Sigma=self.Sigma)
        else:
            return dict(A=self.A,Sigma=self.Sigma)

    @property
    def hypparams(self):
        return dict(dof=self.dof,S=self.S,M=self.M,K=self.K)

    def _get_sigma_chol(self):
        if not hasattr(self,'_sigma_chol') or self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(self.Sigma)
        return self._sigma_chol

    def log_likelihood(self,x):
        D = self.A.shape[0]
        chol = self._get_sigma_chol()
        return -1./2. * self._dotter.dot(scipy.linalg.solve_triangular(
                    chol,
                    (x[:,:-D].dot(self.A.T) - x[:,-D:]).T \
                            + (self.b[:,na] if self.affine else 0),
                    lower=True)**2) \
                - D/2*np.log(2*np.pi) - np.log(chol.diagonal()).sum()

    def rvs(self,prefix,length):
        D = self.M.shape[0]
        assert prefix.ndim == 2 and prefix.shape[0] == self.nlags and prefix.shape[1] == D

        out = np.zeros((length+self.nlags,D))
        out[:self.nlags] = prefix
        strided_out = AR_striding(out,self.nlags-1)

        chol = self._get_sigma_chol()

        randomness = np.random.normal(size=(length,self.M.shape[0])).dot(chol.T)

        for itr in range(length):
            out[itr+self.nlags] = self.A.dot(strided_out[itr]) \
                    + randomness[itr] + (self.b if self.affine else 0)

        return out[self.nlags:]

    def resample(self,data=[]):
        self.A, self.Sigma = sample_mniw(*self._posterior_hypparams(*self._get_statistics(data)))
        if self.affine:
            self.b = self.A[:,0]
            self.A = self.A[:,1:]
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
            M_n = np.linalg.solve(K_n,(Syyt + self.M.dot(self.K)).T).T # TODO call psd solver
            S_n = (Syy - M_n.dot((Syyt + self.M.dot(self.K)).T)) + self.S
            dof_n = n + self.dof
        else:
            K_n = self.K
            M_n = self.M
            S_n = self.S
            dof_n = self.dof

        return dof_n, S_n, M_n, np.linalg.inv(K_n)

    def max_likelihood(self,data,weights=None):
        Syy, Sytyt, Syyt, n = self._get_weighted_statistics(data,weights)

        try:
            self.A = np.linalg.solve(Sytyt, Syyt.T).T # TODO call psd solver
            self.Sigma = (Syy - self.A.dot(Syyt.T))/n
            if self.affine:
                self.b = self.A[:,0]
                self.A = self.A[:,1:]
        except np.linalg.LinAlgError:
            # broken!
            self.A = 999999999 * np.ones_like(self.M)
            self.b = 999999999 * np.ones(self.M.shape[0])
            self.broken = True

        self._sigmachol = None

    def _get_weighted_statistics(self,data,weights=None):
        if weights is None:
            return self._get_statistics(data)
        else:
            D = self.M.shape[0]

            if isinstance(data,np.ndarray):
                neff = weights.sum()
                Syy = data[:,-D:].T.dot(weights[:,na] * data[:,-D:])
                Sytyt = data[:,:-D].T.dot(weights[:,na] * data[:,:-D])
                Syyt = data[:,-D:].T.dot(weights[:,na] * data[:,:-D])

                if self.affine:
                    Syyt = np.hstack((weights.dot(data[:,-D:]),Syyt))
                    Sytytsum = weights.dot(data[:,:-D])
                    Sytyt = np.vstack((
                                np.hstack(((neff,),Sytytsum)),
                                np.hstack((Sytytsum[:,na],Sytyt))
                            ))
            else:
                neff = sum(w.sum() for w in weights)
                Syy = sum(d[:,-D:].T.dot(w[:,na]*d[:,-D:]) for d,w in zip(data,weights))
                Sytyt = sum(d[:,:-D].T.dot(w[:,na]*d[:,:-D]) for d,w in zip(data,weights))
                Syyt = sum(d[:,-D:].T.dot(w[:,na]*d[:,:-D]) for d,w in zip(data,weights))

                if self.affine:
                    Syyt = np.hstack((sum(w.dot(d[:,-D:]) for d,w in zip(data,weights)),Syyt))
                    Sytytsum = sum(w.dot(d[:,:-D]) for d,w in zip(data,weights))
                    Sytyt = np.vstack((
                                np.hstack(((neff,),Sytytsum)),
                                np.hstack((Sytytsum[:,na],Sytyt))
                            ))

        return Syy,Sytyt,Syyt,neff

