from __future__ import division
import numpy as np
from numpy import newaxis as na
import scipy.linalg
import copy

from pyhsmm.basic.abstractions import GibbsSampling, MaxLikelihood
from pyhsmm.util.stats import sample_mniw_kinv, sample_invwishart, sample_mn, \
        getdatasize

from util import AR_striding, undo_AR_striding

# TODO support 'lazy' instantiation

class _ARBase(MaxLikelihood):
    def __init__(self,D,nlags):
        self.D = D
        self.nlags = nlags

    @property
    def params(self):
        if self.affine:
            return dict(A=self.A,b=self.b,sigma=self.sigma)
        else:
            return dict(A=self.A,sigma=self.sigma)

    @property
    def A(self):
        if not self.affine:
            return self.fullA
        else:
            return self.fullA[:,1:]

    @A.setter
    def A(self,A):
        if not self.affine:
            self.fullA = A
        else:
            self.fullA[:,1:] = A

    @property
    def b(self):
        if self.affine:
            return self.fullA[:,0]
        else:
            return np.zeros(self.fullA.shape[0])

    @b.setter
    def b(self,b):
        assert self.affine
        self.fullA[:,0] = b

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self,sigma):
        self._sigma = sigma
        self._sigma_chol = None

    @property
    def sigma_chol(self):
        if not hasattr(self,'_sigma_chol') or self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(self.sigma)
        return self._sigma_chol

    def log_likelihood(self,x):
        if hasattr(self,'broken'):
            return np.repeat(-np.inf,x.shape[0]) if isinstance(x,np.ndarray) else -np.inf
        try:
            chol = self.sigma_chol
            D = self.D
            return -1./2. * scipy.linalg.solve_triangular(
                        chol,
                        (x[:,:-D].dot(self.A.T) - x[:,-D:]).T + self.b[:,na],
                        lower=True).sum(0)**2 \
                    - D/2*np.log(2*np.pi) - np.log(chol.diagonal()).sum()
        except np.linalg.LinAlgError:
            self.broken = True
            return np.repeat(-np.inf,x.shape[0]) if isinstance(x,np.ndarray) else -np.inf

    def rvs(self,prefix,length):
        D = self.D
        assert prefix.ndim == 2 and prefix.shape[0] == self.nlags and prefix.shape[1] == D

        out = np.zeros((length+self.nlags,D))
        out[:self.nlags] = prefix
        strided_out = AR_striding(out,self.nlags-1)

        randomness = np.random.normal(size=(length,D)).dot(self.sigma_chol.T)

        for itr in range(length):
            out[itr+self.nlags] = self.A.dot(strided_out[itr]) + randomness[itr] + self.b

        return out[self.nlags:]

    def _get_statistics(self,data):
        D = self.D
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                Sfull = data.T.dot(data)
                Syy = Sfull[-D:,-D:]
                Sytyt = Sfull[:-D,:-D]
                Syyt = Sfull[-D:,:-D]

                if self.affine:
                    Syyt = np.hstack((data[:,-D:].sum(0)[:,na],Syyt))
                    Sytytsum = data[:,:-D].sum(0)
                    Sytyt = np.vstack((
                                np.hstack(((n,),Sytytsum)),
                                np.hstack((Sytytsum[:,na],Sytyt))
                            ))
            else:
                return sum(self._get_statistics(d) for d in data)
        else:
            nlags = self.nlags
            Syy = np.zeros((D,D))
            Sytyt = np.zeros((nlags*D,nlags*D))
            Syyt = np.zeros((D,nlags*D))
            n = 0
        return np.array([Syy,Syyt,Sytyt,n])


    def _get_weighted_statistics(self,data,weights=None):
        if weights is None:
            return self._get_statistics(data)
        else:
            D = self.D
            if isinstance(data,np.ndarray):
                neff = weights.sum()
                Sfull = np.einsum('ni,nj,n->ij',data,data,weights)
                Syy = Sfull[-D:,-D:]
                Sytyt = Sfull[:-D,:-D]
                Syyt = Sfull[-D:,:-D]

                if self.affine:
                    Syyt = np.hstack((weights.dot(data[:,-D:]),Syyt))
                    Sytytsum = weights.dot(data[:,:-D])
                    Sytyt = np.vstack((
                                np.hstack(((neff,),Sytytsum)),
                                np.hstack((Sytytsum[:,na],Sytyt))
                            ))
                return np.array([Syy,Syyt,Sytyt,neff])
            else:
                return sum(self._get_weighted_statistics(d,w) for d,w in zip(data,weights))

    def max_likelihood(self,data,weights=None):
        D = self.D
        Syy, Syyt, Sytyt, n = self._get_weighted_statistics(data,weights)

        if n > 0:
            try:
                self.fullA = np.linalg.solve(Sytyt, Syyt.T).T
                self.sigma = (Syy - self.fullA.dot(Syyt.T))/n
            except np.linalg.LinAlgError:
                # broken!
                self.broken = True
        else:
            # no data, M step not defined
            self.broken = True

class MNIW(_ARBase,GibbsSampling):
    def __init__(self,nu_0,S_0,M_0,Kinv_0,affine=False,
            A=None,b=None,sigma=None):
        self.affine = affine
        if A is not None:
            self.fullA = np.hstack((b[:,na],A)) if affine else A
        else:
            self.fullA = None
        self.sigma = sigma

        self.natural_hypparam = self._standard_to_natural(nu_0, S_0, M_0, Kinv_0)
        self.D = M_0.shape[0]
        self.nlags = M_0.shape[1] // M_0.shape[0] if not self.affine \
                else (M_0.shape[1]-1) // M_0.shape[0]

        if (affine and (A,sigma,b) == (None,None,None)) or (A,sigma) == (None,None):
            self.resample() # initialize from prior

    @property
    def hypparams(self):
        nu_0, S_0, M_0, Kinv_0 = self._natural_to_standard(self.natural_hypparam)
        return dict(nu_0=nu_0,S_0=S_0,M_0=M_0,Kinv_0=Kinv_0)

    ### converting between natural and standard hyperparameters

    def _standard_to_natural(self,nu,S,M,Kinv):
        A = S + M.dot(Kinv).dot(M.T)
        B = Kinv
        C = M.dot(Kinv)
        d = nu
        return np.array([A,C,B,d])

    def _natural_to_standard(self,natparam):
        A,C,B,d = natparam
        nu = d
        Kinv = B
        M = np.linalg.solve(B,C.T).T
        S = A - M.dot(B).dot(M.T)
        return nu, S, M, Kinv

    ### Gibbs sampling

    def resample(self,data=[]):
        self.fullA, self.sigma = sample_mniw_kinv(
            *self._natural_to_standard(self.natural_hypparam + self._get_statistics(data)))

    def copy_sample(self):
        new = copy.copy(self)
        new.fullA = self.fullA.copy()
        new.sigma = self.sigma.copy()
        return new

class MNFixedSigma(_ARBase,GibbsSampling):
    def __init__(self,sigma,M_0,Uinv_0,Vinv_0,affine=False,
            A=None,b=None):
        self.affine = affine
        self.fullA = np.hstack((b[:,na],A)) if affine else A
        self.sigma = sigma

        self.natural_hypparam = self._standard_to_natural(M_0,Uinv_0,Vinv_0)
        self.D = M_0.shape[0]
        self.nlags = M_0.shape[1] // M_0.shape[0] if not self.affine \
                else (M_0.shape[1]-1) // M_0.shape[0]

        self.resample()

    @property
    def hypparams(self):
        M_0, Uinv_0, Vinv_0 = self._natural_to_standard(self.natural_hypparam)
        return dict(M_0=M_0,Uinv_0=Uinv_0,Vinv_0=Vinv_0)

    ### converting between natural and standard hyperparameters

    def _standard_to_natural(self,M,Uinv,Vinv):
        return np.array([Uinv.dot(M).dot(Vinv),-0.5*Uinv,Vinv])

    def _natural_to_standard(self,natparam):
        Uinv, Vinv, product = natparam
        Uinv = Uinv / -0.5
        M = np.linalg.solve(Uinv,np.linalg.solve(Vinv,product.T).T)
        return dict(M=M,Uinv=Uinv,Vinv=Vinv)

    ### statistics

    def _get_statistics(self,data):
        return self._shape_statistics(
            super(MNFixedSigma,self)._get_statistics(data))

    def _get_weighted_statistics(self,data,weights=None):
        return self._shape_statistics(
            super(MNFixedSigma,self)._get_weighted_statistics(data,weights))

    def _shape_statistics(self,stats):
        Syy, Syyt, Sytyt, n = stats
        sigmainv = self.linalg.inv(sigma)
        return np.array([sigmainv.dot(Syyt), -0.5*sigmainv, Sytyt])

    ### Gibbs sampling

    def resample(self,data=[]):
        self.fullA = sample_mn(
            **self._natural_to_standard(self.natural_hypparam + self._get_statistics(data)))

    def copy_sample(self):
        new = copy.copy(self)
        new.fullA = self.fullA.copy()
        return new

