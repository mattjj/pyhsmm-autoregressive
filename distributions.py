from __future__ import division
import numpy as np
from numpy import newaxis as na
import scipy.linalg
import copy

from pyhsmm.basic.abstractions import GibbsSampling, MaxLikelihood, Distribution
from pyhsmm.util.stats import sample_mniw, sample_invwishart, sample_mn, \
        getdatasize

from util import AR_striding, undo_AR_striding

# TODO support 'lazy' instantiation

class _ARBase(Distribution):
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

class _ARMaxLikelihood(_ARBase,MaxLikelihood):
    def max_likelihood(self,data,weights=None):
        D = self.D
        Syy, Syyt, Sytyt, n = _ARBase._get_weighted_statistics(self,data,weights)

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

class AR_MNIW(GibbsSampling,_ARMaxLikelihood):
    def __init__(self,nu_0,S_0,M_0,Kinv_0,affine=False,
            A=None,b=None,sigma=None):
        self.affine = affine
        if affine and None not in (A,b):
            self.fullA = np.hstack((b[:,na],A))
        elif not affine and A is not None:
            self.fullA = A
        else:
            self.fullA = None
        self.sigma = sigma

        self.natural_hypparam = self._standard_to_natural(nu_0, S_0, M_0, Kinv_0)

        self.D = M_0.shape[0]
        self.nlags = M_0.shape[1] // M_0.shape[0] if not self.affine \
                else (M_0.shape[1]-1) // M_0.shape[0]

        if None in (self.fullA, self.sigma):
            self.resample() # initialize from prior

    @property
    def hypparams(self):
        return {name+'_0':val for name,val in
                self._natural_to_standard(self.natural_hypparam)}

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
        return dict(nu=nu,S=S,M=M,Kinv=Kinv)

    ### Gibbs sampling

    def resample(self,data=[]):
        self.fullA, self.sigma = sample_mniw(
            **self._natural_to_standard(self.natural_hypparam + self._get_statistics(data)))

    def copy_sample(self):
        new = copy.copy(self)
        new.fullA = self.fullA.copy()
        new.sigma = self.sigma.copy()
        return new

class AR_MNFixedSigma(_ARBase,GibbsSampling):
    def __init__(self,sigma,M_0,Uinv_0,Vinv_0,affine=False,
            A=None,b=None):
        self.affine = affine
        if affine and None not in (A,b):
            self.fullA = np.hstack((b[:,na],A))
        elif not affine and A is not None:
            self.fullA = A
        else:
            self.fullA = None
        self.sigma = sigma

        self.natural_hypparam = self._standard_to_natural(M_0,Uinv_0,Vinv_0)

        self.D = M_0.shape[0]
        self.nlags = M_0.shape[1] // M_0.shape[0] if not self.affine \
                else (M_0.shape[1]-1) // M_0.shape[0]

        if self.fullA is None:
            self.resample() # initialize from prior

    @property
    def hypparams(self):
        M_0, Uinv_0, Vinv_0 = self._natural_to_standard(self.natural_hypparam)
        return dict(M_0=M_0,Uinv_0=Uinv_0,Vinv_0=Vinv_0)

    ### converting between natural and standard hyperparameters

    def _standard_to_natural(self,M,Uinv,Vinv):
        return np.array([Uinv.dot(M).dot(Vinv),-0.5*Uinv,Vinv])

    def _natural_to_standard(self,natparam):
        product, Uinv, Vinv = natparam
        Uinv = Uinv / -0.5
        M = np.linalg.solve(Uinv,np.linalg.solve(Vinv,product.T).T)
        return dict(M=M,Uinv=Uinv,Vinv=Vinv)

    ### statistics

    def _get_statistics(self,data):
        return self._shape_statistics(
            super(AR_MNFixedSigma,self)._get_statistics(data))

    def _get_weighted_statistics(self,data,weights=None):
        return self._shape_statistics(
            super(AR_MNFixedSigma,self)._get_weighted_statistics(data,weights))

    def _shape_statistics(self,stats):
        Syy, Syyt, Sytyt, n = stats
        sigmainv = np.linalg.inv(self.sigma)
        return np.array([sigmainv.dot(Syyt), -0.5*sigmainv, Sytyt])

    ### Gibbs sampling

    def resample(self,data=[]):
        self.fullA = sample_mn(
            **self._natural_to_standard(self.natural_hypparam + self._get_statistics(data)))

    def copy_sample(self):
        new = copy.copy(self)
        new.fullA = self.fullA.copy()
        return new

class AR_IWFixedA(_ARBase,GibbsSampling):
    def __init__(self,A,nu_0,S_0,affine=False,sigma=None):
        self.affine = affine
        self.fullA = A
        self.sigma = sigma

        self.natural_hypparam = self._standard_to_natural(nu_0,S_0)

        self.D = A.shape[0]
        self.nlags = A.shape[1] // A.shape[0] if not self.affine \
                else (A.shape[1]-1) // A.shape[0]

        if self.sigma is None:
            self.resample() # initialize from prior

    @property
    def hypparams(self):
        nu_0, S_0 = self._natural_to_standard(self.natural_hypparam)
        return dict(nu_0=nu_0,S_0=S_0)

    ### converting between natural and standard hyperparameters

    def _standard_to_natural(self,nu,S):
        return np.array([S,nu])

    def _natural_to_standard(self,natparam):
        S, nu = natparam
        return dict(nu=nu,S=S)

    ### statistics

    def _get_statistics(self,data):
        return self._shape_statistics(
            super(AR_IWFixedA,self)._get_statistics(data))

    def _get_weighted_statistics(self,data,weights=None):
        return self._shape_statistics(
            super(AR_IWFixedA,self)._get_weighted-statistics(data,weights))

    def _shape_statistics(self,stats):
        Syy, Syyt, Sytyt, n = stats
        A = self.fullA
        return np.array([Syy - Syyt.dot(A.T) - A.dot(Syyt.T) + A.dot(Sytyt).dot(A.T),n])

    ### Gibbs sampling

    def resample(self,data=[]):
        self.sigma = sample_invwishart(
            **self._natural_to_standard(self.natural_hypparam + self._get_statistics(data)))

    def copy_sample(self):
        new = copy.copy(self)
        new.sigma = self.sigma.copy()
        return new

class AR_MN_IW_Nonconj(AR_IWFixedA,AR_MNFixedSigma,_ARMaxLikelihood):
    def __init__(self,nu_0,S_0,M_0,Uinv_0,Vinv_0,affine=False,
            A=None,b=None,sigma=None,niter=1):
        self.affine = affine
        if affine and None not in (A,b):
            self.fullA = np.hstack((b[:,na],A))
        elif not affine and A is not None:
            self.fullA = A
        else:
            self.fullA = None
        self.sigma = sigma

        self.natural_hypparam = self._standard_to_natural(nu_0,S_0,M_0,Uinv_0,Vinv_0)

        self.D = M_0.shape[0]
        self.nlags = M_0.shape[1] // M_0.shape[0] if not self.affine \
                else (M_0.shape[1]-1) // M_0.shape[0]

        self.niter = niter

        if self.sigma is None:
            self.resample_sigma(niter=1)
        if self.fullA is None:
            self.resample_A(niter=1)

    ### converting between natural and standard hyperparameters

    def _standard_to_natural(self,nu,S,M,Uinv,Vinv):
        return np.concatenate((
            AR_IWFixedA._standard_to_natural(self,nu,S),
            AR_MNFixedSigma._standard_to_natural(self,M,Uinv,Vinv),
            ))

    def _natural_to_standard(self,natparam):
        return joindicts((
            AR_IWFixedA._natural_to_standard(natparam[:2]),
            AR_MNFixedSigma._natural_to_standard(natparam[2:]),
            ))

    ### statistics

    def _get_statistics(self,data):
        return _ARBase._get_statistics(self,data)

    def _get_weighted_statistics(self,data,weights=None):
        return _ARBase._get_weighted_statistics(self,data,weights)

    ### Gibbs sampling

    def resample(self,data=[],niter=None):
        stats = self._get_statistics(data)
        niter = self.niter
        for itr in xrange(niter):
            self.resample_A(stats=stats)
            self.resample_sigma(stats=stats)

    def resample_sigma(self,data=[],stats=None):
        stats = self._get_statistics(data) if stats is None else stats
        self.sigma = sample_invwishart(**AR_IWFixedA._natural_to_standard(self,
            self.natural_hypparam[:2] + AR_IWFixedA._shape_statistics(self,stats)))

    def resample_A(self,data=[],stats=None):
        stats = self._get_statistics(data) if stats is None else stats
        self.fullA = sample_mn(**AR_MNFixedSigma._natural_to_standard(self,
            self.natural_hypparam[2:] + AR_MNFixedSigma._shape_statistics(self,stats)))

    def copy_sample(self):
        return _ARBase.copy_sample(self)

