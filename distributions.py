from __future__ import division
import numpy as np
from numpy import newaxis as na
import scipy.linalg

from pyhsmm.basic.abstractions import GibbsSampling, MaxLikelihood
from pyhsmm.util.stats import sample_mniw, sample_invwishart, getdatasize

from util import AR_striding, undo_AR_striding

# TODO take care of D always

class _ARBase(MaxLikelihood):
    affine = False

    @property
    def params(self):
        if self.affine:
            return dict(A=self.A,b=self.b,Sigma=self.Sigma)
        else:
            return dict(A=self.A,Sigma=self.Sigma)

    @property
    def nlags(self):
        return self.A.shape[1] // self.A.shape[0]

    @property
    def D(self):
        return self.A.shape[0]

    def _get_Sigma(self):
        return self._Sigma

    def _set_Sigma(self,Sigma):
        self._Sigma = Sigma
        self._Sigma_chol = None

    Sigma = property(_get_Sigma,_set_Sigma)

    @property
    def Sigma_chol(self):
        if not hasattr(self,'_Sigma_chol') or self._Sigma_chol is None:
            self._Sigma_chol = np.linalg.cholesky(self.Sigma)
        return self._Sigma_chol

    def log_likelihood(self,x):
        if hasattr(self,'broken'):
            return np.repeat(-np.inf,x.shape[0]) if isinstance(x,np.ndarray) else -np.inf
        try:
            chol = self.Sigma_chol
            D = self.D
            return -1./2. * scipy.linalg.solve_triangular(
                        chol,
                        (x[:,:-D].dot(self.A.T) - x[:,-D:]).T \
                                + (self.b[:,na] if self.affine else 0),
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

        randomness = np.random.normal(size=(length,D)).dot(self.Sigma_chol.T)

        for itr in range(length):
            out[itr+self.nlags] = self.A.dot(strided_out[itr]) \
                    + randomness[itr] + (self.b if self.affine else 0)

        return out[self.nlags:]

    def max_likelihood(self,data,weights=None,nlags=None,D=None):
        Syy, Sytyt, Syyt, n = self._get_weighted_statistics(data,weights,nlags,D)

        try:
            self.A = np.linalg.solve(Sytyt, Syyt.T).T # TODO call psd solver
            self.Sigma = (Syy - self.A.dot(Syyt.T))/n
            if self.affine:
                self.b = self.A[:,0]
                self.A = self.A[:,1:]
        except np.linalg.LinAlgError:
            # broken!
            self.broken = True

    def _get_weighted_statistics(self,data,weights=None,nlags=None,D=None):
        assert D is not None
        if weights is None:
            return self._get_statistics(data,D=D)
        else:
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

    def _get_statistics(self,data,D):
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

class MNIW(_ARBase, GibbsSampling):
    def __init__(self,dof=None,S=None,M=None,K=None,affine=False,
            A=None,b=None,Sigma=None):
        self.A = A
        self.b = b
        self.Sigma = Sigma

        self.dof = dof
        self.S = S
        self.M = M
        self.K = K

        self.affine = affine or (b is not None)

        if (A,Sigma,b) == (None,None,None) and None not in (dof,S,M,K):
            self.resample() # initialize from prior

    @property
    def hypparams(self):
        return dict(dof=self.dof,S=self.S,M=self.M,K=self.K)

    @property
    def D(self):
        return self.M.shape[0]

    @property
    def nlags(self):
        return self.M.shape[1] // self.M.shape[0]

    def resample(self,data=[]):
        self.A, self.Sigma = sample_mniw(
                *self._posterior_hypparams(*self._get_statistics(data,D=self.D)))
        if self.affine:
            self.b = self.A[:,0]
            self.A = self.A[:,1:]

        return self

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


class FixedNoiseCov(_ARBase, GibbsSampling):
    def __init__(self,Sigma=None,A=None,b=None,M=None,sigmasq_A=None,affine=False):
        self.Sigma = Sigma # fixed noise cov
        self.A = A
        self.b = b

        self.M = M
        self.sigmasq_A = sigmasq_A # scalar variance of prior on entries of A

        self.affine = affine or (b is not None)

        if (A,b) == (None,None) and None not in (M,sigmasq_A):
            self.resample() # initialize from prior

    @property
    def hypparams(self):
        return dict(M=self.M,sigmasq_A=self.sigmasq_A)

    def resample(self,data=[]):
        J_n_l, J_n_r, M_n = self._posterior_hypparams(
                *self._get_statistics(data,D=self.D))
        chol = np.linalg.cholesky(J_n_r)
        self.A = M_n + \
                scipy.linalg.solve_triangular(
                        chol,
                        np.random.normal(size=M_n.T.shape),
                        lower=True).T / np.sqrt(J_n_l)
        if self.affine:
            self.b = self.A[:,0]
            self.A = self.A[:,1:]

        return self

    def _posterior_hypparams(self,Syy,Sytyt,Syyt,n):
        if n > 0:
            J_n_r = Sytyt
            J_n_r.flat[::J_n_r.shape[0]+1] += 1
            J_n_l = 1 + 1./self.sigmasq_A
            M_n = np.linalg.solve(J_n_r,(self.M / self.sigmasq_A + Syyt).T).T \
                    / (1 + 1./self.sigmasq_A)
        else:
            M_n = self.M
            J_n_r = np.eye(M_n.shape[1])
            J_n_l = 1./self.sigmasq_A
        return J_n_l, J_n_r, M_n

    def _get_statistics(self,data,D):
        # NOTE: the strategy here is to reshape the data according to the fixed
        # noise covariance, then use the same statistics methods as the parent
        assert isinstance(data,(np.ndarray,list))
        if isinstance(data,np.ndarray):
            data = self._reshape_data(data)
        else:
            data = map(self._reshape_data,data)
        return super(FixedNoiseCov,self)._get_statistics(data,D)

    def _get_weighted_statistics(self,data,weights):
        assert isinstance(data,(np.ndarray,list))
        if isinstance(data,np.ndarray):
            data = self._reshape_data(data)
        else:
            data = map(self._reshape_data,data)
        return super(FixedNoiseCov,self)._get_weighted_statistics(data,weights)

    def _reshape_data(self,data):
        assert isinstance(data,np.ndarray)
        if len(data) > 0:
            data = AR_striding(
                    scipy.linalg.solve_triangular(
                        self.Sigma_chol,
                        undo_AR_striding(data,self.nlags).T,
                        lower=True).T,
                    nlags=self.nlags)
        return data

class FixedARCoefficients(_ARBase, GibbsSampling):
    def __init__(self,A=None,b=None,Sigma=None,dof=None,S=None,affine=False):
        self.A = A
        self.b = b
        self.Sigma = Sigma

        self.S = S
        self.dof = dof

        self.affine = affine

        if Sigma is None and None not in (dof,S):
            self.resample() # initialize from prior

    @property
    def hypparams(self):
        return dict(S=self.S,dof=self.dof)

    def resample(self,data=[]):
        self.Sigma = sample_invwishart(
                *self._posterior_hypparams(
                    *self._get_statistics(data,D=self.D)))
        return self

    def _get_statistics(self,data,D):
        # NOTE: similar to pybasicbayes/distributions.py:GaussianFixedMean
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                centered = self._center_data(data)
                sumsq = centered.T.dot(centered)
            else:
                sumsq = sum(c.T.dot(c) for c in map(self._center_data,data))
        else:
            sumsq = None

        return n, sumsq

    def _get_weighted_statistics(self,data,weights):
        if isinstance(data,np.ndarray):
            neff = weights.sum()
            if neff > 0:
                centered = self._center_data(data)
                sumsq = centered.T.dot(weights[:,na]*centered)
            else:
                sumsq = None
        else:
            neff = sum(w.sum() for w in weights)
            if neff > 0:
                sumsq = sum(c.T.dot(w[:,na]*c) for c in map(self._center_data,data))
            else:
                sumsq = None

        return neff, sumsq

    def _center_data(self,data):
        D = self.D
        return ((data[:,:-D].dot(self.A.T) - data[:,-D:]).T \
                + (self.b[:,na] if self.affine else 0)).T

    def _posterior_hypparams(self,n,sumsq):
        # NOTE: same as pybasicbayes/distributions.py:GaussianFixedMean
        S_0, dof_0 = self.S, self.dof
        if n > 0:
            S_n = S_0 + sumsq
            dof_n = dof_0 + n
        else:
            S_n = S_0
            dof_n = dof_0
        return S_n, dof_n

class NIWNonConj(_ARBase, GibbsSampling):
    def __init__(self,A=None,b=None,Sigma=None,
            M=None,sigmasq_A=None,dof=None,S=None,affine=False):
        self._Sigma_obj = FixedARCoefficients(A=A if A is not None else M,
                b=b,Sigma=Sigma,dof=dof,S=S,affine=False)
        self._coeff_obj = FixedNoiseCov(Sigma=self._Sigma_obj.Sigma,
                A=A if A is not None else M,
                b=b,M=M,sigmasq_A=sigmasq_A,affine=affine)
        self.A = self._coeff_obj.A
        self.b = self._coeff_obj.b

    def _get_A(self):
        return self._coeff_obj.A

    def _set_A(self,val):
        self._coeff_obj.A = val
        self._Sigma_obj.A = val

    A = property(_get_A,_set_A)

    def _get_b(self):
        return self._coeff_obj.b

    def _set_b(self,val):
        self._coeff_obj.b = val
        self._Sigma_obj.b = val

    b = property(_get_b,_set_b)

    def _get_Sigma(self):
        return self._Sigma_obj.Sigma

    def _set_Sigma(self,val):
        self._Sigma_obj.Sigma = val
        self._coeff_obj.Sigma = val

    Sigma = property(_get_Sigma,_set_Sigma)

    @property
    def hypparams(self):
        return dict(M=self._coeff_obj.M,sigmasq_A=self._coeff_obj.sigmasq_A,
                dof=self._Sigma_obj.dof,S=self._Sigma_obj.S)

    def resample(self,data=[],niter=10):
        for itr in xrange(niter):
            self._Sigma_obj.resample(data)
            self.Sigma = self._Sigma_obj.Sigma
            self._coeff_obj.resample(data)
            self.A = self._coeff_obj.A
            self.b = self._coeff_obj.b

