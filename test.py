from __future__ import division
import numpy as np

from util import AR_striding

import messages

D = 2
K = 1
M = 2
T = 10
nlags = 2

pi_0 = np.eye(M)[0]
A = np.eye(M)

params = np.random.randn((nlags+1)*D,(nlags+1)*D);
params = params.dot(params.T)
params = np.tile(params[None,:,:],(M,1,1))

normalizers = np.zeros(M)

datas = [np.random.randn(T,D) for _ in range(K)]

stateseqs = [np.empty(T,dtype='int32') for _ in range(K)]
randseqs = [np.random.uniform(size=T) for _ in range(K)]

stats, likes = messages.resample_arhmm(pi_0,A,params,normalizers,datas,stateseqs,randseqs)

