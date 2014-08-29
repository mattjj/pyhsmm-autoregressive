# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -DNDEBUG -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport int32_t, int64_t
from cython cimport floating
from cython.parallel import prange

cdef extern from "messages.h":
    cdef cppclass dummy[Type]:
        dummy()
        Type resample_arhmm(
            int M, int T, int D, int nlags,
            Type *pi_0, Type *A,
            Type *natparams, Type *normalizers,
            Type *data,
            Type *stats, int *counts, int32_t *stateseq,
            Type *randseq) nogil
        void initParallel()

def resample_arhmm(
        double[::1] pi_0,
        double[:,::1] A,
        double[:,:,::1] params,
        double[::1] normalizers,
        list datas,
        list stateseqs,
        list randseqs):
    cdef int i
    cdef dummy[double] ref

    cdef int M = params.shape[0]   # number of states
    cdef int K = len(datas)        # number of sequences
    cdef int D = datas[0].shape[1] # dimension of data (unstrided)
    cdef bool affine = params.shape[2] % D
    cdef int nlags = (params.shape[2] - affine) / D
    cdef int32_t[::1] Ts = np.array([d.shape[0] for d in datas]).astype('int32')

    cdef vector[double[:,:]] datas_v = datas

    cdef vector[int32_t*] stateseqs_v
    cdef int32_t[:] temp
    for i in range(K):
        temp = stateseqs[i]
        stateseqs_v.push_back(&temp[0])

    cdef vector[double*] randseqs_v
    cdef double[:] temp2
    for i in range(K):
        temp2 = randseqs[i]
        randseqs_v.push_back(&temp2[0])

    # NOTE: 2*K for false sharing
    cdef double[:,:,:,::1] stats = np.zeros((2*K,M,params.shape[1],params.shape[2]))
    cdef int[:,::1] ns = np.zeros((2*K,M),dtype='int')
    cdef double[::1] likes = np.zeros(K)

    ref.initParallel()
    with nogil:
        for i in prange(K):
            likes[i] = ref.resample_arhmm(
                    M,Ts[i],D,nlags,
                    &pi_0[0],&A[0,0],
                    &params[0,0,0],&normalizers[0],
                    &datas_v[i][0,0],
                    &stats[2*i,0,0,0],&ns[2*i,0],&stateseqs_v[i][0],
                    &randseqs_v[i][0])

    allstats = []
    for statmat, n in zip(np.sum(stats,axis=0),np.sum(ns,axis=0)):
        xxT, yxT, yyT = statmat[:-D,:-D], statmat[-D:,:-D], statmat[-D:,-D:]
        allstats.append([yyT,yxT,xxT,n])

    return allstats, np.asarray(likes)

