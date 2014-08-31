# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -DNDEBUG -fopenmp -std=c++11 -DEIGEN_NO_MALLOC
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
            int M, int T, int D, int nlags, bool affine,
            Type *pi_0, Type *A,
            Type *natparams, Type *normalizers,
            Type *data,
            Type *stats, int32_t *counts, int32_t *transcounts, int32_t *stateseq,
            Type *randseq, Type *alphan) nogil
        void initParallel()

cdef dummy[double] ref
ref.initParallel()

def resample_arhmm(
        double[::1] pi_0,
        double[:,::1] A,
        double[:,:,::1] params,
        double[::1] normalizers,
        list datas,
        list stateseqs,
        list randseqs,
        list alphans):
    cdef int i, j
    cdef dummy[double] ref

    cdef int M = params.shape[0]   # number of states
    cdef int K = len(datas)        # number of sequences
    cdef int D = datas[0].shape[1] # dimension of data (unstrided)
    cdef bool affine = params.shape[2] % D
    cdef int nlags = (params.shape[2] - affine) / D - 1
    cdef int32_t[::1] Ts = np.array([d.shape[0] for d in datas]).astype('int32')

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

    cdef vector[double*] datas_v
    cdef double[:,:] temp3
    for i in range(K):
        temp3 = datas[i]
        datas_v.push_back(&temp3[0,0])

    cdef vector[double*] alphans_v
    cdef double[:,:] temp4
    for i in range(K):
        temp4 = alphans[i]
        alphans_v.push_back(&temp4[0,0])

    # NOTE: 2*K for false sharing
    cdef double[:,:,:,::1] stats = np.zeros((2*K,M,params.shape[1],params.shape[2]))
    cdef int32_t[:,::1] ns = np.zeros((2*K,M),dtype='int32')
    cdef int32_t[:,:,::1] transcounts = np.zeros((2*K,M,M),dtype='int32')
    cdef double[::1] likes = np.zeros(K)

    # ref.initParallel()
    with nogil:
        for j in prange(K+1):
            if j != 0:
                i = j-1
                likes[i] = ref.resample_arhmm(
                        M,Ts[i],D,nlags,affine,
                        &pi_0[0],&A[0,0],
                        &params[0,0,0],&normalizers[0],
                        datas_v[i],
                        &stats[2*i,0,0,0],&ns[2*i,0],&transcounts[2*i,0,0],
                        stateseqs_v[i], randseqs_v[i],alphans_v[i])

    allstats = []
    for statmat, n in zip(np.sum(stats,0),np.sum(ns,0)):
        xxT, yxT, yyT = statmat[:-D,:-D], statmat[-D:,:-D], statmat[-D:,-D:]
        allstats.append([yyT,yxT,xxT,n])

    return allstats, np.sum(transcounts,axis=0), np.asarray(likes)

