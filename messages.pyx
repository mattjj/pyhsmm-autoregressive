# distutils: language = c++
# distutils: extra_compile_args = -O3 -w -DNDEBUG -fopenmp -std=c++11 -DEIGEN_NO_MALLOC
# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False
# cython: nonecheck = False

import sys

import numpy as np
cimport numpy as np

from libc.stdio cimport fflush, printf, stdout

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

def resample_arhmm(
        double[::1] pi_0,
        double[:,::1] A,
        double[:,:,::1] params,
        double[::1] normalizers,
        int D,
        int32_t[::1] Ts,
        int64_t[::1] datas,
        int64_t[::1] stateseqs,
        int64_t[::1] randseqs,
        int64_t[::1] alphans):
    cdef int i, j
    cdef dummy[double] ref

    cdef int M = params.shape[0]   # number of states
    cdef int K = datas.shape[0]    # number of sequences
    cdef bool affine = params.shape[2] % D
    cdef int nlags = (params.shape[2] - affine) / D - 1

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
                        <double*>datas[i],
                        &stats[2*i,0,0,0],&ns[2*i,0],&transcounts[2*i,0,0],
                        <int32_t*>stateseqs[i],<double*>randseqs[i],<double*>alphans[i])
            printf("%d finished!\n",j)
            fflush(stdout)

    return np.asarray(stats), np.asarray(ns), np.asarray(transcounts), np.asarray(likes)

