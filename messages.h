#include <Eigen/Core>
#include <stdint.h> // int32_t
#include <omp.h> // omp_get_num_threads, omp_get_thread_num
#include <limits> // infinity

#include "nptypes.h"
#include "util.h"

using namespace Eigen;
using namespace nptypes;

// TODO gotta handle affine

template <typename Type>
class dummy
{
    public:

    static void initParallel() {
        Eigen::initParallel();
    }

    static Type resample_arhmm(
            int M, int T, int D, int nlags,
            Type *pi_0, Type *A,
            Type *natparams, Type *normalizers,
            Type *data,
            Type *stats, int32_t *counts, int32_t *stateseq,
            Type *randseq,
            Type *alphan)
    {
        int sz = D*(nlags+1);
        Map<Matrix<Type,Dynamic,Dynamic,RowMajor>,Aligned,OuterStride<>>
            edata(data,T-nlags,sz,OuterStride<>(D));

        NPMatrix<Type> eA(A,M,M);

        NPMatrix<Type> enatparams(natparams,M*sz,sz);
        NPMatrix<Type> estats(stats,M*sz,sz);

        // allocate temporaries
        NPMatrix<Type> ealphan(alphan,T-nlags,M);
        // MatrixXd ealphan(T-nlags,M); // NOTE: memory allocation

        Type temp_buf[sz] __attribute__((aligned(16)));
        NPVector<Type> etemp(temp_buf,sz);
        Type in_potential_buf[M] __attribute__((aligned(16)));
        NPRowVector<Type> ein_potential(in_potential_buf,M);
        Type likes_buf[M] __attribute__((aligned(16)));
        NPRowVector<Type> elikes(likes_buf,M);

        Type norm, cmax, logtot = 0.;

        // likelihoods and forward messages
        ein_potential = NPMatrix<Type>(pi_0,1,M);
        for (int t=0; t < T-nlags; t++) {
            if ((edata.row(t).array() == edata.row(t).array()).all()) {
                for (int m=0; m < M; m++) {
                    etemp =
                        enatparams.block(m*sz,0,sz,sz)
                        * edata.row(t).transpose();
                        //.template selfadjointView<Lower>() * edata.row(t).transpose();
                    elikes(m) = etemp.dot(edata.row(t).transpose()) - normalizers[m];
                }


                cmax = elikes.maxCoeff();
                ealphan.row(t) = ein_potential.array() * (elikes.array() - cmax).exp();
                norm = ealphan.row(t).sum();
                ealphan.row(t) /= norm;
                logtot += log(norm) + cmax;
            } else {
                ealphan.row(t) = ein_potential;
            }


            ein_potential = ealphan.row(t) * eA;
        }

        // backward sampling and stats gathering
        ein_potential.setOnes();
        for (int t=T-nlags-1; t >= 0; t--) {
            elikes = ein_potential.array() * ealphan.row(t).array();
            stateseq[t] = util::sample_discrete(M,elikes.data(),randseq[t]);
            ein_potential = eA.col(stateseq[t]).transpose();

            if ((edata.row(t).array() == edata.row(t).array()).all()) {
                counts[stateseq[t]] += 1;
                // NOTE: could do a rank-1 update
                estats.block(stateseq[t]*sz,0,sz,sz).noalias()
                    += edata.row(t).transpose() * edata.row(t);
            }
        }

        return logtot;
    }
};
