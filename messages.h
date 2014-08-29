#include <Eigen/Core>
#include <stdint.h> // int32_t
#include <omp.h> // omp_get_num_threads, omp_get_thread_num
#include <limits> // infinity
#include <iostream>

#include "nptypes.h"
#include "util.h"

using namespace Eigen;
using namespace nptypes;
using namespace std;

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
            Type *randseq)
    {
        Map<Matrix<Type,Dynamic,Dynamic,RowMajor>,Aligned,OuterStride<>>
            edata(data,T-nlags,D*(nlags+1),OuterStride<>(D));

        NPMatrix<Type> eA(A,M,M);

        NPMatrix<Type> enatparams(natparams,M*D*(nlags+1),D*(nlags+1));
        NPMatrix<Type> estats(stats,D*(nlags+1),D*(nlags+1));

        // allocate temporaries
        // NPMatrix<Type> ealphan(alphan,T-nlags,M);
        MatrixXd ealphan(T-nlags,M); // NOTE: memory allocation

        Type temp_buf[D*(nlags+1)] __attribute__((aligned(16)));
        Map<Matrix<Type,Dynamic,1>,Aligned> etemp(temp_buf,D*(nlags+1));
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
                        enatparams.block(D*(nlags+1)*m,0,D*(nlags+1),D*(nlags+1))
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
        stateseq[T-nlags-1] =
            util::sample_discrete(M,ealphan.row(T-nlags-1).data(),randseq[T-1]);
        counts[stateseq[T-nlags-1]] += 1;
        for (int t=T-nlags-2; t >= 0; t--) {
            elikes = eA.col(stateseq[t+1]).transpose().array() * ealphan.row(t).array();
            stateseq[t] = util::sample_discrete(M,elikes.data(),randseq[t]);

            counts[stateseq[t]] += 1;
            estats.template selfadjointView<Lower>().rankUpdate(edata.row(t).transpose(),1.);
        }

        estats.template triangularView<Upper>() = estats.transpose();

        return logtot;
    }
};
