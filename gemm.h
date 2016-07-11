#pragma once

#include <stdint.h>
#include <omp.h>

#include <algorithm>
#include <limits>

template <typename Tin1,
          typename Tin2,
          typename Tmul,
          typename Tacc,
          bool in1_t,
          bool in2_t>
void gemm_kernel(uint32_t N,
                 uint32_t M,
                 uint32_t K,
                 uint32_t blk_dim_N_local,
                 uint32_t blk_dim_M_local,
                 uint32_t blk_dim_K_local,
                 uint32_t blk_dim_N,
                 uint32_t blk_dim_M,
                 uint32_t blk_dim_K,
                 uint32_t blk_i,
                 uint32_t blk_j,
                 uint32_t blk_k,
                 const Tin1 *in1,
                 const Tin2 *in2,
                 Tacc acc[]) {

    for (uint32_t j = 0; j < blk_dim_N_local; ++j) {
        uint32_t y = blk_j * blk_dim_N + j;

        for (uint32_t i = 0; i < blk_dim_M_local; ++i) {
            uint32_t x = blk_i * blk_dim_M + i;

            for (uint32_t k = 0; k < blk_dim_K_local; ++k) {
                uint32_t z = blk_k * blk_dim_K + k;

                acc[i + j * blk_dim_M] += static_cast<Tacc>(
                    static_cast<Tmul>(in1[in1_t ? x * K + z : x + z * M]) *
                    static_cast<Tmul>(in2[in2_t ? z * N + y : z + y * K]));
            }
        }
    }
}

template <typename Tin1,
          typename Tin2,
          typename Tout,
          typename Tmul,
          typename Tacc,
          bool in1_t,
          bool in2_t>
void gemm(uint32_t M,
          uint32_t N,
          uint32_t K,
          Tacc alpha,
          const Tin1 *in1,
          const Tin2 *in2,
          Tacc beta,
          Tout *out) {

    const uint32_t blk_dim = 16;
    const uint32_t blk_dim_M = blk_dim;
    const uint32_t blk_dim_N = blk_dim;
    const uint32_t blk_dim_K = blk_dim;

    const uint32_t blk_M = (M + blk_dim_M - 1) / blk_dim_M;
    const uint32_t blk_N = (N + blk_dim_N - 1) / blk_dim_N;
    const uint32_t blk_K = (K + blk_dim_K - 1) / blk_dim_K;

    Tacc acc[blk_dim_M * blk_dim_N];

#pragma omp parallel for private(acc)
    for (uint32_t blk_j = 0; blk_j < blk_N; ++blk_j) {
        uint32_t blk_dim_N_local = std::min(blk_dim_N, N - blk_j * blk_dim_N);

        for (uint32_t blk_i = 0; blk_i < blk_M; ++blk_i) {
            uint32_t blk_dim_M_local = std::min(blk_dim_M, M - blk_i * blk_dim_M);

            // Clear accumulator
            for (uint32_t j = 0; j < blk_dim_N_local; ++j) {
                for (uint32_t i = 0; i < blk_dim_M_local; ++i) {
                    acc[i + j * blk_dim_M] = 0;
                }
            }

            // Apply kernel
            for (uint32_t blk_k = 0; blk_k < blk_K; ++blk_k) {
                uint32_t blk_dim_K_local = std::min(blk_dim_K, K - blk_k * blk_dim_K);

                gemm_kernel<Tin1, Tin2, Tmul, Tacc, in1_t, in2_t>(
                    N, M, K,
                    blk_dim_N_local, blk_dim_M_local, blk_dim_K_local,
                    blk_dim_N, blk_dim_M, blk_dim_K,
                    blk_i, blk_j, blk_k,
                    in1, in2, acc);
            }

            // Write accumulator to output
            for (uint32_t j = 0; j < blk_dim_N_local; ++j) {
                uint32_t y = blk_j * blk_dim_N + j;
                for (uint32_t i = 0; i < blk_dim_M_local; ++i) {
                    uint32_t x = blk_i * blk_dim_M + i;

                    out[x + y * M] = static_cast<Tout>(
                        std::max(
                            std::min(
                                alpha * acc[i + j * blk_dim_M] + beta * out[x + y * M],
                                static_cast<Tacc>(std::numeric_limits<Tout>::max())
                            ),
                            static_cast<Tacc>(std::numeric_limits<Tout>::min())
                        )
                    );
                }
            }
        }
    }
}

template <typename Tin1,
          typename Tin2,
          typename Tout,
          typename Tmul,
          typename Tacc>
void gemm(bool in1_t,
          bool in2_t,
          uint32_t M,
          uint32_t N,
          uint32_t K,
          Tacc alpha,
          const Tin1 *in1,
          const Tin2 *in2,
          Tacc beta,
          Tout *out) {

    if (!in1_t && !in2_t) {
        gemm<Tin1, Tin2, Tout, Tmul, Tacc, false, false>(M, N, K, alpha, in1, in2, beta, out);
    } else if (in1_t && !in2_t) {
        gemm<Tin1, Tin2, Tout, Tmul, Tacc, true , false>(M, N, K, alpha, in1, in2, beta, out);
    } else if (!in1_t && in2_t) {
        gemm<Tin1, Tin2, Tout, Tmul, Tacc, false, true >(M, N, K, alpha, in1, in2, beta, out);
    } else if (in1_t && in2_t) {
        gemm<Tin1, Tin2, Tout, Tmul, Tacc, true , true >(M, N, K, alpha, in1, in2, beta, out);
    }
}

