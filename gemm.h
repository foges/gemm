#pragma once

#include <stdint.h>
#include <omp.h>

#include <immintrin.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>

#define _mm256_set_m128i(v0, v1)  _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)

#define _mm256_setr_m128i(v0, v1) _mm256_set_m128i((v1), (v0))

template <uint32_t blk_dim_M,
          uint32_t blk_dim_N,
          uint32_t blk_dim_K>
void gemm_kernel(const int8_t  *__restrict__ in1,
                 const uint8_t *__restrict__ in2,
                 int16_t       *__restrict__ acc) {

    for (uint32_t j = 0; j < blk_dim_N; ++j) {
        for (uint32_t i = 0; i < blk_dim_M / 16; ++i) {
            __m256i c = _mm256_setzero_si256();

            for (uint32_t k = 0; k < blk_dim_K / 2; ++k) {
                __m256i a, b;
                __m128i tmp_lo, tmp_hi;

                tmp_lo = _mm_load_si128((__m128i*)(in1 + (2 * k + 0) * blk_dim_M + i * 16));
                tmp_hi = _mm_load_si128((__m128i*)(in1 + (2 * k + 1) * blk_dim_M + i * 16));

                a = _mm256_setr_m128i(tmp_lo, tmp_hi);

                tmp_lo = _mm_set1_epi8(in2[j * blk_dim_K + 2 * k + 0]);
                tmp_hi = _mm_set1_epi8(in2[j * blk_dim_K + 2 * k + 1]);

                b = _mm256_setr_m128i(tmp_lo, tmp_hi);

                __m256i res = _mm256_maddubs_epi16(b, a);

                c = _mm256_adds_epi16(c, res);
            }

            _mm256_store_si256((__m256i*)(acc + j * blk_dim_M + i * 16), c);
        }
    }
}

template <typename Tin1,
          typename Tin2,
          typename Tmul,
          typename Tacc,
          uint32_t blk_dim_M,
          uint32_t blk_dim_N,
          uint32_t blk_dim_K>
void gemm_kernel(const Tin1 *__restrict__ in1,
                 const Tin2 *__restrict__ in2,
                 Tacc *__restrict__ acc) {

    for (uint32_t j = 0; j < blk_dim_N; ++j) {
        for (uint32_t i = 0; i < blk_dim_M; ++i) {
            for (uint32_t k = 0; k < blk_dim_K; ++k) {

                acc[i + j * blk_dim_M] += static_cast<Tacc>(
                    static_cast<Tmul>(in1[i * blk_dim_K + k]) *
                    static_cast<Tmul>(in2[j * blk_dim_K + k]));
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
          const Tin1 *__restrict__ in1,
          const Tin2 *__restrict__ in2,
          Tacc beta,
          Tout *__restrict__ out) {

    const uint32_t blk_dim = 16;
    const uint32_t blk_dim_M = blk_dim;
    const uint32_t blk_dim_N = blk_dim;
    const uint32_t blk_dim_K = blk_dim;

    const uint32_t blk_M = (M + blk_dim_M - 1) / blk_dim_M;
    const uint32_t blk_N = (N + blk_dim_N - 1) / blk_dim_N;
    const uint32_t blk_K = (K + blk_dim_K - 1) / blk_dim_K;

//    Tacc acc[blk_dim_M * blk_dim_N];
//    Tin1 in1_tmp[blk_dim_M * blk_dim_K];
//    Tin2 in2_tmp[blk_dim_N * blk_dim_K];

    size_t acc_sz     = blk_dim_M * blk_dim_N;
    size_t in1_tmp_sz = blk_dim_M * blk_dim_K;
    size_t in2_tmp_sz = blk_dim_N * blk_dim_K;

    Tacc *acc_arr     = (Tacc*)_mm_malloc(omp_get_max_threads() * sizeof(Tacc) * acc_sz,     64);
    Tin1 *in1_tmp_arr = (Tin1*)_mm_malloc(omp_get_max_threads() * sizeof(Tin1) * in1_tmp_sz, 64);
    Tin2 *in2_tmp_arr = (Tin2*)_mm_malloc(omp_get_max_threads() * sizeof(Tin2) * in2_tmp_sz, 64);

    assert(acc_arr     != nullptr);
    assert(in1_tmp_arr != nullptr);
    assert(in2_tmp_arr != nullptr);

#pragma omp parallel for // private(acc) private(in1_tmp) private(in2_tmp)
    for (uint32_t blk_i = 0; blk_i < blk_M; ++blk_i) {
        uint32_t blk_dim_M_local = std::min(blk_dim_M, M - blk_i * blk_dim_M);

        Tacc *acc     = acc_arr     + omp_get_thread_num() * acc_sz;
        Tin1 *in1_tmp = in1_tmp_arr + omp_get_thread_num() * in1_tmp_sz;
        Tin2 *in2_tmp = in2_tmp_arr + omp_get_thread_num() * in2_tmp_sz;

        for (uint32_t blk_j = 0; blk_j < blk_N; ++blk_j) {
            uint32_t blk_dim_N_local = std::min(blk_dim_N, N - blk_j * blk_dim_N);

            // Clear accumulator
            std::memset(acc, 0, acc_sz * sizeof(Tacc));

            // Apply kernel
            for (uint32_t blk_k = 0; blk_k < blk_K; ++blk_k) {
                uint32_t blk_dim_K_local = std::min(blk_dim_K, K - blk_k * blk_dim_K);

                // Copy to temp
                std::memset(in1_tmp, 0, in1_tmp_sz * sizeof(Tin1));
                for (uint32_t i = 0; i < blk_dim_M_local; ++i) {
                    uint32_t x = blk_i * blk_dim_M + i;
                    for (uint32_t k = 0; k < blk_dim_K_local; ++k) {
                        uint32_t z = blk_k * blk_dim_K + k;

                        in1_tmp[i * blk_dim_K + k] = in1[in1_t ? x * K + z : x + z * M];
                    }
                }

                std::memset(in2_tmp, 0, in2_tmp_sz * sizeof(Tin2));
                for (uint32_t j = 0; j < blk_dim_N_local; ++j) {
                    uint32_t y = blk_j * blk_dim_N + j;
                    for (uint32_t k = 0; k < blk_dim_K_local; ++k) {
                        uint32_t z = blk_k * blk_dim_K + k;

                        in2_tmp[j * blk_dim_K + k] = in2[in2_t ? z * N + y : z + y * K];
                    }
                }

                gemm_kernel<Tin1, Tin2, Tmul, Tacc, blk_dim_M, blk_dim_N, blk_dim_K>(
                    in1_tmp, in2_tmp, acc);

                //gemm_kernel<blk_dim_M, blk_dim_N, blk_dim_K>(in1_tmp, in2_tmp, acc);
            }

            // Write accumulator to output
            for (uint32_t j = 0; j < blk_dim_N_local; ++j) {
                uint32_t y = blk_j * blk_dim_N + j;
                for (uint32_t i = 0; i < blk_dim_M_local; ++i) {
                    uint32_t x = blk_i * blk_dim_M + i;

                    out[x + y * M] = static_cast<Tout>(
                        std::max<Tacc>(
                            std::min<Tacc>(
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

    _mm_free(acc_arr);
    _mm_free(in1_tmp_arr);
    _mm_free(in2_tmp_arr);
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

