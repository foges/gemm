#include <assert.h>

#include <iomanip>
#include <iostream>

#include "gemm.h"
#include "timestamp.h"

template <typename Tin1,
          typename Tin2,
          typename Tout,
          typename Tacc>
void test(bool in1_t,
          bool in2_t,
          uint32_t M,
          uint32_t N,
          uint32_t K) {

    Tin1 *in1 = new Tin1[M * K];
    Tin2 *in2 = new Tin2[K * N];
    Tout *out = new Tout[M * N];

    Tacc alpha = 2;
    Tacc beta  = 1;

    for (uint32_t i = 0; i < M * K; ++i) {
        in1[i] = 1;
    }

    for (uint32_t i = 0; i < K * N; ++i) {
        in2[i] = 2;
    }

    for (uint32_t i = 0; i < M * N; ++i) {
        out[i] = 1;
    }

    Tout expected = alpha * K * in1[0] * in2[0] + beta * out[0];

    gemm<Tin1, Tin2, Tout, Tacc, Tacc>(
        in1_t, in2_t, M, N, K, alpha, in1, in2, beta, out);

    for (uint32_t i = 0; i < M * N; ++i) {
        if (out[i] != expected) {
            std::cout << "ERROR(" << i << "): " << (double)out[i] << " "
                      << (double)expected << std::endl;
            break;
        }
    }
}

int main() {
    using Tin1 = int8_t;
    using Tin2 = uint8_t;
    using Tout = int8_t;
    using Tacc = int16_t;

    test<Tin1, Tin2, Tout, Tacc>(false, false,  3, 4, 5);
    test<Tin1, Tin2, Tout, Tacc>(false, true ,  9 , 19,  13);
    test<Tin1, Tin2, Tout, Tacc>(true , false,  12, 3 ,  8 );
    test<Tin1, Tin2, Tout, Tacc>(true , true ,  11, 32,  9 );

    //test<float, float, float, float>(false, false, 256,  2, 256) == 2063.f;
    //test<float, float, float, float>(true , false, 256,  2, 256) == 2063.f;
    //test<float, float, float, float>(false, true , 256,  2, 256) == 2063.f;
    //test<float, float, float, float>(true , true , 256,  2, 256) == 2063.f;

    uint32_t N = 2560;
    uint64_t t0 = timestamp_us();
    test<int8_t, uint8_t, int16_t, int16_t>(false, false, N,  N, N);
    uint64_t t1 = timestamp_us();
    test<int8_t, uint8_t, int16_t, int16_t>(false, true , N,  N, N);
    uint64_t t2 = timestamp_us();
    test<int8_t, uint8_t, int16_t, int16_t>(true , false, N,  N, N);
    uint64_t t3 = timestamp_us();
    test<int8_t, uint8_t, int16_t, int16_t>(true , true , N,  N, N);
    uint64_t t4 = timestamp_us();

    std::cout << "NN(" << N << "): " << std::fixed << std::setprecision(4)
              << 2. * N * N * N / (t1 - t0) / 1e3 << " GFlop/s" << std::endl;
    std::cout << "NT(" << N << "): " << std::fixed << std::setprecision(4)
              << 2. * N * N * N / (t2 - t1) / 1e3 << " GFlop/s" << std::endl;
    std::cout << "TN(" << N << "): " << std::fixed << std::setprecision(4)
              << 2. * N * N * N / (t3 - t2) / 1e3 << " GFlop/s" << std::endl;
    std::cout << "TT(" << N << "): " << std::fixed << std::setprecision(4)
              << 2. * N * N * N / (t4 - t3) / 1e3 << " GFlop/s" << std::endl;

}

