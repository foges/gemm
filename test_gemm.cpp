#include <assert.h>

#include <iomanip>
#include <iostream>

#include "gemm.h"
#include "timestamp.h"

int test(bool in1_t,
         bool in2_t,
         uint32_t M,
         uint32_t N,
         uint32_t K) {

    float *in1 = new float[M * K];
    float *in2 = new float[K * N];
    float *out = new float[M * N];

    float alpha = 4.f;
    float beta  = 5.f;

    for (uint32_t i = 0; i < M * K; ++i) {
        in1[i] = 1.f;
    }

    for (uint32_t i = 0; i < K * N; ++i) {
        in2[i] = 2.f;
    }

    for (uint32_t i = 0; i < M * N; ++i) {
        out[i] = 3.f;
    }

    gemm<float, float, float, float, float>(
        in1_t, in2_t, M, N, K, alpha, in1, in2, beta, out);

    for (uint32_t i = 0; i < M * N; ++i) {
        if (out[i] != out[0]) {
            std::cout << "ERROR: " << out[i] << " " << out[0] <<std::endl;
            break;
        }
    }

    return out[0];
}

int main() {
    assert(test(false, false,  16, 18,  17) ==  151.f);
    assert(test(false, true ,  9 , 19,  13) ==  119.f);
    assert(test(true , false,  12, 3 ,  8 ) ==   79.f);
    assert(test(true , true ,  11, 32,  9 ) ==   87.f);

    assert(test(false, false, 256,  2, 256) == 2063.f);
    assert(test(true , false, 256,  2, 256) == 2063.f);
    assert(test(false, true , 256,  2, 256) == 2063.f);
    assert(test(true , true , 256,  2, 256) == 2063.f);

    uint32_t N = 1024;
    uint64_t t0 = timestamp_us();
    test(false, false, N,  N, N);
    uint64_t t1 = timestamp_us();
    test(false, true , N,  N, N);
    uint64_t t2 = timestamp_us();
    test(true , false, N,  N, N);
    uint64_t t3 = timestamp_us();
    test(true , true , N,  N, N);
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

