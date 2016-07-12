#include <stdint.h>

#include <immintrin.h>

#include <cstring>
#include <iostream>

//template <typename T>
//void print(__m256i val, const T *tmp, size_t sz) {
//    _mm256_store_si256((__m256i*)(tmp), val);
//    for (size_t i = 0; i < sz; ++i) {
//        std::cout << (int)tmp[i] << " ";
//    }
//    std::cout << std::endl;
//}
//
//template <typename T>
//void print128(__m128i val, const T *tmp, size_t sz) {
//    _mm_store_si128((__m128i*)(tmp), val);
//    for (size_t i = 0; i < sz; ++i) {
//        std::cout << (int)tmp[i] << " ";
//    }
//    std::cout << std::endl;
//}

int main() {
    const size_t sz = 32;
    int16_t *acc = (int16_t*)_mm_malloc(sizeof(int16_t) * 1 * sz, 64);
    int8_t  *in1 = (int8_t *)_mm_malloc(sizeof(int8_t ) * 2 * sz, 64);
    uint8_t *in2 = (uint8_t*)_mm_malloc(sizeof(uint8_t) * 2,      64);

    int16_t *tmp_16i = (int16_t*)_mm_malloc(sizeof(int16_t) * 1 * sz, 64);
    int8_t  *tmp_8i  = (int8_t *)_mm_malloc(sizeof(int8_t ) * 2 * sz, 64);
    uint8_t *tmp_8u  = (uint8_t*)_mm_malloc(sizeof(uint8_t) * 2 * sz, 64);

    std::memset(acc, 0, sizeof(int16_t) * 1 * sz);
    std::memset(in1, 1, sizeof(int8_t)  * 2 * sz);

    in2[0] = 1;
    in2[1] = 3;

    __m256i c = _mm256_setzero_si256();
    __m256i d = _mm256_setzero_si256();
    //print(c, tmp_16i, sz / 2);

    __m256i l = _mm256_loadu_si256((__m256i*)(in1));
    __m256i r = _mm256_loadu_si256((__m256i*)(in1 + sz));
    //print(l, tmp_8i, sz);
    //print(r, tmp_8i, sz);

    __m256i hi = _mm256_unpackhi_epi8(l, r);
    __m256i lo = _mm256_unpacklo_epi8(l, r);
    //print(hi, tmp_8i, sz);
    //print(lo, tmp_8i, sz);

    //std::cout << std::endl;

    __m256i tmp_lo = _mm256_set1_epi8(in2[0]);
    __m256i tmp_hi = _mm256_set1_epi8(in2[1]);
    //print(tmp_lo, tmp_8u, sz);
    //print(tmp_hi, tmp_8u, sz);
    //std::cout << std::endl;

    __m256i b = _mm256_unpackhi_epi8(tmp_lo, tmp_hi);
    //__m256i lo_b = _mm256_unpacklo_epi8(tmp_lo, tmp_hi);
    //print(hi_b, tmp_8u, sz);
    //print(lo_b, tmp_8u, sz);
    //std::cout << std::endl;

    __m256i res_hi = _mm256_maddubs_epi16(b, hi);
    __m256i res_lo = _mm256_maddubs_epi16(b, lo);

    //print(res_hi, tmp_16i, sz / 2);
    //print(res_lo, tmp_16i, sz / 2);

    c = _mm256_adds_epi16(c, res_hi);
    d = _mm256_adds_epi16(d, res_lo);

    //print(c, tmp_16i, sz / 2);
    //print(d, tmp_16i, sz / 2);

    _mm256_store_si256((__m256i*)(acc), c);
    //_mm256_store_si256((__m256i*)(acc), c);
}

