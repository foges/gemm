#pragma once

#include <stdint.h>
#include <time.h>

#ifdef _WIN32
    #include <windows.h>
#elif __APPLE__
    #include <mach/mach_time.h>
#endif

inline uint64_t timestamp_ns() {

#ifdef _WIN32
    uint64_t cycles = 0;
    uint64_t frequency = 0;

    QueryPerformanceFrequency((LARGE_INTEGER*) &frequency);
    QueryPerformanceCounter((LARGE_INTEGER*) &cycles);

    return cycles / frequency;

#elif __APPLE__
    uint64_t absolute_time = mach_absolute_time();
    mach_timebase_info_data_t info = {0,0};

    if (info.denom == 0) mach_timebase_info(&info);
    uint64_t elapsednano = absolute_time * (info.numer / info.denom);

    timespec spec;
    spec.tv_sec  = elapsednano * 1e-9;
    spec.tv_nsec = elapsednano - (spec.tv_sec * 1e9);

    return spec.tv_nsec + (uint64_t)spec.tv_sec * 1e9;
#else
    timespec spec;

    clock_gettime(CLOCK_REALTIME, &spec);

    return spec.tv_nsec + (uint64_t)spec.tv_sec * 1e9;
#endif
}

inline uint64_t timestamp_us() {
    return timestamp_ns() / 1000u;
}

inline uint64_t timestamp_ms() {
    return timestamp_ns() / 1000u / 1000u;
}

inline uint64_t timestamp_s() {
    return timestamp_ns() / 1000u / 1000u / 1000u;
}

constexpr inline uint64_t us_to_ms(uint64_t us) {
    return us / 1000u;
}

constexpr inline uint64_t ms_to_s(uint64_t ms) {
    return ms / 1000u;
}

constexpr inline uint64_t us_to_s(uint64_t us) {
    return ms_to_s(us_to_ms(us));
}

constexpr inline uint64_t s_to_ms(uint64_t s) {
    return s * 1000u;
}

constexpr inline uint64_t ms_to_us(uint64_t ms) {
    return ms * 1000u;
}

constexpr inline uint64_t s_to_us(uint64_t s) {
    return ms_to_us(s_to_ms(s));
}

