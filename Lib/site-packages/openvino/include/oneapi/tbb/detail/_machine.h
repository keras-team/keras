/*
    Copyright (c) 2005-2024 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_detail__machine_H
#define __TBB_detail__machine_H

#include "_config.h"
#include "_assert.h"

#include <atomic>
#include <climits>
#include <cstdint>
#include <cstddef>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(__rdtsc)
#endif
#if __TBB_x86_64 || __TBB_x86_32
#include <immintrin.h> // _mm_pause
#endif
#if (_WIN32 || _WIN64)
#include <float.h> // _control87
#endif

#if __TBB_GLIBCXX_THIS_THREAD_YIELD_BROKEN
#include <sched.h> // sched_yield
#else
#include <thread> // std::this_thread::yield()
#endif

namespace tbb {
namespace detail {
inline namespace d0 {

//--------------------------------------------------------------------------------------------------
// Yield implementation
//--------------------------------------------------------------------------------------------------

#if __TBB_GLIBCXX_THIS_THREAD_YIELD_BROKEN
static inline void yield() {
    int err = sched_yield();
    __TBB_ASSERT_EX(err == 0, "sched_yiled has failed");
}
#else
using std::this_thread::yield;
#endif

//--------------------------------------------------------------------------------------------------
// atomic_fence implementation
//--------------------------------------------------------------------------------------------------

#if (_WIN32 || _WIN64)
#pragma intrinsic(_mm_mfence)
#endif

static inline void atomic_fence(std::memory_order order) {
#if (_WIN32 || _WIN64)
    if (order == std::memory_order_seq_cst ||
        order == std::memory_order_acq_rel ||
        order == std::memory_order_acquire ||
        order == std::memory_order_release )
    {
        _mm_mfence();
        return;
    }
#endif /*(_WIN32 || _WIN64)*/
    std::atomic_thread_fence(order);
}

//--------------------------------------------------------------------------------------------------
// Pause implementation
//--------------------------------------------------------------------------------------------------

static inline void machine_pause(int32_t delay) {
#if __TBB_x86_64 || __TBB_x86_32
    while (delay-- > 0) { _mm_pause(); }
#elif __ARM_ARCH_7A__ || __aarch64__
    while (delay-- > 0) { __asm__ __volatile__("isb sy" ::: "memory"); }
#else /* Generic */
    (void)delay; // suppress without including _template_helpers.h
    yield();
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// tbb::detail::log2() implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: Use log2p1() function that will be available in C++20 standard

#if defined(__GNUC__) || defined(__clang__)
namespace gnu_builtins {
    inline uintptr_t clz(unsigned int x) { return __builtin_clz(x); }
    inline uintptr_t clz(unsigned long int x) { return __builtin_clzl(x); }
    inline uintptr_t clz(unsigned long long int x) { return __builtin_clzll(x); }
}
#elif defined(_MSC_VER)
#pragma intrinsic(__TBB_W(_BitScanReverse))
namespace msvc_intrinsics {
    static inline uintptr_t bit_scan_reverse(uintptr_t i) {
        unsigned long j;
        __TBB_W(_BitScanReverse)( &j, i );
        return j;
    }
}
#endif

template <typename T>
constexpr std::uintptr_t number_of_bits() {
    return sizeof(T) * CHAR_BIT;
}

// logarithm is the index of the most significant non-zero bit
static inline uintptr_t machine_log2(uintptr_t x) {
#if defined(__GNUC__) || defined(__clang__)
    // If P is a power of 2 and x<P, then (P-1)-x == (P-1) XOR x
    return (number_of_bits<decltype(x)>() - 1) ^ gnu_builtins::clz(x);
#elif defined(_MSC_VER)
    return msvc_intrinsics::bit_scan_reverse(x);
#elif __i386__ || __i386 /*for Sun OS*/ || __MINGW32__
    uintptr_t j, i = x;
    __asm__("bsr %1,%0" : "=r"(j) : "r"(i));
    return j;
#elif __powerpc__ || __POWERPC__
    #if __TBB_WORDSIZE==8
    __asm__ __volatile__ ("cntlzd %0,%0" : "+r"(x));
    return 63 - static_cast<intptr_t>(x);
    #else
    __asm__ __volatile__ ("cntlzw %0,%0" : "+r"(x));
    return 31 - static_cast<intptr_t>(x);
    #endif /*__TBB_WORDSIZE*/
#elif __sparc
    uint64_t count;
    // one hot encode
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    x |= (x >> 32);
    // count 1's
    __asm__ ("popc %1, %0" : "=r"(count) : "r"(x) );
    return count - 1;
#else
    intptr_t result = 0;

    if( sizeof(x) > 4 && (uintptr_t tmp = x >> 32) ) { x = tmp; result += 32; }
    if( uintptr_t tmp = x >> 16 ) { x = tmp; result += 16; }
    if( uintptr_t tmp = x >> 8 )  { x = tmp; result += 8; }
    if( uintptr_t tmp = x >> 4 )  { x = tmp; result += 4; }
    if( uintptr_t tmp = x >> 2 )  { x = tmp; result += 2; }

    return (x & 2) ? result + 1 : result;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// tbb::detail::reverse_bits() implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
#if TBB_USE_CLANG_BITREVERSE_BUILTINS
namespace  llvm_builtins {
    inline uint8_t  builtin_bitreverse(uint8_t  x) { return __builtin_bitreverse8 (x); }
    inline uint16_t builtin_bitreverse(uint16_t x) { return __builtin_bitreverse16(x); }
    inline uint32_t builtin_bitreverse(uint32_t x) { return __builtin_bitreverse32(x); }
    inline uint64_t builtin_bitreverse(uint64_t x) { return __builtin_bitreverse64(x); }
}
#else // generic
template<typename T>
struct reverse {
    static const T byte_table[256];
};

template<typename T>
const T reverse<T>::byte_table[256] = {
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
    0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
    0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
    0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
    0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
    0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
    0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
    0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
    0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
    0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
    0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
    0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
    0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
    0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
    0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
    0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};

inline unsigned char reverse_byte(unsigned char src) {
    return reverse<unsigned char>::byte_table[src];
}
#endif // TBB_USE_CLANG_BITREVERSE_BUILTINS

template<typename T>
T machine_reverse_bits(T src) {
#if TBB_USE_CLANG_BITREVERSE_BUILTINS
    return builtin_bitreverse(fixed_width_cast(src));
#else /* Generic */
    T dst;
    unsigned char *original = (unsigned char *) &src;
    unsigned char *reversed = (unsigned char *) &dst;

    for ( int i = sizeof(T) - 1; i >= 0; i-- ) {
        reversed[i] = reverse_byte( original[sizeof(T) - i - 1] );
    }

    return dst;
#endif // TBB_USE_CLANG_BITREVERSE_BUILTINS
}

} // inline namespace d0

namespace d1 {

#if (_WIN32 || _WIN64)
// API to retrieve/update FPU control setting
#define __TBB_CPU_CTL_ENV_PRESENT 1
struct cpu_ctl_env {
    unsigned int x87cw{};
#if (__TBB_x86_64)
    // Changing the infinity mode or the floating-point precision is not supported on x64.
    // The attempt causes an assertion. See
    // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/control87-controlfp-control87-2
    static constexpr unsigned int X87CW_CONTROL_MASK = _MCW_DN | _MCW_EM | _MCW_RC;
#else
    static constexpr unsigned int X87CW_CONTROL_MASK = ~0U;
#endif
#if (__TBB_x86_32 || __TBB_x86_64)
    unsigned int mxcsr{};
    static constexpr unsigned int MXCSR_CONTROL_MASK = ~0x3fu; /* all except last six status bits */
#endif

    bool operator!=( const cpu_ctl_env& ctl ) const {
        return
#if (__TBB_x86_32 || __TBB_x86_64)
            mxcsr != ctl.mxcsr ||
#endif
            x87cw != ctl.x87cw;
    }
    void get_env() {
        x87cw = _control87(0, 0);
#if (__TBB_x86_32 || __TBB_x86_64)
        mxcsr = _mm_getcsr();
#endif
    }
    void set_env() const {
        _control87(x87cw, X87CW_CONTROL_MASK);
#if (__TBB_x86_32 || __TBB_x86_64)
        _mm_setcsr(mxcsr & MXCSR_CONTROL_MASK);
#endif
    }
};
#elif (__TBB_x86_32 || __TBB_x86_64)
// API to retrieve/update FPU control setting
#define __TBB_CPU_CTL_ENV_PRESENT 1
struct cpu_ctl_env {
    int     mxcsr{};
    short   x87cw{};
    static const int MXCSR_CONTROL_MASK = ~0x3f; /* all except last six status bits */

    bool operator!=(const cpu_ctl_env& ctl) const {
        return mxcsr != ctl.mxcsr || x87cw != ctl.x87cw;
    }
    void get_env() {
        __asm__ __volatile__(
            "stmxcsr %0\n\t"
            "fstcw %1"
            : "=m"(mxcsr), "=m"(x87cw)
        );
        mxcsr &= MXCSR_CONTROL_MASK;
    }
    void set_env() const {
        __asm__ __volatile__(
            "ldmxcsr %0\n\t"
            "fldcw %1"
            : : "m"(mxcsr), "m"(x87cw)
        );
    }
};
#endif

} // namespace d1

} // namespace detail
} // namespace tbb

#if !__TBB_CPU_CTL_ENV_PRESENT
#include <fenv.h>

#include <cstring>

namespace tbb {
namespace detail {

namespace r1 {
void* __TBB_EXPORTED_FUNC cache_aligned_allocate(std::size_t size);
void __TBB_EXPORTED_FUNC cache_aligned_deallocate(void* p);
} // namespace r1

namespace d1 {

class cpu_ctl_env {
    fenv_t *my_fenv_ptr;
public:
    cpu_ctl_env() : my_fenv_ptr(NULL) {}
    ~cpu_ctl_env() {
        if ( my_fenv_ptr )
            r1::cache_aligned_deallocate( (void*)my_fenv_ptr );
    }
    // It is possible not to copy memory but just to copy pointers but the following issues should be addressed:
    //   1. The arena lifetime and the context lifetime are independent;
    //   2. The user is allowed to recapture different FPU settings to context so 'current FPU settings' inside
    //   dispatch loop may become invalid.
    // But do we really want to improve the fenv implementation? It seems to be better to replace the fenv implementation
    // with a platform specific implementation.
    cpu_ctl_env( const cpu_ctl_env &src ) : my_fenv_ptr(NULL) {
        *this = src;
    }
    cpu_ctl_env& operator=( const cpu_ctl_env &src ) {
        __TBB_ASSERT( src.my_fenv_ptr, NULL );
        if ( !my_fenv_ptr )
            my_fenv_ptr = (fenv_t*)r1::cache_aligned_allocate(sizeof(fenv_t));
        *my_fenv_ptr = *src.my_fenv_ptr;
        return *this;
    }
    bool operator!=( const cpu_ctl_env &ctl ) const {
        __TBB_ASSERT( my_fenv_ptr, "cpu_ctl_env is not initialized." );
        __TBB_ASSERT( ctl.my_fenv_ptr, "cpu_ctl_env is not initialized." );
        return std::memcmp( (void*)my_fenv_ptr, (void*)ctl.my_fenv_ptr, sizeof(fenv_t) );
    }
    void get_env () {
        if ( !my_fenv_ptr )
            my_fenv_ptr = (fenv_t*)r1::cache_aligned_allocate(sizeof(fenv_t));
        fegetenv( my_fenv_ptr );
    }
    const cpu_ctl_env& set_env () const {
        __TBB_ASSERT( my_fenv_ptr, "cpu_ctl_env is not initialized." );
        fesetenv( my_fenv_ptr );
        return *this;
    }
};

} // namespace d1
} // namespace detail
} // namespace tbb

#endif /* !__TBB_CPU_CTL_ENV_PRESENT */

#endif // __TBB_detail__machine_H
