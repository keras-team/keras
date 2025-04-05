/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_NSTL_HPP
#define COMMON_NSTL_HPP

#include <float.h>
#include <limits.h>
#include <stdint.h>

#include <cassert>
#include <cstdlib>
#include <map>
#include <vector>

#include "bfloat16.hpp"
#include "float16.hpp"
#include "float8.hpp"
#include "int4.hpp"
#include "internal_defs.hpp"
#include "z_magic.hpp"

namespace dnnl {
namespace impl {

#ifdef DNNL_ENABLE_MEM_DEBUG
// Export the malloc symbol (for SHARED build) or define it as weak (for STATIC
// build) in order to replace it with a custom one from an object file at link
// time.
void DNNL_WEAK *malloc(size_t size, int alignment);
// Use glibc malloc (and, consequently, free) when DNNL_ENABLE_MEM_DEBUG is
// enabled, such that the operator new doesn't fail during out_of_memory
// testing.
#define MALLOC(size, alignment) ::malloc(size)
#define FREE(ptr) ::free(ptr)
#else
void *malloc(size_t size, int alignment);
#define MALLOC(size, alignment) malloc(size, alignment)
#define FREE(ptr) free(ptr)
#endif
void free(void *p);

struct c_compatible {
    enum { default_alignment = 64 };
    static void *operator new(size_t sz) {
        return MALLOC(sz, default_alignment);
    }
    static void *operator new(size_t sz, void *p) {
        UNUSED(sz);
        return p;
    }
    static void *operator new[](size_t sz) {
        return MALLOC(sz, default_alignment);
    }
    static void operator delete(void *p) { FREE(p); }
    static void operator delete[](void *p) { FREE(p); }

protected:
    // This member is intended as a check for whether all necessary members in
    // derived classes have been allocated. Consequently status::out_of_memory
    // should be returned if a fail occurs. This member should be modified only
    // in a constructor.
    bool is_initialized_ = true;
};

#undef MALLOC
#undef FREE

namespace nstl {

template <typename T>
constexpr const T abs(const T &a) {
    return a >= 0 ? a : -a;
}

// Computes the modulus and returns the result as the least positive residue
// when the divisor > 0.
template <typename T>
inline const T modulo(const T &dividend, const T &divisor) {
    static_assert(std::is_integral<T>::value, "T must be an integer type.");
    assert(divisor > 0);
    T result = dividend % divisor;
    return result < 0 ? result + divisor : result;
}

// Computes the additive inverse modulus and returns the result as the least
// positive residue when the divisor > 0.
template <typename T>
inline const T additive_inverse_modulo(const T &dividend, const T &divisor) {
    static_assert(std::is_integral<T>::value, "T must be an integer type.");
    assert(divisor > 0);
    T result = modulo(dividend, divisor);
    return result > 0 ? divisor - result : 0;
}

template <typename T>
constexpr const T &max(const T &a, const T &b) {
    return a > b ? a : b;
}

template <typename T>
constexpr const T &min(const T &a, const T &b) {
    return a < b ? a : b;
}

template <typename T>
static inline constexpr T clamp(T val, T lo, T hi) {
    return std::min<T>(hi, std::max<T>(lo, val));
}

template <typename T>
void swap(T &t1, T &t2) {
    T tmp(t1);
    t1 = t2;
    t2 = tmp;
}

// Rationale: oneDNN needs numeric limits implementation that does not
// generate dependencies on C++ run-time libraries.

template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<float> : public std::numeric_limits<float> {};

template <>
struct numeric_limits<double> : public std::numeric_limits<double> {};

template <>
struct numeric_limits<int32_t> : public std::numeric_limits<int32_t> {};

template <>
struct numeric_limits<int16_t> : public std::numeric_limits<int16_t> {};

template <>
struct numeric_limits<uint16_t> : public std::numeric_limits<uint16_t> {};

template <>
struct numeric_limits<int8_t> : public std::numeric_limits<int8_t> {};

template <>
struct numeric_limits<uint8_t> : public std::numeric_limits<uint8_t> {};

template <>
struct numeric_limits<float8_e5m2_t> {
    static constexpr float8_e5m2_t lowest() {
        return float8_e5m2_t(0xfb, true);
    }

    static constexpr float8_e5m2_t max() { return float8_e5m2_t(0x7b, true); }

    static constexpr int bias = 0xf;
    static constexpr int digits = 3; // s1e5m2 -> 2+1 bits

    static constexpr float8_e5m2_t epsilon() {
        return float8_e5m2_t(((bias - (digits - 1)) << (digits - 1)), true);
    }
};

template <>
struct numeric_limits<float8_e4m3_t> {
    static constexpr float8_e4m3_t lowest() {
        return float8_e4m3_t(0xfe, true);
    }

    static constexpr float8_e4m3_t max() { return float8_e4m3_t(0x7e, true); }

    static constexpr int bias = 0x7;
    static constexpr int digits = 4; // s1e4m3 -> 3+1 bits

    static constexpr float8_e4m3_t epsilon() {
        return float8_e4m3_t(((bias - (digits - 1)) << (digits - 1)), true);
    }
};

template <>
struct numeric_limits<bfloat16_t> {
    static constexpr bfloat16_t lowest() { return bfloat16_t(0xff7f, true); }

    static constexpr bfloat16_t max() { return bfloat16_t(0x7f7f, true); }

    static constexpr int digits = 8;

    static constexpr bfloat16_t epsilon() {
        return bfloat16_t(((0x7f - (digits - 1)) << (digits - 1)), true);
    }
};

template <>
struct numeric_limits<float16_t> {
    static constexpr float16_t lowest() { return float16_t(0xfbff, true); }

    static constexpr float16_t max() { return float16_t(0x7bff, true); }

    static constexpr int digits = 11;

    static constexpr float16_t epsilon() {
        return float16_t(((0x0f - (digits - 1)) << (digits - 1)), true);
    }
};

template <>
struct numeric_limits<uint4_t> {
    static constexpr uint4_t lowest() { return uint4_t(0); }

    static constexpr uint4_t max() { return uint4_t(15); }

    static constexpr int digits = 4;

    static constexpr uint4_t epsilon() { return uint4_t(0); }
};

template <>
struct numeric_limits<int4_t> {
    static constexpr int4_t lowest() { return int4_t(-8); }

    static constexpr int4_t max() { return int4_t(7); }

    static constexpr int digits = 4;

    static constexpr int4_t epsilon() { return int4_t(0); }
};

template <typename T>
struct is_integral {
    static constexpr bool value = false;
};
template <>
struct is_integral<int32_t> {
    static constexpr bool value = true;
};
template <>
struct is_integral<int16_t> {
    static constexpr bool value = true;
};
template <>
struct is_integral<int8_t> {
    static constexpr bool value = true;
};
template <>
struct is_integral<uint8_t> {
    static constexpr bool value = true;
};
template <>
struct is_integral<int4_t> {
    static constexpr bool value = true;
};
template <>
struct is_integral<uint4_t> {
    static constexpr bool value = true;
};

template <typename T, typename U>
struct is_same {
    static constexpr bool value = false;
};
template <typename T>
struct is_same<T, T> {
    static constexpr bool value = true;
};

// Rationale: oneDNN needs container implementations that do not generate
// dependencies on C++ run-time libraries.
//
// Implementation philosophy: caller is responsible to check if the operation
// is valid. The only functions that have to return status are those that
// depend on memory allocation or similar operations.
//
// This means that e.g. an operator [] does not have to check for boundaries.
// The caller should have checked the boundaries. If it did not we crash and
// burn: this is a bug in oneDNN and throwing an exception would not have been
// recoverable.
//
// On the other hand, insert() or resize() or a similar operation needs to
// return a status because the outcome depends on factors external to the
// caller. The situation is probably also not recoverable also, but oneDNN
// needs to be nice and report "out of memory" to the users.

enum nstl_status_t { success = 0, out_of_memory };

template <typename T>
class vector : public c_compatible {
private:
    std::vector<T> _impl;

public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;
    typedef typename std::vector<T>::size_type size_type;
    vector() {}
    vector(size_type n) : _impl(n) {}
    vector(size_type n, const T &value) : _impl(n, value) {}
    template <typename input_iterator>
    vector(input_iterator first, input_iterator last) : _impl(first, last) {}
    ~vector() {}
    size_type size() const { return _impl.size(); }
    T &operator[](size_type i) { return _impl[i]; }
    const T &operator[](size_type i) const { return _impl[i]; }
    iterator begin() { return _impl.begin(); }
    const_iterator begin() const { return _impl.begin(); }
    iterator end() { return _impl.end(); }
    const_iterator end() const { return _impl.end(); }
    template <typename input_iterator>
    nstl_status_t insert(
            iterator pos, input_iterator begin, input_iterator end) {
        _impl.insert(pos, begin, end);
        return success;
    }
    void clear() { _impl.clear(); }
    void push_back(const T &t) { _impl.push_back(t); }
    void resize(size_type count) { _impl.resize(count); }
    void reserve(size_type count) { _impl.reserve(count); }
};

template <typename Key, typename T>
class map : public c_compatible {
private:
    std::map<Key, T> _impl;

public:
    typedef typename std::map<Key, T>::iterator iterator;
    typedef typename std::map<Key, T>::const_iterator const_iterator;
    typedef typename std::map<Key, T>::size_type size_type;
    map() {}
    ~map() {}
    size_type size() const { return _impl.size(); }
    T &operator[](const Key &k) { return _impl[k]; }
    const T &operator[](const Key &k) const { return _impl[k]; }
    iterator begin() { return _impl.begin(); }
    const_iterator begin() const { return _impl.begin(); }
    iterator end() { return _impl.end(); }
    const_iterator end() const { return _impl.end(); }
    template <typename input_iterator>
    void clear() {
        _impl.clear();
    }
};

// Compile-time sequence of indices (part of C++14)
template <size_t... Ints>
struct index_sequence {};

template <size_t N, size_t... Next>
struct make_index_sequence_helper
    : public make_index_sequence_helper<N - 1, N - 1, Next...> {};

template <size_t... Next>
struct make_index_sequence_helper<0, Next...> {
    using type = index_sequence<Next...>;
};

// Generator of compile-time sequence of indices
template <size_t N>
using make_index_sequence = typename make_index_sequence_helper<N>::type;
} // namespace nstl
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
