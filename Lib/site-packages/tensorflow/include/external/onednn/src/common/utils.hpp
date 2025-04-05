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

#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#include <atomic>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <memory>
#include <string>
#include <tuple>

#define MSAN_ENABLED 0
#define ATTR_NO_MSAN
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#undef MSAN_ENABLED
#define MSAN_ENABLED 1
#undef ATTR_NO_MSAN
#define ATTR_NO_MSAN __attribute__((no_sanitize("memory")))
#include <sanitizer/msan_interface.h>
#endif
#endif

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "z_magic.hpp"

namespace dnnl {
namespace impl {

#define DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other) \
    do { \
        if (this == &other) return *this; \
    } while (0)

#define DNNL_SHORT_CIRCUIT_SELF_COMPARISON(other) \
    do { \
        if (this == &other) return true; \
    } while (0)

#define DNNL_DISALLOW_COPY_AND_ASSIGN(T) \
    T(const T &) = delete; \
    T &operator=(const T &) = delete;

// Sanity check for 64 bits
static_assert(sizeof(void *) == 8, "oneDNN supports 64-bit architectures only");

#define CHECK(f) \
    do { \
        status_t _status_ = f; \
        if (_status_ != status::success) return _status_; \
    } while (0)

#define CHECK_BOOL(f) \
    do { \
        status_t _status_ = f; \
        if (_status_ != status::success) return false; \
    } while (0)

#define UNUSED_STATUS(f) \
    do { \
        status_t _status_ = f; \
        assert(_status_ == status::success); \
        MAYBE_UNUSED(_status_); \
    } while (0)

#define IMPLICATION(cause, effect) (!(cause) || !!(effect))

namespace utils {

/* a bunch of std:: analogues to be compliant with any msvs version
 *
 * Rationale: msvs c++ (and even some c) headers contain special pragma that
 * injects msvs-version check into object files in order to abi-mismatches
 * during the static linking. This makes sense if e.g. std:: objects are passed
 * through between application and library, which is not the case for oneDNN
 * (since there is no any c++-rt dependent stuff, ideally...). */

/* SFINAE helper -- analogue to std::enable_if */
template <bool expr, class T = void>
struct enable_if {};
template <class T>
struct enable_if<true, T> {
    typedef T type;
};

/* analogue std::conditional */
template <bool, typename, typename>
struct conditional {};
template <typename T, typename F>
struct conditional<true, T, F> {
    typedef T type;
};
template <typename T, typename F>
struct conditional<false, T, F> {
    typedef F type;
};

template <bool, typename, bool, typename, typename>
struct conditional3 {};
template <typename T, typename FT, typename FF>
struct conditional3<true, T, false, FT, FF> {
    typedef T type;
};
template <typename T, typename FT, typename FF>
struct conditional3<false, T, true, FT, FF> {
    typedef FT type;
};
template <typename T, typename FT, typename FF>
struct conditional3<false, T, false, FT, FF> {
    typedef FF type;
};

template <bool, typename U, U, U>
struct conditional_v {};
template <typename U, U t, U f>
struct conditional_v<true, U, t, f> {
    static constexpr U value = t;
};
template <typename U, U t, U f>
struct conditional_v<false, U, t, f> {
    static constexpr U value = f;
};

template <typename T>
struct remove_reference {
    typedef T type;
};
template <typename T>
struct remove_reference<T &> {
    typedef T type;
};
template <typename T>
struct remove_reference<T &&> {
    typedef T type;
};

template <typename T>
inline T &&forward(typename utils::remove_reference<T>::type &t) {
    return static_cast<T &&>(t);
}
template <typename T>
inline T &&forward(typename utils::remove_reference<T>::type &&t) {
    return static_cast<T &&>(t);
}

template <typename T>
inline typename remove_reference<T>::type zero() {
    auto zero = typename remove_reference<T>::type();
    return zero;
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T, typename P>
constexpr bool everyone_is(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
constexpr bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

template <typename T, typename P>
constexpr bool one_of(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename T, typename P>
constexpr P map(T pat, P def) {
    return def;
}
template <typename T, typename P, typename... Args>
constexpr P map(T pat, P def, T item, P ival, Args... item_others) {
    return pat == item ? ival : map(pat, def, item_others...);
}

template <typename... Args>
constexpr bool any_null(Args... ptrs) {
    return one_of(nullptr, ptrs...);
}

template <typename T>
inline void array_copy(T *dst, const T *src, size_t size) {
    for (size_t i = 0; i < size; ++i)
        dst[i] = src[i];
}
template <typename T>
inline bool array_cmp(const T *a1, const T *a2, size_t size) {
    for (size_t i = 0; i < size; ++i)
        if (a1[i] != a2[i]) return false;
    return true;
}
template <typename T, typename U>
inline void array_set(T *arr, const U &val, size_t size) {
    for (size_t i = 0; i < size; ++i)
        arr[i] = static_cast<T>(val);
}

namespace product_impl {
template <size_t>
struct int2type {};

template <typename T>
constexpr int product_impl(const T *arr, int2type<0>) {
    return arr[0];
}

template <typename T, size_t num>
constexpr T product_impl(const T *arr, int2type<num>) {
    return arr[0] * product_impl(arr + 1, int2type<num - 1>());
}
} // namespace product_impl

template <size_t num, typename T>
constexpr T array_product(const T *arr) {
    return product_impl::product_impl(arr, product_impl::int2type<num - 1>());
}

template <typename T, typename R = T>
inline R array_product(const T *arr, size_t size) {
    R prod = 1;
    for (size_t i = 0; i < size; ++i)
        prod *= arr[i];
    return prod;
}

template <typename T, typename R = T>
inline R array_product(const std::vector<T> &v) {
    return array_product<T, R>(v.data(), v.size());
}

template <typename T, typename R = T>
inline R array_min(const T *arr, size_t size) {
    R min = std::numeric_limits<R>::max();
    for (size_t i = 0; i < size; ++i)
        min = std::min(min, arr[i]);
    return min;
}

inline bool equal_with_nan(float v1, float v2) {
    return (v1 == v2) || (std::isnan(v1) && std::isnan(v2));
}

/* Sorts an array of @p vals using @p comparator. Uses @p vals_2nd_level as a
 * second level comparing criteria in case comparator returns 0 (equal values)
 * for @p vals elements.
 * While sorting the array of @p vals, the function permutes an array of
 * @p vals_2nd_level and @p keys accordingly.
 */
template <typename T, typename U, typename F>
inline void simultaneous_sort(
        T *vals, T *vals_2nd_level, U *keys, size_t size, F comparator) {
    if (size == 0) return;

    for (size_t i = 0; i < size - 1; ++i) {
        bool swapped = false;

        for (size_t j = 0; j < size - i - 1; j++) {
            auto res = comparator(vals[j], vals[j + 1]);
            if (res == 0)
                res = comparator(vals_2nd_level[j], vals_2nd_level[j + 1]);

            if (res > 0) {
                nstl::swap(vals[j], vals[j + 1]);
                nstl::swap(vals_2nd_level[j], vals_2nd_level[j + 1]);
                nstl::swap(keys[j], keys[j + 1]);
                swapped = true;
            }
        }

        if (swapped == false) break;
    }
}

template <typename T>
constexpr const T &saturate(const T &low, const T &upper, const T &a) {
    return nstl::max(low, nstl::min(upper, a));
}

template <typename T, typename U>
inline typename remove_reference<T>::type div_up(const T a, const U b) {
    assert(b);
    return static_cast<typename remove_reference<T>::type>((a + b - 1) / b);
}

template <typename T, typename U>
inline typename remove_reference<T>::type rnd_up(const T a, const U b) {
    return static_cast<typename remove_reference<T>::type>(div_up(a, b) * b);
}

template <typename T, typename U>
constexpr typename remove_reference<T>::type rnd_dn(const T a, const U b) {
    return static_cast<typename remove_reference<T>::type>((a / b) * b);
}

template <typename T>
inline typename remove_reference<T>::type rnd_up_pow2(const T a) {
    using R = typename remove_reference<T>::type;
    if (a <= 0)
        return static_cast<R>(1);
    else {
        T b = a - 1;
        for (size_t v = 1; v < sizeof(T) * CHAR_BIT; v <<= 1)
            b |= (b >> v);
        return static_cast<R>(b + 1);
    }
}

template <typename T>
inline typename remove_reference<T>::type rnd_down_pow2(const T a) {
    auto ret = rnd_up_pow2(a);
    return ret == a ? ret : ret / 2;
}

template <typename T, typename U>
inline typename remove_reference<T>::type max_div(const T a, const U b) {
    U div = b;
    while (div > 1) {
        if (a % div == 0) return div;
        div--;
    }
    return static_cast<typename remove_reference<T>::type>(div);
}

template <typename T>
inline typename remove_reference<T>::type max_pow2_div(const T a) {
    return static_cast<typename remove_reference<T>::type>(((a - 1) & ~a) + 1);
}

template <typename T>
T *align_ptr(T *ptr, uintptr_t alignment) {
    return (T *)(((uintptr_t)ptr + alignment - 1) & ~(alignment - 1));
}

template <typename T, typename U, typename V>
inline typename remove_reference<U>::type this_block_size(
        const T offset, const U max, const V block_size) {
    assert(offset < max);
    // TODO (Roma): can't use nstl::max() due to circular dependency... we
    // need to fix this
    const T block_boundary = offset + block_size;
    if (block_boundary > max)
        return max - offset;
    else
        return block_size;
}

template <typename T>
inline T nd_iterator_init(T start) {
    return start;
}
template <typename T, typename U, typename W, typename... Args>
inline T nd_iterator_init(T start, U &x, const W &X, Args &&...tuple) {
    start = nd_iterator_init(start, utils::forward<Args>(tuple)...);
    x = start % X;
    return start / X;
}

inline bool nd_iterator_step() {
    return true;
}
template <typename U, typename W, typename... Args>
inline bool nd_iterator_step(U &x, const W &X, Args &&...tuple) {
    if (nd_iterator_step(utils::forward<Args>(tuple)...)) {
        if (++x - X == 0) {
            x = 0;
            return true;
        }
    }
    return false;
}

template <typename U, typename W, typename Y>
inline bool nd_iterator_jump(U &cur, const U end, W &x, const Y &X) {
    U max_jump = end - cur;
    U dim_jump = X - x;
    if (dim_jump <= max_jump) {
        x = 0;
        cur += dim_jump;
        return true;
    } else {
        cur += max_jump;
        x += max_jump;
        return false;
    }
}
template <typename U, typename W, typename Y, typename... Args>
inline bool nd_iterator_jump(
        U &cur, const U end, W &x, const Y &X, Args &&...tuple) {
    if (nd_iterator_jump(cur, end, utils::forward<Args>(tuple)...)) {
        if (++x - X == 0) {
            x = 0;
            return true;
        }
    }
    return false;
}

template <typename T>
constexpr T pick(size_t i, const T &x0) {
    return x0;
}
template <typename T, typename... Args>
constexpr T pick(size_t i, const T &x0, Args &&...args) {
    return i == 0 ? x0 : pick(i - 1, utils::forward<Args>(args)...);
}

template <typename T>
T pick_by_prop_kind(prop_kind_t prop_kind, const T &val_fwd_inference,
        const T &val_fwd_training, const T &val_bwd_d, const T &val_bwd_w) {
    switch (prop_kind) {
        case prop_kind::forward_inference: return val_fwd_inference;
        case prop_kind::forward_training: return val_fwd_training;
        case prop_kind::backward_data: return val_bwd_d;
        case prop_kind::backward_weights: return val_bwd_w;
        default: assert(!"unsupported prop_kind");
    }
    return T();
}

template <typename T>
T pick_by_prop_kind(prop_kind_t prop_kind, const T &val_fwd, const T &val_bwd_d,
        const T &val_bwd_w) {
    return pick_by_prop_kind(prop_kind, val_fwd, val_fwd, val_bwd_d, val_bwd_w);
}

template <typename Telem, size_t Tdims>
struct array_offset_calculator {
    template <typename... Targs>
    array_offset_calculator(Telem *base, Targs... Fargs) : _dims {Fargs...} {
        _base_ptr = base;
    }

    template <typename... Targs>
    array_offset_calculator(std::nullptr_t, Targs... Fargs) = delete;

    template <typename... Targs>
    inline Telem &operator()(Targs... Fargs) const {
        assert(static_cast<bool>(_base_ptr));
        return *(_base_ptr + _offset(1, Fargs...));
    }

private:
    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t element) const {
        return element;
    }

    template <typename... Targs>
    inline size_t _offset(
            size_t const dimension, size_t theta, size_t element) const {
        return element + (_dims[dimension] * theta);
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element,
            Targs... Fargs) const {
        size_t t_prime = element + (_dims[dimension] * theta);
        return _offset(dimension + 1, t_prime, Fargs...);
    }

    Telem *_base_ptr;
    const dim_t _dims[Tdims];
};

template <typename derived_type, typename base_type>
inline derived_type downcast(base_type *base) {
    assert(dynamic_cast<derived_type>(base) == base);
    return static_cast<derived_type>(base);
}

template <typename T,
        typename std::enable_if<!std::is_same<typename std::decay<T>::type,
                std::string>::value>::type * = nullptr>
auto format_cvt_impl(T &&t) -> decltype(std::forward<T>(t)) {
    return std::forward<T>(t);
}

template <typename T,
        typename std::enable_if<std::is_same<typename std::decay<T>::type,
                std::string>::value>::type * = nullptr>
const char *format_cvt_impl(T &&t) {
    return std::forward<T>(t).c_str();
}

template <typename... Args>
std::string format_impl(const char *fmt, Args... args) {
    size_t sz = snprintf(nullptr, 0, fmt, args...);
    std::string buf(sz + 1, '\0');
    snprintf(&buf[0], sz + 1, fmt, args...);
    buf.resize(sz);
    return buf;
}

template <typename... Args>
std::string format(const char *fmt, Args &&...args) {
    return format_impl(fmt, format_cvt_impl(std::forward<Args>(args))...);
}

inline bool need_src_or_dst_check(
        bool is_fwd, int o, int i, int k, int p, int s, int d) {
    if (is_fwd) {
        int i_min = -p;
        int i_max = (o - 1) * s - p + (k - 1) * (1 + d);
        return (i_min < 0) || (i_max >= i);
    }
    // Backward.
    int os_min = p - (k - 1) * (1 + d);
    int os_max = (i - 1) + p;
    return (os_min < 0) || (os_max >= o * s);
}

// transforms @param l(ogical)_offset into a @param dims_pos based on input
// dimensions @param dims and @param ndims.
inline void l_dims_by_l_offset(
        dims_t dims_pos, dim_t l_offset, const dims_t dims, int ndims) {
    for (int rd = 0; rd < ndims; ++rd) {
        const int d = ndims - 1 - rd;
        /* switch to faster 32-bit division when possible. */
        if (l_offset <= INT32_MAX && dims[d] <= INT32_MAX) {
            dims_pos[d] = (int32_t)l_offset % (int32_t)dims[d];
            l_offset = (int32_t)l_offset / (int32_t)dims[d];
        } else {
            dims_pos[d] = l_offset % dims[d];
            l_offset /= dims[d];
        }
    }
}

inline int get_dims_mask(const dims_t dims1, const dims_t dims2, int ndims,
        bool skip_dim_of_one = false) {
    int mask = 0;
    for (int d = 0; d < ndims; ++d) {
        // Disable mask_bit for dimensions of `1` by request.
        int mask_bit = skip_dim_of_one && dims1[d] == 1 ? 0 : (1 << d);
        mask += dims1[d] == dims2[d] ? mask_bit : 0;
    }
    return mask;
};

inline void copy_dims_with_mask(
        dims_t ddims, const dims_t sdims, int ndims, int mask) {
    for (int d = 0; d < ndims; ++d) {
        ddims[d] = (mask & (1 << d)) ? sdims[d] : 0;
    }
}

inline void apply_mask_on_dims(dims_t dims, int ndims, int mask) {
    copy_dims_with_mask(dims, dims, ndims, mask);
}

inline void dim_iterator(const dims_t dims, dims_t indices, int ndims) {
    while (--ndims >= 0 && ++indices[ndims] >= dims[ndims]) {
        indices[ndims] = 0;
    }
}

template <typename T, size_t S>
inline size_t array_size(T (&t)[S]) {
    return S;
}

inline bool validate_dims(int ndims, const dims_t dims) {
    for (int d = 0; d < ndims; ++d)
        if (dims[d] <= 0) return false;
    return true;
}

} // namespace utils

int32_t fetch_and_add(int32_t *dst, int32_t val);
inline void yield_thread() {}
bool is_destroying_cache_safe();

// Reads an environment variable 'name' and stores its string value in the
// 'buffer' of 'buffer_size' bytes (including the terminating zero) on
// success.
//
// - Returns the length of the environment variable string value (excluding
// the terminating 0) if it is set and its contents (including the terminating
// 0) can be stored in the 'buffer' without truncation.
//
// - Returns negated length of environment variable string value and writes
// "\0" to the buffer (if it is not NULL) if the 'buffer_size' is to small to
// store the value (including the terminating 0) without truncation.
//
// - Returns 0 and writes "\0" to the buffer (if not NULL) if the environment
// variable is not set.
//
// - Returns INT_MIN if the 'name' is NULL.
//
// - Returns INT_MIN if the 'buffer_size' is negative.
//
// - Returns INT_MIN if the 'buffer' is NULL and 'buffer_size' is greater than
// zero. Passing NULL 'buffer' with 'buffer_size' set to 0 can be used to
// retrieve the length of the environment variable value string.
//
int getenv(const char *name, char *buffer, int buffer_size);
// Reads an integer from the environment. For internal needs.
int getenv_int(const char *name, int default_value = 0);
// Reads an integer from user environment. Takes a var name without
// prefix and checks both supported variants - with "ONEDNN_" (primary) and
// "DNNL_" (secondary) prefixes.
int getenv_int_user(const char *name, int default_value = 0);
// Reads a string literal from user environment. Takes a var name without
// prefix and checks both supported variants - with "ONEDNN_" (primary) and
// "DNNL_" (secondary) prefixes.
std::string getenv_string_user(const char *name);

// Various getter for profiling info
bool get_jit_dump();
unsigned get_jit_profiling_flags();
std::string get_jit_profiling_jitdumpdir();
FILE *fopen(const char *filename, const char *mode);
int getpagesize();

// return current library fpmath_mode
fpmath_mode_t get_fpmath_mode();
// checks if an fpmath_mode is valid
status_t check_fpmath_mode(fpmath_mode_t mode);
// Returns true if values reprensented by type sub_dt can all be
// represented in dt. return false eotherwise
bool is_fpsubtype(data_type_t sub_dt, data_type_t dt);

constexpr int msan_enabled = MSAN_ENABLED;
inline void msan_unpoison(void *ptr, size_t size) {
#if MSAN_ENABLED
    __msan_unpoison(ptr, size);
#endif
}

// Helper to avoid #ifdefs for DNNL_DEV_MODE related code
static constexpr bool is_dev_mode() {
#ifdef DNNL_DEV_MODE
    return true;
#else
    return false;
#endif
}

// std::optional? std::maybe? std::whatever
template <typename T>
struct setting_t {
private:
    T value_;
    bool initialized_;

public:
    constexpr setting_t() : value_ {}, initialized_ {false} {}
    constexpr setting_t(const T init) : value_ {init}, initialized_ {false} {}
    bool initialized() { return initialized_; }
    T get() { return value_; }
    void set(T new_value) {
        value_ = new_value;
        initialized_ = true;
    }
    DNNL_DISALLOW_COPY_AND_ASSIGN(setting_t);
};

// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
template <typename T>
static size_t hash_combine(size_t seed, const T &v) {
    return seed ^= std::hash<T> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

inline int float2int(float x) {
    return utils::bit_cast<int>(x);
}

inline float int2float(int x) {
    return utils::bit_cast<float>(x);
}

// XXX: Currently SYCL doesn't provide an API to get device UUID but
// we need to be able to distinguish OpenCL device from Level0 device.
// As a temporary solution the compound ID will be used for that.
// Below is a table explaning what the numbers are for different backends:
//
// -------------------------------------------------------------
//  Backend      | Compound ID
// -------------------------------------------------------------
//  Host         | <backend_t::host, 0, 0>
//  OpenCL       | <backend_t::opencl, cl_device, 0>
//  NVIDIA       | <backend_t::nvidia, cuDevice, 0>
//  Level0       | <backend_t::level0, uuid[0-63], uuid[64-127]>
//  Pure CPU     | <0, 0, 0>
//  Pure GPU     | <0, cl_device, 0>
using device_id_t = std::tuple<int, uint64_t, uint64_t>;

struct device_id_hash_t {
    size_t operator()(const device_id_t &id) const {
        size_t result = 0;
        result = hash_combine(result, std::get<0>(id));
        result = hash_combine(result, std::get<1>(id));
        result = hash_combine(result, std::get<2>(id));
        return result;
    }
};

// A setting (basically a value) that can be set() multiple times until the
// time first time the get() method is called. The set() method is expected to
// be as expensive as a busy-waiting spinlock. The get() method is expected to
// be asymptotically as expensive as a single lock-prefixed memory read. The
// get() method also has a 'soft' mode when the setting is not locked for
// re-setting. This is used for testing purposes.
template <typename T>
struct set_once_before_first_get_setting_t {
private:
    T value_;
    std::atomic<unsigned> state_;
    enum : unsigned { idle = 0, busy_setting = 1, locked = 2 };

public:
    set_once_before_first_get_setting_t(T init)
        : value_ {init}, state_ {idle} {}

    bool set(T new_value) {
        if (state_.load() == locked) return false;

        while (true) {
            unsigned expected = idle;
            if (state_.compare_exchange_weak(expected, busy_setting)) break;
            if (expected == locked) return false;
        }

        value_ = new_value;
        state_.store(locked);
        return true;
    }

    T get(bool soft = false) {
        if (!soft && state_.load() != locked) {
            while (true) {
                unsigned expected = idle;
                if (state_.compare_exchange_weak(expected, locked)) break;
                if (expected == locked) break;
            }
        }
        return value_;
    }
};

inline bool is_native_runtime(runtime_kind_t kind) {
    return utils::one_of(kind, runtime_kind::seq, runtime_kind::omp,
            runtime_kind::tbb, runtime_kind::threadpool);
}

// Convenience wrapper to choose at compile-time between std::unique_ptr's
// default deleter and a no-op one.
//
// This is useful for static pointers to objects with non-trivial destructors.
// In some environments (e.g. tests where not all threads are joined at exit
// time) these destructors can result in sanitizer failures (e.g. races in
// thread sanitizer) when destructing unique_ptr's, but not with raw pointers.
// Of course in a shared library environment using raw pointers (that are
// therefore never freed) would result in memory leaks; this is why
// DNNL_MAYBE_UNIQUE_PTR_IS_UNIQUE defaults to 1.
#ifndef DNNL_MAYBE_UNIQUE_PTR_IS_UNIQUE
#define DNNL_MAYBE_UNIQUE_PTR_IS_UNIQUE 1
#endif

#if DNNL_MAYBE_UNIQUE_PTR_IS_UNIQUE
template <typename T>
using maybe_unique_ptr = std::unique_ptr<T>;
#else
struct nop_deleter_t {
    template <typename T>
    void operator()(T const &) const noexcept {}
};
template <typename T>
using maybe_unique_ptr = std::unique_ptr<T, nop_deleter_t>;
#endif // DNNL_MAYBE_UNIQUE_PTR_IS_UNIQUE

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
