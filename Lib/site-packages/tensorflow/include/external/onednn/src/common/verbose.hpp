/*******************************************************************************
* Copyright 2018-2024 Intel Corporation
* Copyright 2023 Arm Ltd. and affiliates
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

#ifndef COMMON_VERBOSE_HPP
#define COMMON_VERBOSE_HPP

#include <cinttypes>
#include <mutex>
#include <stdio.h>

#include "c_types_map.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "utils.hpp"
#include "z_magic.hpp"

#include "profiler.hpp"
#include "verbose_msg.hpp"

namespace dnnl {
namespace impl {

// Trick to print filename only on all compilers supporting C++11
namespace utility {

template <typename T, size_t S>
inline constexpr size_t get_file_name_offset(
        const T (&str)[S], size_t i = S - 1) {
    // we match 'src/' from the right
    return (i + 3 < S) && str[i] == 's' && str[i + 1] == 'r'
                    && str[i + 2] == 'c'
                    && (str[i + 3] == '/' || str[i + 3] == '\\')
            ? i
            : (i > 3 ? get_file_name_offset(str, i - 1) : 0);
}

template <typename T>
inline constexpr size_t get_file_name_offset(T (&str)[1]) {
    return 0;
}
template <typename T, T v>
struct const_expr_value {
    static constexpr const T value = v;
};

} // namespace utility

#define UTILITY_CONST_EXPR_VALUE(exp) \
    utility::const_expr_value<decltype(exp), exp>::value

#define __FILENAME__ (&__FILE__[utility::get_file_name_offset(__FILE__)])

// General formatting macro for verbose.
// msg is typically a constant string pulled from verbose_msg.hpp
// The string can contain format specifiers which are provided in VA_ARGS
// Note: using ##__VAR_ARGS__ is necessary to avoid trailing comma in printf call

#define VFORMAT(stamp, apitype, logtype, logsubtype, msg, ...) \
    do { \
        std::string stamp_; \
        if (dnnl::impl::get_verbose_timestamp()) \
            stamp_ = "," + std::to_string(stamp); \
        printf("onednn_verbose%s," CONCAT2(VERBOSE_, apitype) "," CONCAT2( \
                       VERBOSE_, logtype) "%s," msg "\n", \
                stamp_.c_str(), logsubtype, ##__VA_ARGS__); \
    } while (0)

// Logging info
#define VINFO(apitype, logtype, logsubtype, component, msg, ...) \
    do { \
        if (dnnl::impl::get_verbose(verbose_t::logtype##_##logsubtype)) \
            VFORMAT(get_msec(), apitype, logtype, VERBOSE_##logsubtype, \
                    #component "," msg ",%s:%d", ##__VA_ARGS__, __FILENAME__, \
                    __LINE__); \
    } while (0)

// Macro for boolean checks
#define VCONDCHECK( \
        apitype, logtype, logsubtype, component, condition, status, msg, ...) \
    do { \
        if (!(condition)) { \
            VINFO(apitype, logtype, logsubtype, component, msg, \
                    ##__VA_ARGS__); \
            return status; \
        } \
    } while (0)

// Macro for status checks
#define VCHECK(apitype, logtype, logsubtype, component, f, msg, ...) \
    do { \
        status_t _status_ = (f); \
        VCONDCHECK(apitype, logtype, logsubtype, component, \
                _status_ == status::success, _status_, msg, ##__VA_ARGS__); \
    } while (0)

// Special syntactic sugar for error, plus flush of the output stream
#define VERROR(apitype, component, msg, ...) \
    do { \
        if (dnnl::impl::get_verbose(verbose_t::error)) { \
            VFORMAT(get_msec(), apitype, error, "", #component "," msg, \
                    ##__VA_ARGS__); \
            fflush(stdout); \
        } \
    } while (0)

// Special syntactic sugar for debuginfo prints.
// `level` is responsible to set the bar to be printed.
#define VDEBUGINFO(level, apitype, component, msg, ...) \
    do { \
        if (dnnl::impl::get_verbose_dev_mode(verbose_t::debuginfo) \
                >= (level)) { \
            VFORMAT(get_msec(), apitype, debuginfo, "", #component "," msg, \
                    ##__VA_ARGS__); \
            fflush(stdout); \
        } \
    } while (0)

// Special syntactic sugar for logging performance
// NOTE: the VPROF macro does not check for verbose flags, it is the
// responsibility of the caller do check those (it should happen
// anyway to condition collecting stamp/duration)
#define VPROF(stamp, apitype, logtype, logsubtype, info, duration) \
    { \
        VFORMAT(stamp, apitype, logtype, logsubtype, "%s,%g", info, duration); \
        fflush(stdout); \
    }

struct verbose_t {
    enum flag_kind : uint32_t {
        // we reserve the 24 lower bits for user info
        none = 0,
        // We reserve bits 0,1 to maintain backward compatibility support
        // of VERBOSE={1,2}
        error = 1 << 2,
        create_check = 1 << 3,
        create_dispatch = 1 << 4,
        create_profile = 1 << 5,
        exec_check = 1 << 6,
        exec_profile = 1 << 7,
        profile_externals = 1 << 8,
        // the upper 8 bits are reserved for devinfo levels
        debuginfo = 1 << 24,
        //
        all = (uint32_t)-1,
    };

    static uint32_t make_debuginfo(uint32_t level) { return level << 24; }
    static uint32_t get_debuginfo(uint32_t flag) { return flag >> 24; }
};

struct component_t {
    enum flag_kind : uint32_t {
        none = 0,
        primitive = 1 << 0,
        // keep the same order with dnnl_primitive_kind_t
        reorder = 1 << 1,
        shuffle = 1 << 2,
        concat = 1 << 3,
        sum = 1 << 4,
        convolution = 1 << 5,
        deconvolution = 1 << 6,
        eltwise = 1 << 7,
        lrn = 1 << 8,
        batch_normalization = 1 << 9,
        inner_product = 1 << 10,
        rnn = 1 << 11,
        gemm = 1 << 12,
        binary = 1 << 13,
        matmul = 1 << 14,
        resampling = 1 << 15,
        pooling = 1 << 16,
        reduction = 1 << 17,
        prelu = 1 << 18,
        softmax = 1 << 19,
        layer_normalization = 1 << 20,
        group_normalization = 1 << 21,
        graph = 1 << 22,
        gemm_api = 1 << 23,
        all = (uint32_t)-1,
    };
};

struct filter_status_t {
    enum flags : uint32_t {
        none = 0,
        valid,
        invalid,
    };

    flags status = flags::none;
    // used to form a message about proper components used
    std::string components;
    std::string err_msg;
};

inline component_t::flag_kind prim_kind2_comp_kind(
        const primitive_kind_t prim_kind) {
    return static_cast<component_t::flag_kind>(1 << prim_kind | 1 << 0);
}

uint32_t get_verbose(verbose_t::flag_kind kind = verbose_t::none,
        component_t::flag_kind filter_kind = component_t::all) noexcept;

// Helper to avoid #ifdefs for DNNL_DEV_MODE related logging
static inline uint32_t get_verbose_dev_mode(
        verbose_t::flag_kind kind = verbose_t::none) {
    return is_dev_mode() ? get_verbose(kind) : 0;
}

bool get_verbose_timestamp();

/// A container for primitive desc verbose string.
struct primitive_desc_t;
struct pd_info_t {
    pd_info_t() = default;
    pd_info_t(const pd_info_t &rhs)
        : str_(rhs.str_), is_initialized_(rhs.is_initialized_) {}
    pd_info_t &operator=(const pd_info_t &rhs) {
        is_initialized_ = rhs.is_initialized_;
        str_ = rhs.str_;
        return *this;
    }
    ~pd_info_t() = default;

    const char *c_str() const { return str_.c_str(); }
    bool is_initialized() const { return is_initialized_; }

    void init(engine_t *engine, const primitive_desc_t *pd);

private:
    std::string str_;

#if defined(DISABLE_VERBOSE)
    bool is_initialized_ = true; // no verbose -> info is always ready
#else
    bool is_initialized_ = false;
#endif

    // Alas, `std::once_flag` cannot be manually set and/or copied (in terms of
    // its state). Hence, when `pd_info_t` is copied the `initialization_flag_`
    // is always reset. To avoid re-initialization we use an extra
    // `is_initialized_` flag, that should be checked before calling `init()`.
    std::once_flag initialization_flag_;
};

// Enum to define which dims member of memory::desc to be dumped.
enum class dims_type_t {
    undef,
    dims,
    strides,
};

std::string md2fmt_str(const memory_desc_t *md, format_kind_t user_format);
std::string md2dim_str(
        const memory_desc_t *md, dims_type_t dims_type = dims_type_t::dims);
// Returns a verbose string of dimensions or descriptor from src, wei, and/or
// dst memory descs. Can be called externally to provide info about actual
// values of runtime dimensions.
std::string rt_dims2fmt_str(primitive_kind_t prim_kind,
        const memory_desc_t *src_md, const memory_desc_t *wei_md,
        const memory_desc_t *dst_md);
// Returns a verbose string of all supported by a primitive memory descriptors.
// Can be called externally to provide info about actual tag and stride values
// of runtime dimensions.
std::string rt_mds2str(primitive_kind_t prim_kind, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *bia_md,
        const memory_desc_t *dst_md);

} // namespace impl
} // namespace dnnl

#endif
