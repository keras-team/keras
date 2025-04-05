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

#ifndef COMMON_TYPE_HELPERS_HPP
#define COMMON_TYPE_HELPERS_HPP

#include <algorithm>
#include <assert.h>
#include <math.h>

#include "oneapi/dnnl/dnnl.h"

#include "bit_cast.hpp"
#include "c_types_map.hpp"
#include "dnnl_traits.hpp"
#include "math_utils.hpp"
#include "memory_desc.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

// Global zero memory descriptor. Mostly used for queries to return
extern memory_desc_t DNNL_API glob_zero_md;

template <typename base_type, typename derived_type>
status_t safe_ptr_assign(base_type *&lhs, derived_type *rhs) {
    if (rhs == nullptr) return status::out_of_memory;
    lhs = rhs;
    return status::success;
}

template <typename base_type, typename derived_type>
status_t safe_ptr_assign(std::unique_ptr<base_type> &lhs, derived_type *rhs) {
    if (rhs == nullptr) return status::out_of_memory;
    lhs.reset(rhs);
    return status::success;
}

template <typename T, typename U>
struct is_subset {
    static constexpr bool value = false;
};
template <typename T>
struct is_subset<T, T> {
    static constexpr bool value = true;
};
template <typename T>
struct is_subset<T,
        typename utils::enable_if<nstl::is_integral<T>::value, float>::type> {
    static constexpr bool value = true;
};
#define ISSPEC(t1, t2) \
    template <> \
    struct is_subset<t1, t2> { \
        static constexpr bool value = true; \
    }
ISSPEC(int16_t, int32_t);
ISSPEC(int8_t, int32_t);
ISSPEC(uint8_t, int32_t);
ISSPEC(int8_t, int16_t);
ISSPEC(uint8_t, int16_t);
#undef ISSPEC

inline bool operator==(const memory_desc_t &lhs, const memory_desc_t &rhs);

namespace types {

inline size_t data_type_size(data_type_t data_type) {
    using namespace data_type;
    switch ((int)data_type) {
        case f8_e5m2: return sizeof(prec_traits<f8_e5m2>::type);
        case f8_e4m3: return sizeof(prec_traits<f8_e4m3>::type);
        case f16: return sizeof(prec_traits<f16>::type);
        case bf16: return sizeof(prec_traits<bf16>::type);
        case tf32: // the tf32 type is an f32
        case f32: return sizeof(prec_traits<f32>::type);
        case f64: return sizeof(prec_traits<f64>::type);
        case s32: return sizeof(prec_traits<s32>::type);
        case s8: return sizeof(prec_traits<s8>::type);
        case u8: return sizeof(prec_traits<u8>::type);
        case s4: return sizeof(prec_traits<s4>::type);
        case u4: return sizeof(prec_traits<u4>::type);
        case boolean: return sizeof(prec_traits<boolean>::type);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return (size_t)-1; /* not supposed to be reachable */
}

template <typename T>
inline T max_value(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<T>(nstl::numeric_limits<prec_traits<x>::type>::max())
    switch (data_type) {
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(f16);
        CASE(bf16);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        CASE(s4);
        CASE(u4);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return static_cast<T>(0); /* not supposed to be reachable */
#undef CASE
}

// This is a hack to comply with a big comment below.
template <>
inline float max_value(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<float>( \
                nstl::numeric_limits<prec_traits<x>::type>::max())
    switch (data_type) {
        CASE(f8_e5m2);
        CASE(f8_e4m3);
        CASE(f16);
        CASE(bf16);
        CASE(s8);
        CASE(u8);
        CASE(s4);
        CASE(u4);
        // INT_MAX is not representable in float. The nearest float to it is
        // INT_MAX + 1 = 2^31 (0x4f000000). Regular conversion instructions such
        // as `cvtps2dq` or `cvtss2si` will convert this number to INT_MIN
        // making the result negative. We on purpose choose the previous float
        // number (0x4effffff) to return leaving the output close to INT_MAX but
        // still positive. In addition, we adjust validation of this approach.
        // The main concern against `real` saturation is performance, which
        // likely to drop (but it was not proved). The only drawback of current
        // approach is saturating on some integer values before it should happen
        // in the reality.
        case s32: return 2147483520.f;
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return 0.f; /* not supposed to be reachable */
#undef CASE
}

inline format_kind_t format_tag_to_kind(format_tag_t tag) {
    switch (tag) {
        case format_tag::undef: return format_kind::undef;
        case format_tag::any: return format_kind::any;
        case format_tag::last: return format_kind::undef;
        default: return format_kind::blocked;
    }

    assert(!"unreachable");
    return format_kind::undef;
}

// Currently rnn_s8s8_compensation has common bits with rnn_u8s8_compensation
// and scale_adjust constants so we have to perform additional checks to
// separate these two cases
inline bool extra_flag_rnn_s8s8_compensation_is_set(uint64_t flags) {
    return ((flags & memory_extra_flags::rnn_s8s8_compensation)
                   ^ memory_extra_flags::rnn_s8s8_compensation)
            == 0;
}

inline bool memory_extra_desc_is_equal(
        const memory_extra_desc_t &lhs, const memory_extra_desc_t &rhs) {
    using namespace memory_extra_flags;
    return true && lhs.flags == rhs.flags
            && IMPLICATION(lhs.flags & compensation_conv_s8s8,
                    lhs.compensation_mask == rhs.compensation_mask)
            && IMPLICATION((lhs.flags & rnn_u8s8_compensation)
                            && !extra_flag_rnn_s8s8_compensation_is_set(
                                    lhs.flags),
                    lhs.compensation_mask == rhs.compensation_mask)
            && IMPLICATION((lhs.flags & scale_adjust)
                            && !extra_flag_rnn_s8s8_compensation_is_set(
                                    lhs.flags),
                    lhs.scale_adjust == rhs.scale_adjust)
            && IMPLICATION(lhs.flags & compensation_conv_asymmetric_src,
                    lhs.asymm_compensation_mask == rhs.asymm_compensation_mask);
}

inline bool blocking_desc_is_equal(const memory_desc_t &lhs_md,
        const memory_desc_t &rhs_md, bool ignore_strides = false) {
    using dnnl::impl::utils::array_cmp;

    auto is_sparse_packed_desc = [](const memory_desc_t &md) {
        return md.format_kind == format_kind::sparse
                && md.format_desc.sparse_desc.encoding
                == sparse_encoding::packed;
    };

    const bool lhs_is_sparse_packed_desc = is_sparse_packed_desc(lhs_md);
    const bool rhs_is_sparse_packed_desc = is_sparse_packed_desc(rhs_md);

    if (lhs_md.format_kind != format_kind::blocked
            && !lhs_is_sparse_packed_desc)
        return false;
    if (rhs_md.format_kind != format_kind::blocked
            && !rhs_is_sparse_packed_desc)
        return false;

    const auto &lhs = lhs_md.format_kind == format_kind::sparse
            ? lhs_md.format_desc.sparse_desc.packed_desc
            : lhs_md.format_desc.blocking;
    const auto &rhs = rhs_md.format_kind == format_kind::sparse
            ? rhs_md.format_desc.sparse_desc.packed_desc
            : rhs_md.format_desc.blocking;
    bool equal = lhs.inner_nblks == rhs.inner_nblks
            && array_cmp(lhs.inner_blks, rhs.inner_blks, lhs.inner_nblks)
            && array_cmp(lhs.inner_idxs, rhs.inner_idxs, lhs.inner_nblks);
    if (ignore_strides) return equal;

    // Check the strides.
    // Note: for dimensions of size `1` the stride doesn't really matter.
    for (int d = 0; d < lhs_md.ndims; ++d) {
        if (lhs_md.dims[d] == 1 && lhs_md.padded_dims[d] == 1) continue;
        equal = equal && lhs.strides[d] == rhs.strides[d];
    }

    return equal;
}

inline bool wino_desc_is_equal(const wino_desc_t &lhs, const wino_desc_t &rhs) {
    return lhs.wino_format == rhs.wino_format && lhs.alpha == rhs.alpha
            && lhs.ic == rhs.ic && lhs.oc == rhs.oc
            && lhs.ic_block == rhs.ic_block && lhs.oc_block == rhs.oc_block
            && lhs.ic2_block == rhs.ic2_block && lhs.oc2_block == rhs.oc2_block
            && lhs.r == rhs.r;
}

inline bool rnn_packed_desc_is_equal(
        const rnn_packed_desc_t &lhs, const rnn_packed_desc_t &rhs) {
    bool ok = true && lhs.format == rhs.format && lhs.ldb == rhs.ldb
            && lhs.n_parts == rhs.n_parts
            && lhs.offset_compensation == rhs.offset_compensation
            && lhs.size == rhs.size && lhs.n == rhs.n;
    if (!ok) return false;

    for (int i = 0; i < rhs.n_parts; i++)
        ok = ok && lhs.parts[i] == rhs.parts[i];
    for (int i = 0; i < rhs.n_parts; i++)
        ok = ok && lhs.part_pack_size[i] == rhs.part_pack_size[i];
    return ok;
}

inline bool sparse_desc_is_equal(
        const sparse_desc_t &lhs, const sparse_desc_t &rhs) {
    bool ok = lhs.encoding == rhs.encoding && lhs.nnz == rhs.nnz;
    if (!ok) return false;

    for (int i = 0; i < sparse_desc_t::max_metadata_types; i++)
        ok = ok && lhs.metadata_types[i] == rhs.metadata_types[i];

    return ok;
}

inline memory_desc_t zero_md() {
    auto zero = memory_desc_t();
    return zero;
}

inline bool is_zero_md(const memory_desc_t *md) {
    return md == nullptr || *md == zero_md();
}

inline data_type_t default_accum_data_type(
        data_type_t src_dt, data_type_t dst_dt, bool strict = true) {
    using namespace utils;
    using namespace data_type;

    // we allow to use f32 accumulation type only when the
    // accumulation chain is small. Otherwise, strict should be set to
    // true
    if (one_of(src_dt, s8, u8, u4, s4) && (dst_dt != f32 || strict)) return s32;

    if (one_of(f8_e5m2, src_dt, dst_dt)) return f32;
    if (one_of(f8_e4m3, src_dt, dst_dt)) return f32;
    if (one_of(f16, src_dt, dst_dt)) return f32;
    if (one_of(bf16, src_dt, dst_dt)) return f32;
    if (one_of(f32, src_dt, dst_dt)) return f32;
    if (one_of(f64, src_dt, dst_dt)) return f64;
    if (one_of(s32, src_dt, dst_dt)) return s32;

    if (one_of(s8, src_dt, dst_dt) || one_of(u8, src_dt, dst_dt)
            || one_of(s4, src_dt, dst_dt) || one_of(u4, src_dt, dst_dt))
        return s32;

    return data_type::undef;
}

inline data_type_t default_accum_data_type(data_type_t src_dt,
        data_type_t wei_dt, data_type_t dst_dt, prop_kind_t prop_kind) {
    using namespace utils;
    using namespace data_type;
    using namespace prop_kind;

    /* prop_kind doesn't matter */
    if (everyone_is(f32, src_dt, wei_dt)) return f32;
    if (everyone_is(f64, src_dt, wei_dt)) return f64;

    if (one_of(prop_kind, forward_training, forward_inference)) {
        if ((src_dt == u8 || src_dt == s8) && wei_dt == s8) return s32;
        if (one_of(f16, src_dt, wei_dt)) return f32;
        // weights decompression
        if (one_of(src_dt, bf16, f32) && one_of(wei_dt, u8, s8, s4, u4))
            return f32;
    } else if (prop_kind == backward_data) {
        if (one_of(src_dt, f32, s32, s8, u8) && wei_dt == s8
                && one_of(dst_dt, s8, u8, s32))
            return s32;
        if (one_of(f16, dst_dt, wei_dt)) return f32;
        if (everyone_is(f32, dst_dt, wei_dt) && one_of(src_dt, s8, u8))
            return f32;
    }

    if (one_of(f8_e5m2, src_dt, wei_dt, dst_dt)) return f32;
    if (one_of(f8_e4m3, src_dt, wei_dt, dst_dt)) return f32;
    if (one_of(bf16, src_dt, wei_dt, dst_dt)) return f32;
    if (one_of(f16, src_dt, wei_dt, dst_dt)) return f32;

    return data_type::undef;
}

inline bool is_integral_dt(data_type_t dt) {
    using namespace data_type;
    return utils::one_of(dt, s32, s8, u8, u4, s4);
}

template <typename data_t>
inline void cvt_from_float(data_t *out, const float *inp, size_t nelems)
        = delete;

template <typename data_t>
inline void cvt_to_float(float *out, const data_t *inp, size_t nelems) = delete;

template <>
inline void cvt_from_float<float>(float *out, const float *inp, size_t nelems) {
    // This operation should be avoided as it does nothing useful
    assert(!"unexpected");
    for (size_t i = 0; i < nelems; i++)
        out[i] = inp[i];
}

template <>
inline void cvt_to_float<float>(float *out, const float *inp, size_t nelems) {
    // This operation should be avoided as it does nothing useful
    assert(!"unexpected");
    for (size_t i = 0; i < nelems; i++)
        out[i] = inp[i];
}

template <>
inline void cvt_from_float<bfloat16_t>(
        bfloat16_t *out, const float *inp, size_t nelems) {
    cvt_float_to_bfloat16(out, inp, nelems);
}

template <>
inline void cvt_to_float<bfloat16_t>(
        float *out, const bfloat16_t *inp, size_t nelems) {
    cvt_bfloat16_to_float(out, inp, nelems);
}

template <>
inline void cvt_from_float<float16_t>(
        float16_t *out, const float *inp, size_t nelems) {
    cvt_float_to_float16(out, inp, nelems);
}

template <>
inline void cvt_to_float<float16_t>(
        float *out, const float16_t *inp, size_t nelems) {
    cvt_float16_to_float(out, inp, nelems);
}

inline void cvt_from_float(
        data_type_t dt, void *out, const float *inp, size_t nelems) {
    switch (dt) {
        case data_type::bf16:
            cvt_from_float((bfloat16_t *)out, inp, nelems);
            break;
        case data_type::f16:
            cvt_from_float((float16_t *)out, inp, nelems);
            break;
        default: assert(!"unimplemented");
    }
}

inline void cvt_to_float(
        data_type_t dt, float *out, const void *inp, size_t nelems) {
    switch (dt) {
        case data_type::bf16:
            cvt_to_float(out, (const bfloat16_t *)inp, nelems);
            break;
        case data_type::f16:
            cvt_to_float(out, (const float16_t *)inp, nelems);
            break;
        default: assert(!"unimplemented");
    }
}

} // namespace types

inline bool operator==(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    using namespace dnnl::impl::utils;
    // quick path for zero_mds
    if (utils::everyone_is(0, lhs.ndims, rhs.ndims)) return true;

    bool base_equal = true && lhs.ndims == rhs.ndims
            && array_cmp(lhs.dims, rhs.dims, lhs.ndims)
            && lhs.data_type == rhs.data_type
            && array_cmp(lhs.padded_dims, rhs.padded_dims, lhs.ndims)
            && array_cmp(lhs.padded_offsets, rhs.padded_offsets, lhs.ndims)
            && lhs.offset0 == rhs.offset0 && lhs.format_kind == rhs.format_kind;
    if (!base_equal) return false;
    if (!types::memory_extra_desc_is_equal(lhs.extra, rhs.extra)) return false;
    if (lhs.format_kind == format_kind::blocked)
        return types::blocking_desc_is_equal(lhs, rhs);
    else if (lhs.format_kind == format_kind::wino)
        return types::wino_desc_is_equal(
                lhs.format_desc.wino_desc, rhs.format_desc.wino_desc);
    else if (lhs.format_kind == format_kind::rnn_packed)
        return types::rnn_packed_desc_is_equal(lhs.format_desc.rnn_packed_desc,
                rhs.format_desc.rnn_packed_desc);
    else if (lhs.format_kind == format_kind::sparse)
        return types::sparse_desc_is_equal(
                lhs.format_desc.sparse_desc, rhs.format_desc.sparse_desc);
    return true;
}

inline bool operator!=(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    return !operator==(lhs, rhs);
}

// Comparison operators for descriptors
#define COMPARE_DESC_MEMBERS(m) lhs.m == rhs.m
#define COMPARE_DESC_ARRAY_MEMBERS(m, s) utils::array_cmp(lhs.m, rhs.m, s)
#define DEREF_AND_COMPARE_DESC_MEMBERS(m) *lhs.m == *rhs.m
#define COMPARE_FLOAT_DESC_MEMBERS(m) utils::equal_with_nan(lhs.m, rhs.m)
#define COMPARE_FLOAT_DESC_ARRAY_MEMBERS(m, s) \
    !std::memcmp(lhs.m, rhs.m, sizeof(float) * s)

// clang-format off
inline bool operator==(const batch_normalization_desc_t &lhs,
        const batch_normalization_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(scaleshift_desc)
            && COMPARE_DESC_MEMBERS(diff_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(stat_desc)
            && COMPARE_FLOAT_DESC_MEMBERS(batch_norm_epsilon)
            && COMPARE_DESC_MEMBERS(flags);
    return ret;
}

inline bool operator==(const binary_desc_t &lhs, const binary_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc[0])
            && COMPARE_DESC_MEMBERS(src_desc[1])
            && COMPARE_DESC_MEMBERS(dst_desc);
    return ret;
}

inline bool operator==(const concat_desc_t &lhs, const concat_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && DEREF_AND_COMPARE_DESC_MEMBERS(dst_md)
            && COMPARE_DESC_MEMBERS(n)
            && COMPARE_DESC_MEMBERS(concat_dimension);

    if (!ret) return ret;

    for (int i = 0; i < lhs.n; i++) {
        ret = *lhs.src_mds[i] == *rhs.src_mds[i];
        if (!ret) break;
    }
    return ret;
}

inline bool operator==(
        const convolution_desc_t &lhs, const convolution_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(diff_bias_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_ARRAY_MEMBERS(strides, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(dilates, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[0], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[1], DNNL_MAX_NDIMS)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}

inline bool operator==(const eltwise_desc_t &lhs, const eltwise_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_FLOAT_DESC_MEMBERS(alpha)
            && COMPARE_FLOAT_DESC_MEMBERS(beta);
    return ret;
}

inline bool operator==(const gemm_desc_t &lhs, const gemm_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(a_desc)
            && COMPARE_DESC_MEMBERS(b_desc)
            && COMPARE_DESC_MEMBERS(c_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(acc_type)
            && COMPARE_DESC_MEMBERS(sum_ab)
            && COMPARE_DESC_MEMBERS(sum_ab_type);
    return ret;
}

inline bool operator==(
        const group_normalization_desc_t &lhs, const group_normalization_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(scaleshift_desc)
            && COMPARE_DESC_MEMBERS(diff_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(stat_desc)
            && COMPARE_DESC_MEMBERS(groups)
            && COMPARE_FLOAT_DESC_MEMBERS(group_norm_epsilon)
            && COMPARE_DESC_MEMBERS(flags);
     return ret;
}

inline bool operator==(
        const inner_product_desc_t &lhs, const inner_product_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(diff_bias_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}

inline bool operator==(
        const layer_normalization_desc_t &lhs, const layer_normalization_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(data_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(diff_data_scaleshift_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(stat_desc)
            && COMPARE_FLOAT_DESC_MEMBERS(layer_norm_epsilon)
            && COMPARE_DESC_MEMBERS(flags);
     return ret;
}

inline bool operator==(const lrn_desc_t &lhs, const lrn_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(local_size)
            && COMPARE_FLOAT_DESC_MEMBERS(lrn_alpha)
            && COMPARE_FLOAT_DESC_MEMBERS(lrn_beta)
            && COMPARE_FLOAT_DESC_MEMBERS(lrn_k);
    return ret;
}

inline bool operator==(const matmul_desc_t &lhs, const matmul_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}

inline bool operator==(
        const pooling_desc_t &lhs, const pooling_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_ARRAY_MEMBERS(strides, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(kernel, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[0], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[1], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(dilation, DNNL_MAX_NDIMS)
            && COMPARE_DESC_MEMBERS(accum_data_type);
     return ret;
}

inline bool operator==(const prelu_desc_t &lhs, const prelu_desc_t &rhs) {
    const bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc);
    return ret;
}

inline bool operator==(
        const reduction_desc_t &lhs, const reduction_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_FLOAT_DESC_MEMBERS(p)
            && COMPARE_FLOAT_DESC_MEMBERS(eps);
    return ret;
}

inline bool operator==(const reorder_desc_t &lhs, const reorder_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && DEREF_AND_COMPARE_DESC_MEMBERS(src_md)
            && DEREF_AND_COMPARE_DESC_MEMBERS(dst_md)
            && COMPARE_DESC_MEMBERS(src_engine_kind)
            && COMPARE_DESC_MEMBERS(dst_engine_kind)
            && COMPARE_DESC_MEMBERS(is_cross_engine);
    return ret;
}

inline bool operator==(
        const resampling_desc_t &lhs, const resampling_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_FLOAT_DESC_ARRAY_MEMBERS(factors, DNNL_MAX_NDIMS);
    return ret;
}

inline bool operator==(const rnn_desc_t &lhs, const rnn_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(cell_kind)
            && COMPARE_DESC_MEMBERS(direction)
            && COMPARE_DESC_MEMBERS(src_layer_desc)
            && COMPARE_DESC_MEMBERS(src_iter_desc)
            && COMPARE_DESC_MEMBERS(src_iter_c_desc)
            && COMPARE_DESC_MEMBERS(weights_layer_desc)
            && COMPARE_DESC_MEMBERS(weights_iter_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(dst_layer_desc)
            && COMPARE_DESC_MEMBERS(dst_iter_desc)
            && COMPARE_DESC_MEMBERS(dst_iter_c_desc)
            && COMPARE_DESC_MEMBERS(weights_peephole_desc)
            && COMPARE_DESC_MEMBERS(weights_projection_desc)
            && COMPARE_DESC_MEMBERS(diff_src_layer_desc)
            && COMPARE_DESC_MEMBERS(diff_src_iter_desc)
            && COMPARE_DESC_MEMBERS(diff_src_iter_c_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_layer_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_iter_desc)
            && COMPARE_DESC_MEMBERS(diff_bias_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_layer_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_iter_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_iter_c_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_peephole_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_projection_desc)
            && COMPARE_DESC_MEMBERS(flags)
            && COMPARE_DESC_MEMBERS(activation_kind)
            && COMPARE_FLOAT_DESC_MEMBERS(alpha)
            && COMPARE_FLOAT_DESC_MEMBERS(beta);
    return ret;
}

inline bool operator==(const shuffle_desc_t &lhs, const shuffle_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(axis)
            && COMPARE_DESC_MEMBERS(group_size);
    return ret;
}

inline bool operator==(
        const softmax_desc_t &lhs, const softmax_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_MEMBERS(softmax_axis);
     return ret;
}

inline bool operator==(const sum_desc_t &lhs, const sum_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && DEREF_AND_COMPARE_DESC_MEMBERS(dst_md)
            && COMPARE_DESC_MEMBERS(n);
    if (!ret) return ret;

    for (int i = 0; i < lhs.n; i++) {
        ret = *lhs.src_mds[i] == *rhs.src_mds[i];
        if (!ret) break;
    }
    if (!ret) return ret;

    for (int i = 0; i < lhs.n; i++) {
        ret = ret && COMPARE_FLOAT_DESC_MEMBERS(scales[i]);
        if (!ret) break;
    }

    return ret;
}

inline bool operator==(const zero_pad_desc_t &lhs, const zero_pad_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind);
    return ret;
}
// clang-format on

#undef COMPARE_DESC_MEMBERS
#undef COMPARE_DESC_ARRAY_MEMBERS
#undef DEREF_AND_COMPARE_DESC_MEMBERS
#undef COMPARE_FLOAT_DESC_MEMBERS
#undef COMPARE_FLOAT_DESC_ARRAY_MEMBERS

inline bool is_dense_format_kind(const std::vector<const memory_desc_t *> mds) {
#ifdef DNNL_EXPERIMENTAL_SPARSE
    for (const auto *md : mds)
        if (md->format_kind == format_kind::sparse) return false;
#endif
    return true;
}

inline memory_desc_t cvt_blocked2sparse_packed(
        const memory_desc_t &blocked_md, dim_t nnz) {
    if (blocked_md.format_kind != format_kind::blocked) return glob_zero_md;

    auto sparse_packed_md = blocked_md;
    sparse_packed_md.format_kind = format_kind::sparse;
    sparse_packed_md.format_desc.sparse_desc.encoding = sparse_encoding::packed;
    sparse_packed_md.format_desc.sparse_desc.nnz = nnz;
    sparse_packed_md.format_desc.sparse_desc.packed_desc
            = blocked_md.format_desc.blocking;
    return sparse_packed_md;
}

inline memory_desc_t cvt_sparse_packed2blocked(
        const memory_desc_t &sparse_packed_md) {
    if (sparse_packed_md.format_kind != format_kind::sparse
            || sparse_packed_md.format_desc.sparse_desc.encoding
                    != sparse_encoding::packed)
        return glob_zero_md;

    const blocking_desc_t &blk_desc
            = sparse_packed_md.format_desc.sparse_desc.packed_desc;
    auto blocked_md = sparse_packed_md;
    blocked_md.format_desc.blocking = blk_desc;
    blocked_md.format_kind = format_kind::blocked;
    return blocked_md;
}

/** returns true if strides are compatible with memory_desc_t */
inline bool memory_desc_strides_check(
        const memory_desc_t &md, const dims_t strides) {
    if (strides == nullptr || md.ndims == 0
            || md.format_kind != format_kind::blocked)
        return true;

    dims_t blocks = {0};
    int perm[DNNL_MAX_NDIMS] = {0};
    for (int d = 0; d < md.ndims; ++d) {
        // no strides check needed for empty tensor
        if (md.padded_dims[d] == 0) return true;

        // no strides verification for runtime dims
        const bool has_runtime_dim = utils::one_of(
                DNNL_RUNTIME_DIM_VAL, strides[d], md.padded_dims[d]);
        if (has_runtime_dim) return true;

        perm[d] = d;
        blocks[d] = 1;
    }

    dim_t block_size = 1;
    const auto &blk = md.format_desc.blocking;
    for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
        blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];
        block_size *= blk.inner_blks[iblk];
    }

    // A custom comparator to yield linear order on perm
    auto idx_sorter = [&](const int a, const int b) -> bool {
        if (strides[a] == strides[b] && md.padded_dims[a] == md.padded_dims[b])
            return a < b;
        else if (strides[a] == strides[b])
            return md.padded_dims[a] < md.padded_dims[b];
        else
            return strides[a] < strides[b];
    };
    std::sort(perm, perm + md.ndims, idx_sorter);

    dim_t min_stride = block_size;
    for (int idx = 0; idx < md.ndims; ++idx) {
        const int d = perm[idx];

        // Make an exception for strides[d] == 0 as it has broadcast semantics
        // Note: owing to being sorted, these are the initial strides

        // FIXME: make an exception for dims[d] == 1 with the
        // assumption that no code applies that stride when the only
        // index accessed for that dimenstion is 0. This is because PT
        // can use "dummy" padding in those situations
        if ((strides[d] == 0) || (md.padded_dims[d] == 1))
            continue;
        else if (strides[d] < min_stride)
            return false;

        // update min_stride for next iteration
        const auto padded_dim = md.padded_dims[d];
        min_stride = block_size * strides[d] * (padded_dim / blocks[d]);
    }
    return true;
}

inline status_t memory_desc_init_by_strides(
        memory_desc_t &md, const dims_t strides) {
    return memory_desc_init_by_strides(
            md, md.ndims, md.dims, md.data_type, strides);
}

inline status_t memory_desc_init_by_tag(
        memory_desc_t &md, format_tag_t tag, const dims_t strides = nullptr) {

    const bool is_sparse = md.format_kind == format_kind::sparse;
    auto md_tmp = memory_desc_t();

    CHECK(memory_desc_init_by_tag(
            md_tmp, md.ndims, md.dims, md.data_type, tag));

    if (strides != nullptr && !memory_desc_strides_check(md_tmp, strides))
        return status::invalid_arguments;

    if (is_sparse) {
        if (md.format_desc.sparse_desc.encoding != sparse_encoding::packed
                || md.offset0 != 0)
            return status::invalid_arguments;
        md = cvt_blocked2sparse_packed(md_tmp, md.format_desc.sparse_desc.nnz);
    } else {
        md = md_tmp;
    }

    if (strides == nullptr) return status::success;

    for (int d = 0; d < md.ndims; ++d) {
        if (is_sparse)
            md.format_desc.sparse_desc.packed_desc.strides[d] = strides[d];
        else
            md.format_desc.blocking.strides[d] = strides[d];
    }

    return status::success;
}

/** inits memory descriptor based on logical dimensions kept in @p md, and the
 * blocking structure @p blk.
 *
 * @note blk.strides represent the order only (from smaller to bigger)
 *
 * TODO: move md related functions to one single place
 */
inline status_t memory_desc_init_by_blocking_desc(
        memory_desc_t &md, const blocking_desc_t &blk) {
    dims_t blocks = {0};
    utils::array_set(blocks, 1, md.ndims);
    dim_t block_size = 1;
    for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
        blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];
        block_size *= blk.inner_blks[iblk];
    }

    for (int d = 0; d < md.ndims; ++d) {
        md.padded_dims[d] = utils::rnd_up(md.dims[d], blocks[d]);
        md.padded_offsets[d] = 0;
    }
    md.offset0 = 0;

    md.format_kind = format_kind::blocked;
    auto &mblk = md.format_desc.blocking;
    mblk = blk;

    const int ndims = nstl::min(DNNL_MAX_NDIMS, md.ndims); // make GCC 5 happy
    utils::array_copy(mblk.strides, blk.strides, ndims);

    dims_t ou_blocks = {0};
    utils::array_copy(ou_blocks, md.padded_dims, ndims);

    int perm[DNNL_MAX_NDIMS];
    for (int d = 0; d < ndims; ++d) {
        perm[d] = d;
        ou_blocks[d] /= blocks[d];
    }

    utils::simultaneous_sort(
            mblk.strides, ou_blocks, perm, ndims, [](stride_t a, stride_t b) {
                if (utils::one_of(DNNL_RUNTIME_DIM_VAL, a, b))
                    return DNNL_RUNTIME_DIM_VAL;
                return b - a;
            });

    dim_t stride = block_size;
    for (int _d = ndims - 1; _d >= 0; --_d) {
        const int d = perm[_d];
        md.format_desc.blocking.strides[d] = stride;
        if (md.padded_dims[d] != 0) { // Keep same stride for zero dim
            stride *= md.padded_dims[d] / blocks[d];
        }
    }

    md.extra = utils::zero<memory_extra_desc_t>();

    return status::success;
}

/** inits memory descriptor @p md based on another one memory descriptor
 * @p md_base and given @p data_type.
 * Essentially: { md = md_base; md.dt = data_type; } */
inline status_t memory_desc_init_by_md_and_dt(memory_desc_t &md,
        const memory_desc_t &md_base, data_type_t data_type) {
    if (&md != &md_base) md = md_base;
    md.data_type = data_type;
    return status::success;
}

/** returns true if memory desc @p md corresponds to the given format tag.
 * Assumes a dense structure such as that returned by memory_desc_init_by_tag().
 * Strides must match those returned by memory_desc_init_by_tag(), with one
 * exception: the strides of unit dimensions are ignored in order to align with
 * memory descriptor equality comparisons and hashing.
 */
inline bool memory_desc_matches_tag(const memory_desc_t &md, format_tag_t tag) {
    if (md.format_kind != format_kind::sparse) {
        if (md.format_kind != types::format_tag_to_kind(tag)) return false;
    }

    memory_desc_t md_gold;
    status_t status = memory_desc_init_by_tag(
            md_gold, md.ndims, md.dims, md.data_type, tag);
    if (status != status::success) return false;

    return types::blocking_desc_is_equal(md, md_gold);
}

/** returns matching tag (or undef if match is not found)
 * XXX: This is a workaround that eventually should go away! */
template <typename... Tags>
format_tag_t memory_desc_matches_one_of_tag(
        const memory_desc_t &md, Tags... tags) {
    for (const auto tag : {tags...}) {
        if (memory_desc_matches_tag(md, tag)) return tag;
    }
    return format_tag::undef;
}

/** returns true if fp32 value denotes DNNL_RUNTIME_F32_VAL */
inline bool is_runtime_value(float val) {
    return utils::bit_cast<unsigned>(val) == DNNL_RUNTIME_F32_VAL_REP.u;
}

/** returns true if s32 value denotes DNNL_RUNTIME_S32_VAL */
inline bool is_runtime_value(int val) {
    return val == DNNL_RUNTIME_S32_VAL;
}

/** returns true if dim_t value denotes DNNL_RUNTIME_DIM_VAL */
inline bool is_runtime_value(dim_t val) {
    return val == DNNL_RUNTIME_DIM_VAL;
}

inline bool memory_desc_sanity_check(int ndims, const dims_t dims,
        data_type_t data_type, format_kind_t format_kind) {
    using namespace data_type;

    if (ndims == 0) return true;

    bool ok = dims != nullptr && 0 < ndims && ndims <= DNNL_MAX_NDIMS
            && utils::one_of(data_type, f8_e5m2, f8_e4m3, f16, bf16, f32, f64,
                    s32, s8, u8, s4, u4);
    if (!ok) return false;

    bool has_runtime_dims = false;
    for (int d = 0; d < ndims; ++d) {
        if (dims[d] != DNNL_RUNTIME_DIM_VAL && dims[d] < 0) return false;
        if (dims[d] == DNNL_RUNTIME_DIM_VAL) has_runtime_dims = true;
    }

    if (has_runtime_dims) {
        // format `any` is currently not supported for run-time dims
        if (format_kind == format_kind::any) return false;
    }

    return true;
}

inline bool memory_desc_sanity_check(const memory_desc_t &md) {
    return memory_desc_sanity_check(
            md.ndims, md.dims, md.data_type, format_kind::undef);
}

inline void copy_c_op_desc(op_desc_t *dst, const op_desc_t *src) {
#define CASE_OP_DESC(pkind) \
    case primitive_kind::pkind: dst->pkind = src->pkind; break;

    switch ((int)src->kind) {
        CASE_OP_DESC(batch_normalization);
        CASE_OP_DESC(binary);
        CASE_OP_DESC(convolution);
        CASE_OP_DESC(deconvolution);
        CASE_OP_DESC(eltwise);
        CASE_OP_DESC(gemm);
        CASE_OP_DESC(group_normalization);
        CASE_OP_DESC(inner_product);
        CASE_OP_DESC(layer_normalization);
        CASE_OP_DESC(lrn);
        CASE_OP_DESC(matmul);
        CASE_OP_DESC(pooling);
        CASE_OP_DESC(prelu);
        CASE_OP_DESC(reduction);
        CASE_OP_DESC(resampling);
        CASE_OP_DESC(rnn);
        CASE_OP_DESC(shuffle);
        CASE_OP_DESC(softmax);

        // Internal descs
        CASE_OP_DESC(zero_pad);
        default: assert(!"unknown C primitive kind");
    }
#undef CASE_OP_DESC
}

} // namespace impl
} // namespace dnnl

#include "memory_desc_wrapper.hpp"

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
