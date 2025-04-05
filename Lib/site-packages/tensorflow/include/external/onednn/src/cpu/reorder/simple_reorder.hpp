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

#ifndef CPU_REORDER_SIMPLE_REORDER_HPP
#define CPU_REORDER_SIMPLE_REORDER_HPP

#include <algorithm>
#include <assert.h>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using bd = block_dim_t;
using ib = inner_blk_t;

template <impl::data_type_t type>
using data_t = typename prec_traits<type>::type;

template <impl::data_type_t type_i, impl::data_type_t type_o>
using _qz_a1b0 = q10n::qz_a1b0<data_t<type_i>, data_t<type_o>>;

template <impl::data_type_t type_i, impl::data_type_t type_o>
using _qz = q10n::qz<data_t<type_i>, data_t<type_o>>;

namespace fmt_order {
const bool keep = true;
const bool reverse = false;
const bool any = keep;
} // namespace fmt_order

namespace spec {
struct direct_copy {};
struct direct_copy_except_dim_0 {};
struct reference {};
struct conv_req_comp {}; // {s8, u8: asymmetric quantization}
} // namespace spec

#define SIMPLE_REORDER_TEMPL_DECL \
    impl::data_type_t type_i, impl::format_tag_t tag_i, \
            impl::data_type_t type_o, impl::format_tag_t tag_o, \
            bool order_keep
#define SIMPLE_REORDER_TEMPL_CALL type_i, tag_i, type_o, tag_o, order_keep

#define DECLARE_COMMON_PARAMS() \
    auto input = CTX_IN_MEM(const data_t<type_i> *, DNNL_ARG_FROM); \
    auto output = CTX_OUT_MEM(data_t<type_o> *, DNNL_ARG_TO); \
    const auto &scratchpad = ctx.get_scratchpad_grantor(); \
    MAYBE_UNUSED(scratchpad); \
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd->src_md()); \
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd->dst_md()); \
    DEFINE_ARG_SCALES_BUFFER_ATTR(pd->attr(), src_scales, DNNL_ARG_FROM); \
    DEFINE_ARG_SCALES_BUFFER_ATTR(pd->attr(), dst_scales_, DNNL_ARG_TO); \
    int src_scales_mask, dst_scales_mask; \
    CHECK(get_scales_mask(pd->attr(), &src_scales_mask, &dst_scales_mask)); \
    int scales_mask = std::max(src_scales_mask, dst_scales_mask); \
    MAYBE_UNUSED(scales_mask); \
    dim_t D_start, D_mask, D_rest; \
    pd->get_D_values(input_d, scales_mask, &D_start, &D_mask, &D_rest); \
    const float *dst_scales = pd->precompute_scales( \
            scratchpad, pd->attr(), D_mask, dst_scales_); \
    MAYBE_UNUSED(dst_scales); \
    DEFINE_ZERO_POINT_VALUE_ATTR(pd->attr(), src_zp, DNNL_ARG_FROM); \
    DEFINE_ZERO_POINT_VALUE_ATTR(pd->attr(), dst_zp, DNNL_ARG_TO); \
    const float alpha = src_scales[0] * dst_scales[0]; \
    MAYBE_UNUSED(alpha); \
    const float beta = pd->beta(); \
    MAYBE_UNUSED(beta);

#define GET_SCRATCHPAD_SIZE_ZERO() \
    static size_t get_scratchpad_size(const memory_desc_wrapper &input_d, \
            const memory_desc_wrapper &output_d) { \
        return 0; \
    }

/* specific reorders: common template */
template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_impl {};

namespace {
inline bool simple_fmt_check(bool order_keep, impl::format_tag_t tag_i,
        impl::format_tag_t tag_o, const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d) {
    if (input_d.has_runtime_dims_or_strides()) return false;
    return input_d.matches_tag(order_keep ? tag_i : tag_o)
            && output_d.matches_tag(order_keep ? tag_o : tag_i);
}
inline bool simple_po_check(const primitive_attr_t *attr) {
    const auto &po = attr->post_ops_;
    return po.len() == 0 || (po.len() == 1 && po.entry_[0].is_sum(false));
}
inline status_t get_scales_mask(
        const primitive_attr_t *attr, int *src_mask, int *dst_mask) {
    const auto &s = attr->scales_;

    if (src_mask == nullptr || dst_mask == nullptr)
        return status::invalid_arguments;

    *src_mask = 0;
    if (!s.get(DNNL_ARG_SRC).has_default_values())
        *src_mask = s.get(DNNL_ARG_SRC).mask_;

    *dst_mask = 0;
    if (!s.get(DNNL_ARG_DST).has_default_values())
        *dst_mask = s.get(DNNL_ARG_DST).mask_;

    // This is used in a check function.
    if (*src_mask > 0 && *dst_mask > 0 && *dst_mask != *src_mask)
        return status::invalid_arguments;
    return status::success;
}
inline bool simple_attr_check(const primitive_attr_t *attr,
        bool many_scales_support, bool sum_support) {
    using smask_t = primitive_attr_t::skip_mask_t;
    smask_t skip_mask = smask_t::scales_runtime;
    if (sum_support) skip_mask = skip_mask | smask_t::post_ops;
    if (!attr->has_default_values(skip_mask)) return false;
    if (many_scales_support) return true;
    int src_mask, dst_mask;
    if (get_scales_mask(attr, &src_mask, &dst_mask) != status::success)
        return false;
    return src_mask == 0 && dst_mask == 0;
}
} // namespace

/* specific reorders: implementation */
template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                        && utils::one_of(tag_o, format_tag::wio,
                                format_tag::wigo, format_tag::hwio,
                                format_tag::hwigo, format_tag::dhwio,
                                format_tag::dhwigo),
                spec::conv_req_comp>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace data_type;
        using namespace utils;

        if (input_d.has_runtime_dims_or_strides()) return false;

        int src_scales_mask, dst_scales_mask;
        auto status = get_scales_mask(attr, &src_scales_mask, &dst_scales_mask);
        if (status != status::success) return false;
        int scales_mask = std::max(src_scales_mask, dst_scales_mask);

        static constexpr bool w_groups = one_of(
                tag_o, format_tag::wigo, format_tag::hwigo, format_tag::dhwigo);

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool req_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        auto mask_ok = [&](bool check, int mask) {
            return IMPLICATION(check, mask == (w_groups ? 0x3 : 0x1));
        };

        return simple_attr_check(attr, true, false)
                && output_d.matches_tag(tag_o) && input_d.is_plain()
                && (req_comp || req_asymmetric_comp)
                && mask_ok(req_comp, output_d.extra().compensation_mask)
                && mask_ok(req_asymmetric_comp,
                        output_d.extra().asymm_compensation_mask)
                && IMPLICATION(!w_groups, one_of(scales_mask, 0, 0x1))
                && IMPLICATION(w_groups, one_of(scales_mask, 0, 0x3))
                && one_of(input_d.data_type(), f32, s8, bf16)
                && output_d.data_type() == s8;
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        static constexpr bool w_groups = utils::one_of(
                tag_o, format_tag::wigo, format_tag::hwigo, format_tag::dhwigo);
        static constexpr bool w_height
                = !utils::one_of(tag_o, format_tag::wio, format_tag::wigo);
        static constexpr bool w_depth
                = utils::one_of(tag_o, format_tag::dhwio, format_tag::dhwigo);

        const auto &dims = input_d.dims();

        const dim_t G = w_groups ? dims[0] : 1;
        const dim_t OC = dims[w_groups + 0];
        const dim_t IC = dims[w_groups + 1];
        const dim_t D = w_depth ? dims[w_groups + 2] : 1;
        const dim_t H = w_height ? dims[w_groups + w_depth + 2] : 1;
        const dim_t W = dims[w_groups + w_depth + w_height + 2];

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool has_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        assert(req_comp || has_asymmetric_comp);

        float adj_scale
                = (output_d.extra().flags & memory_extra_flags::scale_adjust)
                ? output_d.extra().scale_adjust
                : 1.f;

        size_t offset = output_d.size() - output_d.additional_buffer_size();
        size_t comp_size = output_d.additional_buffer_size(
                memory_extra_flags::compensation_conv_s8s8);
        size_t zp_offset = offset + (req_comp ? comp_size : 0);
        int32_t *cp = req_comp ? reinterpret_cast<int32_t *>(output + offset)
                               : nullptr;
        int32_t *zp = has_asymmetric_comp
                ? reinterpret_cast<int32_t *>(output + zp_offset)
                : nullptr;

        const bool per_oc = scales_mask & (1 << (w_groups + 0));
        const bool per_ic = scales_mask & (1 << (w_groups + 1));
        const size_t ic_stride = per_ic ? 1 : 0;
        const size_t oc_stride = per_oc ? per_ic ? IC : 1 : 0;

        parallel_nd(G, OC, [&](dim_t g, dim_t oc) {
            if (req_comp) cp[g * OC + oc] = 0;
            if (has_asymmetric_comp) zp[g * OC + oc] = 0;
            for_(dim_t ic = 0; ic < IC; ic++)
            for_(dim_t d = 0; d < D; d++)
            for_(dim_t h = 0; h < H; h++)
            for (dim_t w = 0; w < W; w++) {
                auto i = w_depth
                        ? input[input_d.blk_off<!w_groups>(g, oc, ic, d, h, w)]
                        : w_height
                        ? input[input_d.blk_off<!w_groups>(g, oc, ic, h, w)]
                        : input[input_d.blk_off<!w_groups>(g, oc, ic, w)];
                auto &o = w_depth ? output[output_d.blk_off<!w_groups>(
                                  g, oc, ic, d, h, w)]
                        : w_height
                        ? output[output_d.blk_off<!w_groups>(g, oc, ic, h, w)]
                        : output[output_d.blk_off<!w_groups>(g, oc, ic, w)];
                const size_t os_off
                        = (g * OC + oc) * oc_stride + ic * ic_stride;
                const float s = src_scales[src_scales_mask == 0 ? 0 : os_off];
                const float d = dst_scales[dst_scales_mask == 0 ? 0 : os_off];

                o = q10n::qz_b0<data_t<type_i>, data_t<type_o>>()(
                        i, s * adj_scale * d);
                if (req_comp) cp[g * OC + oc] -= (int32_t)o;
                if (has_asymmetric_comp) zp[g * OC + oc] -= (int32_t)o;
            }
            if (req_comp) cp[g * OC + oc] *= 128;
        });
        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<
                (utils::one_of(tag_i, format_tag::iwo, format_tag::oiw,
                         format_tag::wio)
                        && utils::one_of(tag_o, format_tag::OIw4i16o4i,
                                format_tag::OIw4i32o4i, format_tag::OIw4i64o4i,
                                format_tag::OIw2i8o4i, format_tag::OIw4o4i))
                        || (utils::one_of(tag_i, format_tag::oi, format_tag::io)
                                && utils::one_of(tag_o, format_tag::OI4i16o4i,
                                        format_tag::OI4i32o4i,
                                        format_tag::OI4i64o4i))
                        || (utils::one_of(
                                    tag_i, format_tag::goiw, format_tag::wigo)
                                && utils::one_of(tag_o, format_tag::gOIw4i16o4i,
                                        format_tag::gOIw2i8o4i,
                                        format_tag::gOIw4o4i))
                        || (utils::one_of(tag_i, format_tag::ihwo,
                                    format_tag::hwio, format_tag::oihw)
                                && utils::one_of(tag_o, format_tag::OIhw4i16o4i,
                                        format_tag::OIhw4i32o4i,
                                        format_tag::OIhw4i64o4i,
                                        format_tag::OIhw2i8o4i,
                                        format_tag::OIhw4o4i))
                        || (utils::one_of(tag_i, format_tag::idhwo,
                                    format_tag::dhwio, format_tag::oidhw)
                                && utils::one_of(tag_o,
                                        format_tag::OIdhw4i16o4i,
                                        format_tag::OIdhw4i32o4i,
                                        format_tag::OIdhw4i64o4i,
                                        format_tag::OIdhw2i8o4i,
                                        format_tag::OIdhw4o4i))
                        || (utils::one_of(
                                    tag_i, format_tag::goihw, format_tag::hwigo)
                                && utils::one_of(tag_o, format_tag::gOIhw4o4i,
                                        format_tag::gOIhw2i8o4i,
                                        format_tag::gOIhw4i16o4i))
                        || (utils::one_of(tag_i, format_tag::goidhw)
                                && (utils::one_of(tag_o,
                                        format_tag::gOIdhw4i16o4i,
                                        format_tag::gOIdhw2i8o4i,
                                        format_tag::gOIdhw4o4i))),
                spec::conv_req_comp>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace format_tag;
        using namespace data_type;
        using namespace utils;

        if (input_d.has_runtime_dims_or_strides()) return false;

        int src_scales_mask, dst_scales_mask;
        auto status = get_scales_mask(attr, &src_scales_mask, &dst_scales_mask);
        if (status != status::success) return false;
        int scales_mask = std::max(src_scales_mask, dst_scales_mask);

        const bool w_groups = !one_of(tag_o, OIw4i16o4i, OIw2i8o4i, OIw4o4i,
                OIhw4i16o4i, OIhw2i8o4i, OIhw4o4i, OIdhw4i16o4i, OIdhw2i8o4i,
                OIdhw4o4i, OI4i16o4i, OI4i32o4i, OI4i64o4i, OIw4i32o4i,
                OIw4i64o4i, OIhw4i32o4i, OIhw4i64o4i, OIdhw4i32o4i,
                OIdhw4i64o4i);

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool req_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        auto mask_ok = [&](bool check, int mask) {
            return IMPLICATION(check, mask == (w_groups ? 0x3 : 0x1));
        };

        return simple_attr_check(attr, true, false)
                && input_d.matches_tag(tag_i) && output_d.matches_tag(tag_o)
                && (req_comp || req_asymmetric_comp)
                && mask_ok(req_comp, output_d.extra().compensation_mask)
                && mask_ok(req_asymmetric_comp,
                        output_d.extra().asymm_compensation_mask)
                && IMPLICATION(!w_groups, one_of(scales_mask, 0, 0x1))
                && IMPLICATION(w_groups, one_of(scales_mask, 0, 0x3))
                && one_of(input_d.data_type(), f32, s8, bf16)
                && output_d.data_type() == s8;
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace format_tag;

        static constexpr bool w_groups = !utils::one_of(tag_o, OIw4o4i,
                OIw4i16o4i, OIhw4i16o4i, OIdhw4i16o4i, OIhw4o4i, OIw2i8o4i,
                OIhw2i8o4i, OIdhw2i8o4i, OIdhw4o4i, OI4i16o4i, OI4i32o4i,
                OI4i64o4i, OIw4i32o4i, OIw4i64o4i, OIhw4i32o4i, OIhw4i64o4i,
                OIdhw4i32o4i, OIdhw4i64o4i);

        constexpr int is_0d
                = utils::one_of(tag_o, OI4i16o4i, OI4i32o4i, OI4i64o4i);
        constexpr int is_1d
                = utils::one_of(tag_o, gOIw4i16o4i, OIw4i16o4i, gOIw2i8o4i,
                        OIw2i8o4i, gOIw4o4i, OIw4o4i, OIw4i32o4i, OIw4i64o4i);
        constexpr int is_3d = utils::one_of(tag_o, gOIdhw4i16o4i, OIdhw4i16o4i,
                gOIdhw2i8o4i, OIdhw2i8o4i, gOIdhw4o4i, OIdhw4o4i, OIdhw4i32o4i,
                OIdhw4i64o4i);
        constexpr dim_t icblksize = utils::one_of(tag_traits<tag_o>::inner_blks,
                                            ib::_4a4b, ib::_4b4c)
                ? 4
                : utils::one_of(tag_traits<tag_o>::inner_blks, ib::_2c8b4c,
                          ib::_2b8a4b)
                ? 8
                : 16;
        constexpr dim_t ocblksize
                = tag_traits<tag_o>::inner_blks == ib::_4b32a4b ? 32
                : tag_traits<tag_o>::inner_blks == ib::_4b64a4b ? 64
                                                                : icblksize;

        const auto &plain_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const int ndims = input_d.ndims();
        const auto &pdims
                = order_keep ? output_d.padded_dims() : input_d.padded_dims();

        const dim_t G = w_groups ? dims[0] : 1;
        const dim_t OC = dims[w_groups + 0];
        const dim_t PADDED_OC = pdims[w_groups + 0];
        const dim_t NB_OC = pdims[w_groups + 0] / ocblksize;
        const dim_t IC = dims[w_groups + 1];
        const dim_t NB_IC = pdims[w_groups + 1] / icblksize;
        const dim_t D = is_3d ? dims[2 + w_groups] : 1;
        const dim_t H = is_1d || is_0d ? 1 : dims[2 + w_groups + is_3d];
        const dim_t W = is_0d ? 1 : dims[w_groups + is_3d + 3 - is_1d];

        // XXX: Currently user can pass a mask that has non-zero values in
        // dimensions that do not exist in a md. Since attributes are created
        // separately mask can't be validated.
        // This line truncates a given mask in range [0, 1 << ndims - 1]
        // TODO: Such masks can be either prohibited at pd creation step at
        // API level or checked by each implementation that relies on it.
        scales_mask &= (1 << ndims) - 1;

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool has_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        assert(req_comp || has_asymmetric_comp);

        float adj_scale
                = (output_d.extra().flags & memory_extra_flags::scale_adjust)
                ? output_d.extra().scale_adjust
                : 1.f;

        const bool per_oc = scales_mask & (1 << (w_groups + 0));
        const bool per_ic = scales_mask & (1 << (w_groups + 1));
        const size_t ic_stride = per_ic ? 1 : 0;
        const size_t oc_stride = per_oc ? per_ic ? IC : 1 : 0;
        const size_t nb_ic_stride = (per_ic ? 1 : 0) * icblksize;
        const size_t nb_oc_stride = (per_oc ? per_ic ? IC : 1 : 0) * ocblksize;

        // This kernel is used primarily for tensors with multiple inner
        // blocks for which generic zero padding must be used.
        // TODO: apply zero padding inside parallel_nd()
        ctx.zero_pad_output(DNNL_ARG_TO);

        auto ker = [&](const data_t<type_i> *inp, data_t<type_o> *out,
                           int32_t *c, int32_t *zp, const float *s,
                           const float *d, const dim_t oc_block,
                           const dim_t ic_block) {
#define index AB_or_BC_blk_off<tag_traits<tag_o>::inner_blks>
            for_(dim_t ic = 0; ic < ic_block; ++ic)
            for (dim_t oc = 0; oc < oc_block; ++oc) {
                const auto plain_off
                        = oc * plain_d.blocking_desc().strides[w_groups + 0]
                        + ic * plain_d.blocking_desc().strides[w_groups + 1];
                const size_t os_off = oc * oc_stride + ic * ic_stride;
                const float src_scale = s[src_scales_mask == 0 ? 0 : os_off];
                const float dst_scale = d[dst_scales_mask == 0 ? 0 : os_off];
                out[index(oc, ic)]
                        = q10n::qz_b0<data_t<type_i>, data_t<type_o>>()(
                                inp[plain_off],
                                src_scale * adj_scale * dst_scale);
                if (req_comp) c[oc] -= (128 * (int32_t)(out[index(oc, ic)]));
                if (has_asymmetric_comp)
                    zp[oc] -= (int32_t)(out[index(oc, ic)]);
            }
#undef index
        };

        constexpr dim_t i_mult_ic = icblksize;
        constexpr dim_t i_mult_oc = ocblksize;
        constexpr dim_t o_mult = 1;

        size_t offset = output_d.size() - output_d.additional_buffer_size();
        size_t comp_size = output_d.additional_buffer_size(
                memory_extra_flags::compensation_conv_s8s8);
        size_t zp_offset = offset + (req_comp ? comp_size : 0);
        int32_t *cp = req_comp ? reinterpret_cast<int32_t *>(output + offset)
                               : nullptr;
        int32_t *zp = has_asymmetric_comp
                ? reinterpret_cast<int32_t *>(output + zp_offset)
                : nullptr;

        parallel_nd(G * PADDED_OC, [&](dim_t i) {
            if (req_comp) cp[i] = 0;
            if (has_asymmetric_comp) zp[i] = 0;
        });

#define wei_blk_off(md, g, o, i, d, h, w) \
    (is_0d                  ? (md).blk_off<!w_groups>(g, o, i) \
                    : is_1d ? (md).blk_off<!w_groups>(g, o, i, w) \
                    : is_3d ? (md).blk_off<!w_groups>(g, o, i, d, h, w) \
                            : (md).blk_off<!w_groups>(g, o, i, h, w))
        parallel_nd(G, NB_OC, [&](dim_t g, dim_t O) {
            for_(dim_t I = 0; I < NB_IC; I++)
            for_(dim_t d = 0; d < D; d++)
            for_(dim_t h = 0; h < H; h++)
            for (dim_t w = 0; w < W; w++) {
                auto i = &input[wei_blk_off(
                        input_d, g, i_mult_oc * O, i_mult_ic * I, d, h, w)];
                auto o = &output[wei_blk_off(
                        output_d, g, o_mult * O, o_mult * I, d, h, w)];
                const dim_t oc_block = nstl::min(ocblksize, OC - O * ocblksize);
                const dim_t ic_block = nstl::min(icblksize, IC - I * icblksize);
                dim_t _offset = (g * NB_OC + O) * ocblksize;
                dim_t os_nb_off
                        = (g * NB_OC + O) * nb_oc_stride + I * nb_ic_stride;
                const float *src_scales_ptr
                        = &src_scales[src_scales_mask == 0 ? 0 : os_nb_off];
                const float *dst_scales_ptr
                        = &dst_scales[dst_scales_mask == 0 ? 0 : os_nb_off];
                ker(i, o, (order_keep && req_comp) ? &cp[_offset] : nullptr,
                        (order_keep && has_asymmetric_comp) ? &zp[_offset]
                                                            : nullptr,
                        src_scales_ptr, dst_scales_ptr, oc_block, ic_block);
            }
        });

#undef wei_blk_off

        return status::success;
    }
};

/* Asymmetric Blocking */
template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<(utils::one_of(tag_i, format_tag::iwo,
                                           format_tag::oiw, format_tag::wio)
                                          && utils::one_of(
                                                  tag_o, format_tag::Owi16o))
                        || (utils::one_of(
                                    tag_i, format_tag::goiw, format_tag::wigo)
                                && utils::one_of(tag_o, format_tag::gOwi16o))
                        || (utils::one_of(tag_i, format_tag::ihwo,
                                    format_tag::hwio, format_tag::oihw)
                                && utils::one_of(tag_o, format_tag::Owhi16o))
                        || (utils::one_of(
                                    tag_i, format_tag::goihw, format_tag::hwigo)
                                && utils::one_of(tag_o, format_tag::gOwhi16o)),
                spec::conv_req_comp>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace format_tag;
        using namespace data_type;
        using namespace utils;

        if (input_d.has_runtime_dims_or_strides()) return false;

        const bool w_groups = !one_of(tag_o, Owi16o, Owhi16o);

        // Current formats are only used in jit kernels that natively
        // support s8 instructions, hence, there is no need for signed
        // compensation.
        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;

        const bool req_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        auto mask_ok = [&](bool check, int mask) {
            const int c_mask = 0x1,
                      g_mask = 0x3; // mask for i/o-channel and ngroups
            return IMPLICATION(check, mask == (w_groups ? g_mask : c_mask));
        };

        return simple_attr_check(attr, true, false)
                && input_d.matches_tag(tag_i) && output_d.matches_tag(tag_o)
                && mask_ok(req_asymmetric_comp,
                        output_d.extra().asymm_compensation_mask)
                && one_of(input_d.data_type(), f32, s8, bf16)
                && output_d.data_type() == s8 && !req_comp;
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace format_tag;

        static constexpr bool w_groups = !utils::one_of(tag_o, Owi16o, Owhi16o);
        constexpr int is_1d = utils::one_of(tag_o, Owi16o, gOwi16o);
        const bool is_3d = false; // TODO once enabled

        constexpr dim_t oc_blksize = 16;

        const auto &plain_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims
                = order_keep ? output_d.padded_dims() : input_d.padded_dims();

        const dim_t G = w_groups ? dims[0] : 1;
        const dim_t OC = dims[w_groups + 0];
        const dim_t NB_OC = pdims[w_groups + 0] / oc_blksize;
        const dim_t IC = dims[w_groups + 1];

        const dim_t D = is_3d ? dims[2 + w_groups] : 1;
        const dim_t H = is_1d ? 1 : dims[2 + w_groups + is_3d];
        const dim_t W = dims[w_groups + is_3d + 3 - is_1d];

        const bool has_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        float adj_scale
                = (output_d.extra().flags & memory_extra_flags::scale_adjust)
                ? output_d.extra().scale_adjust
                : 1.f;

        auto ker = [&](const data_t<type_i> *inp, data_t<type_o> *out,
                           int32_t *zp, const float *s, const float *d,
                           const dim_t oc_block) {
            for (dim_t oc = 0; oc < oc_block; ++oc) {
                const auto plain_off
                        = oc * plain_d.blocking_desc().strides[w_groups + 0];
                out[oc] = q10n::qz_b0<data_t<type_i>, data_t<type_o>>()(
                        inp[plain_off], s[oc] * adj_scale * d[oc]);
                if (has_asymmetric_comp) zp[oc] -= (int32_t)(out[oc]);
            }
            // fill memory with '0' in case of padded channel dimensions
            for (dim_t oc = oc_block; oc < oc_blksize; ++oc) {
                out[oc] = 0;
            }
        };

        size_t offset = output_d.size() - output_d.additional_buffer_size();
        int32_t *zp = has_asymmetric_comp
                ? reinterpret_cast<int32_t *>(output + offset)
                : nullptr;

        if (has_asymmetric_comp) {
            parallel_nd(G * NB_OC * oc_blksize, [&](dim_t i) { zp[i] = 0; });
        }

#define wei_blk_off(md, g, o, i, d, h, w) \
    (is_1d                  ? (md).blk_off<!w_groups>(g, o, i, w) \
                    : is_3d ? (md).blk_off<!w_groups>(g, o, i, d, h, w) \
                            : (md).blk_off<!w_groups>(g, o, i, h, w))

        parallel_nd(G, NB_OC, [&](dim_t g, dim_t O) {
            for_(dim_t I = 0; I < IC; I++)
            for_(dim_t d = 0; d < D; d++)
            for_(dim_t h = 0; h < H; h++)
            for (dim_t w = 0; w < W; w++) {
                auto i = &input[wei_blk_off(
                        input_d, g, oc_blksize * O, I, d, h, w)];
                auto o = &output[wei_blk_off(output_d, g, O, I, d, h, w)];
                const dim_t oc_block
                        = nstl::min(oc_blksize, OC - O * oc_blksize);
                dim_t _offset = (g * NB_OC + O) * oc_blksize;
                int32_t *zp_ptr = (order_keep && has_asymmetric_comp)
                        ? &zp[_offset]
                        : nullptr;
                const float *src_scales_ptr
                        = &src_scales[src_scales_mask == 0 ? 0 : _offset];
                const float *dst_scales_ptr
                        = &dst_scales[dst_scales_mask == 0 ? 0 : _offset];
                ker(i, o, zp_ptr, src_scales_ptr, dst_scales_ptr, oc_block);
            }
        });

#undef wei_blk_off

        return status::success;
    }
};

/* Asymmetric Blocking */
template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<(utils::one_of(tag_i, format_tag::iwo,
                                           format_tag::oiw, format_tag::wio)
                                          && utils::one_of(tag_o,
                                                  format_tag::OwI16o4i,
                                                  format_tag::OIw16i16o4i))
                        || (utils::one_of(
                                    tag_i, format_tag::goiw, format_tag::wigo)
                                && utils::one_of(tag_o, format_tag::gOwI16o4i,
                                        format_tag::gOIw16i16o4i))
                        || (utils::one_of(tag_i, format_tag::ihwo,
                                    format_tag::hwio, format_tag::oihw)
                                && utils::one_of(tag_o, format_tag::OhwI16o4i,
                                        format_tag::OIhw16i16o4i))
                        || (utils::one_of(
                                    tag_i, format_tag::goihw, format_tag::hwigo)
                                && utils::one_of(tag_o, format_tag::gOhwI16o4i,
                                        format_tag::gOIhw16i16o4i))
                        || (utils::one_of(tag_i, format_tag::idhwo,
                                    format_tag::dhwio, format_tag::oidhw)
                                && utils::one_of(tag_o, format_tag::OdhwI16o4i,
                                        format_tag::OIdhw16i16o4i))
                        || (utils::one_of(tag_i, format_tag::goidhw)
                                && utils::one_of(tag_o, format_tag::gOdhwI16o4i,
                                        format_tag::gOIdhw16i16o4i)),
                spec::conv_req_comp>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace format_tag;
        using namespace data_type;
        using namespace utils;

        if (input_d.has_runtime_dims_or_strides()) return false;

        int src_scales_mask, dst_scales_mask;
        auto status = get_scales_mask(attr, &src_scales_mask, &dst_scales_mask);
        if (status != status::success) return false;
        int scales_mask = std::max(src_scales_mask, dst_scales_mask);

        const bool w_groups = !one_of(tag_o, OwI16o4i, OIw16i16o4i, OhwI16o4i,
                OIhw16i16o4i, OdhwI16o4i, OIdhw16i16o4i);

        // Current formats are only used in jit kernels that natively
        // support s8 instructions, hence, there is no need for signed
        // compensation.
        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;

        const bool req_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        auto mask_ok = [&](bool check, int mask) {
            const int c_mask = 0x1,
                      g_mask = 0x3; // mask for o-channel and ngroups
            return IMPLICATION(check, mask == (w_groups ? g_mask : c_mask));
        };

        return simple_attr_check(attr, true, false)
                && input_d.matches_tag(tag_i) && output_d.matches_tag(tag_o)
                && mask_ok(req_asymmetric_comp,
                        output_d.extra().asymm_compensation_mask)
                && one_of(input_d.data_type(), f32, s8, bf16)
                && IMPLICATION(!w_groups, one_of(scales_mask, 0, 0x1))
                && IMPLICATION(w_groups, one_of(scales_mask, 0, 0x3))
                && output_d.data_type() == s8 && !req_comp;
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace format_tag;

        static constexpr bool w_groups
                = !utils::one_of(tag_o, OwI16o4i, OIw16i16o4i, OhwI16o4i,
                        OIhw16i16o4i, OdhwI16o4i, OIdhw16i16o4i);
        constexpr int is_1d = utils::one_of(
                tag_o, OwI16o4i, gOwI16o4i, OIw16i16o4i, gOIw16i16o4i);
        const bool is_3d = utils::one_of(
                tag_o, OdhwI16o4i, gOdhwI16o4i, OIdhw16i16o4i, gOIdhw16i16o4i);

        constexpr dim_t oc_blksize = 16;
        constexpr dim_t ic_blksize
                = utils::one_of(tag_traits<tag_o>::inner_blks, ib::_16b16a4b,
                          ib::_16c16b4c)
                ? 64
                : utils::one_of(
                          tag_traits<tag_o>::inner_blks, ib::_16a4b, ib::_16b4c)
                ? 4
                : 1;
        assert(ic_blksize != 1);

        const auto &plain_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims
                = order_keep ? output_d.padded_dims() : input_d.padded_dims();

        const dim_t G = w_groups ? dims[0] : 1;
        const dim_t OC = dims[w_groups + 0];
        const dim_t NB_OC = pdims[w_groups + 0] / oc_blksize;
        const dim_t IC = dims[w_groups + 1];
        const dim_t NB_IC = pdims[w_groups + 1] / ic_blksize;

        const dim_t D = is_3d ? dims[2 + w_groups] : 1;
        const dim_t H = is_1d ? 1 : dims[2 + w_groups + is_3d];
        const dim_t W = dims[w_groups + is_3d + 3 - is_1d];

        const bool has_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        float adj_scale
                = (output_d.extra().flags & memory_extra_flags::scale_adjust)
                ? output_d.extra().scale_adjust
                : 1.f;

        // This kernel is used primarily for tensors with multiple inner
        // blocks for which generic zero padding must be used.
        // TODO: apply zero padding inside parallel_nd()
        ctx.zero_pad_output(DNNL_ARG_TO);

        auto ker = [&](const data_t<type_i> *inp, data_t<type_o> *out,
                           int32_t *zp, const float *s, const float *d,
                           const dim_t oc_block, const dim_t ic_block) {
            for_(dim_t ic = 0; ic < ic_block; ++ic)
            for (dim_t oc = 0; oc < oc_block; ++oc) {
                const auto plain_off
                        = oc * plain_d.blocking_desc().strides[w_groups + 0]
                        + ic * plain_d.blocking_desc().strides[w_groups + 1];
                auto index = AB_or_BC_blk_off<tag_traits<tag_o>::inner_blks>(
                        oc, ic);
                out[index] = q10n::qz_b0<data_t<type_i>, data_t<type_o>>()(
                        inp[plain_off], s[oc] * adj_scale * d[oc]);

                if (has_asymmetric_comp) zp[oc] -= (int32_t)(out[index]);
            }
        };

        size_t offset = output_d.size() - output_d.additional_buffer_size();
        int32_t *zp = has_asymmetric_comp
                ? reinterpret_cast<int32_t *>(output + offset)
                : nullptr;

        if (has_asymmetric_comp) {
            parallel_nd(G * NB_OC * oc_blksize, [&](dim_t i) { zp[i] = 0; });
        }

#define wei_blk_off(md, g, o, i, d, h, w) \
    (is_1d                  ? (md).blk_off<!w_groups>(g, o, i, w) \
                    : is_3d ? (md).blk_off<!w_groups>(g, o, i, d, h, w) \
                            : (md).blk_off<!w_groups>(g, o, i, h, w))

        parallel_nd(G, NB_OC, [&](dim_t g, dim_t O) {
            for_(dim_t I = 0; I < NB_IC; I++)
            for_(dim_t d = 0; d < D; d++)
            for_(dim_t h = 0; h < H; h++)
            for (dim_t w = 0; w < W; w++) {
                auto i = &input[wei_blk_off(
                        input_d, g, oc_blksize * O, ic_blksize * I, d, h, w)];
                auto o = &output[wei_blk_off(output_d, g, O, I, d, h, w)];
                const dim_t oc_block
                        = nstl::min(oc_blksize, OC - O * oc_blksize);
                const dim_t ic_block
                        = nstl::min(ic_blksize, IC - I * ic_blksize);
                dim_t _offset = (g * NB_OC + O) * oc_blksize;
                int32_t *zp_ptr = (order_keep && has_asymmetric_comp)
                        ? &zp[_offset]
                        : nullptr;
                const float *src_scales_ptr
                        = &src_scales[src_scales_mask == 0 ? 0 : _offset];
                const float *dst_scales_ptr
                        = &dst_scales[dst_scales_mask == 0 ? 0 : _offset];
                ker(i, o, zp_ptr, src_scales_ptr, dst_scales_ptr, oc_block,
                        ic_block);
            }
        });

#undef wei_blk_off

        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<
                (utils::one_of(tag_i, format_tag::ab, format_tag::ba,
                         format_tag::abc, format_tag::acb)
                        && utils::one_of(tag_o, format_tag::BA16a16b4a,
                                format_tag::BA16a32b4a, format_tag::BA16a48b4a,
                                format_tag::BA16a64b4a, format_tag::aCB16b16c4b,
                                format_tag::aCB16b32c4b,
                                format_tag::aCB16b48c4b,
                                format_tag::aCB16b64c4b)),
                spec::conv_req_comp>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace format_tag;
        using namespace data_type;
        using namespace utils;

        if (input_d.has_runtime_dims_or_strides()) return false;

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool req_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        const auto ndims = input_d.ndims();
        auto mask_ok = [&](bool check, int mask) {
            return IMPLICATION(
                    check, mask == (1 << ndims) - 1 - (1 << (ndims - 2)));
        };

        int src_scales_mask, dst_scales_mask;
        auto status = get_scales_mask(attr, &src_scales_mask, &dst_scales_mask);
        if (status != status::success) return false;
        int scales_mask = std::max(src_scales_mask, dst_scales_mask);
        const size_t D_mask
                = array_product(input_d.dims(), math::ilog2q(scales_mask + 1));

        return simple_attr_check(attr, true, false)
                && input_d.matches_tag(tag_i) && output_d.matches_tag(tag_o)
                && mask_ok(req_comp, output_d.extra().compensation_mask)
                && mask_ok(req_asymmetric_comp,
                        output_d.extra().asymm_compensation_mask)
                && one_of(input_d.data_type(), f32, s8, bf16, f16, f8_e5m2,
                        f8_e4m3)
                && output_d.data_type() == s8 && D_mask == 1;
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace format_tag;

        // {[batch_dim][d0][d1], [batch_dim][d1][d0]} -> [batch_dim][D1][D0][16][D1_blksize][4]
        // 2D: batch_dim - none, d0 <-> a, d1 <-> b
        // 3D: batch_dim <-> a, d0 <-> b, d1 <-> c
        constexpr dim_t D0_blksize = 64;
        constexpr dim_t D1_blksize
                = (utils::one_of(tag_traits<tag_o>::inner_blks, ib::_16a64b4a,
                          ib::_16b64c4b))
                ? 64
                : (utils::one_of(tag_traits<tag_o>::inner_blks, ib::_16a48b4a,
                          ib::_16b48c4b))
                ? 48
                : (utils::one_of(tag_traits<tag_o>::inner_blks, ib::_16a32b4a,
                          ib::_16b32c4b))
                ? 32
                : (utils::one_of(tag_traits<tag_o>::inner_blks, ib::_16a16b4a,
                          ib::_16b16c4b))
                ? 16
                : 1;
        assert(D1_blksize != 1);

        const auto &plain_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto ndims = input_d.ndims();
        const auto &pdims
                = order_keep ? output_d.padded_dims() : input_d.padded_dims();

        const dim_t batch_dim = ndims > 2 ? dims[ndims - 3] : 1;
        const dim_t D0dim = dims[ndims - 2];
        const dim_t NB_D0dim = pdims[ndims - 2] / D0_blksize;
        const dim_t D1dim = dims[ndims - 1];
        const dim_t NB_D1dim = pdims[ndims - 1] / D1_blksize;
        assert(pdims[ndims - 1] == NB_D1dim * D1_blksize);

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool has_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        float adj_scale
                = (output_d.extra().flags & memory_extra_flags::scale_adjust)
                ? output_d.extra().scale_adjust
                : 1.f;

        auto ker = [&](const data_t<type_i> *inp, data_t<type_o> *out,
                           int32_t *cp, int32_t *zp, const float *s,
                           const float *d, const int d0_block,
                           const int d1_block) {
            for (int d0 = 0; d0 < d0_block; ++d0) {
                for (int d1 = 0; d1 < d1_block; ++d1) {
                    const auto plain_off
                            = d0 * plain_d.blocking_desc().strides[ndims - 2]
                            + d1 * plain_d.blocking_desc().strides[ndims - 1];
                    auto index
                            = AB_or_BC_blk_off<tag_traits<tag_o>::inner_blks>(
                                    d0, d1);
                    out[index] = q10n::qz_b0<data_t<type_i>, data_t<type_o>>()(
                            inp[plain_off], s[0] * adj_scale * d[0]);

                    auto o = static_cast<int32_t>(out[index]);
                    if (req_comp) cp[d1] -= (128 * o);
                    if (has_asymmetric_comp) zp[d1] -= o;
                }
                for (int d1 = d1_block; d1 < D1_blksize; ++d1) {
                    auto index
                            = AB_or_BC_blk_off<tag_traits<tag_o>::inner_blks>(
                                    d0, d1);
                    out[index] = q10n::qz_b0<data_t<type_i>, data_t<type_o>>()(
                            0, s[0] * adj_scale * d[0]);
                }
            }

            for_(int d0 = d0_block; d0 < D0_blksize; ++d0)
            for (int d1 = 0; d1 < D1_blksize; ++d1) {
                auto index = AB_or_BC_blk_off<tag_traits<tag_o>::inner_blks>(
                        d0, d1);
                out[index] = q10n::qz_b0<data_t<type_i>, data_t<type_o>>()(
                        0, s[0] * adj_scale * d[0]);
            }
        };

        const auto w_d = order_keep ? output_d : input_d;
        size_t offset = w_d.size() - w_d.additional_buffer_size();
        size_t comp_size = output_d.additional_buffer_size(
                memory_extra_flags::compensation_conv_s8s8);
        size_t zp_offset = offset + (req_comp ? comp_size : 0);
        int32_t *cp = req_comp ? reinterpret_cast<int32_t *>(output + offset)
                               : nullptr;
        int32_t *zp = has_asymmetric_comp
                ? reinterpret_cast<int32_t *>(output + zp_offset)
                : nullptr;

        if (has_asymmetric_comp || req_comp) {
            parallel_nd(batch_dim * NB_D1dim * D1_blksize, [&](dim_t i) {
                if (req_comp) cp[i] = 0;
                if (has_asymmetric_comp) zp[i] = 0;
            });
        }

#define get_blk_off(md, batch, d0, d1) \
    (ndims == 3 ? (md).blk_off((batch), (d0), (d1)) : (md).blk_off((d0), (d1)))

        parallel_nd(batch_dim, NB_D1dim, [&](dim_t batch, dim_t D1) {
            for (int D0 = 0; D0 < NB_D0dim; D0++) {
                auto i = &input[get_blk_off(
                        input_d, batch, D0_blksize * D0, D1_blksize * D1)];
                auto o = &output[get_blk_off(output_d, batch, D0, D1)];
                const dim_t d0_block
                        = nstl::min(D0_blksize, D0dim - D0 * D0_blksize);
                const dim_t d1_block
                        = nstl::min(D1_blksize, D1dim - D1 * D1_blksize);
                dim_t _offset = batch * NB_D1dim * D1_blksize + D1 * D1_blksize;
                int32_t *zp_ptr = (order_keep && has_asymmetric_comp)
                        ? &zp[_offset]
                        : nullptr;
                const float *src_scales_ptr
                        = &src_scales[src_scales_mask == 0 ? 0 : _offset];
                const float *dst_scales_ptr
                        = &dst_scales[dst_scales_mask == 0 ? 0 : _offset];
                ker(i, o, (order_keep && req_comp) ? &cp[_offset] : nullptr,
                        zp_ptr, src_scales_ptr, dst_scales_ptr, d0_block,
                        d1_block);
            }
        });

#undef get_blk_off

        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<
                (utils::one_of(tag_i, format_tag::goiw, format_tag::wigo)
                        && utils::one_of(tag_o, format_tag::Goiw16g,
                                format_tag::Goiw8g, format_tag::Goiw4g))
                        || (utils::one_of(
                                    tag_i, format_tag::goihw, format_tag::hwigo)
                                && utils::one_of(tag_o, format_tag::Goihw16g,
                                        format_tag::Goihw8g,
                                        format_tag::Goihw4g)),
                spec::conv_req_comp>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace data_type;
        using namespace utils;

        if (input_d.has_runtime_dims_or_strides()) return false;

        int src_scales_mask, dst_scales_mask;
        auto status = get_scales_mask(attr, &src_scales_mask, &dst_scales_mask);
        if (status != status::success) return false;
        int scales_mask = std::max(src_scales_mask, dst_scales_mask);

        const dim_t g = input_d.dims()[0];
        const dim_t oc = input_d.dims()[1];
        const dim_t ic = input_d.dims()[2];

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool req_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;
        int s8s8_comp_mask = output_d.extra().compensation_mask;
        int zp_comp_mask = output_d.extra().asymm_compensation_mask;
        int comp_mask = std::max(s8s8_comp_mask, zp_comp_mask);

        const size_t D_mask
                = array_product(input_d.dims(), math::ilog2q(comp_mask + 1));

        return order_keep && oc == 1 && ic == 1 // depth-wise case
                && simple_attr_check(attr, true, false)
                && (req_comp || req_asymmetric_comp)
                && IMPLICATION(req_comp && req_asymmetric_comp,
                        output_d.extra().compensation_mask
                                == output_d.extra().asymm_compensation_mask)
                && input_d.matches_tag(tag_i) && output_d.matches_tag(tag_o)
                && IMPLICATION(
                        req_comp, one_of(D_mask, (size_t)1, (size_t)g * oc))
                && one_of(scales_mask, 0, 0x3)
                && one_of(input_d.data_type(), f32, s8, bf16)
                && output_d.data_type() == s8;
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        constexpr bool is_1d
                = utils::one_of(tag_i, format_tag::goiw, format_tag::wigo);
        constexpr dim_t blksize
                = utils::one_of(tag_o, format_tag::Goihw4g, format_tag::Goiw4g)
                ? 4
                : utils::one_of(tag_o, format_tag::Goihw8g, format_tag::Goiw8g)
                ? 8
                : 16;

        const auto &dims = input_d.dims();
        const auto &pdims = output_d.padded_dims();
        const dim_t G = dims[0];
        const dim_t Gp = pdims[0];
        const dim_t OC = dims[1];
        const dim_t IC = dims[2];
        const dim_t H = is_1d ? 1 : dims[3];
        const dim_t W = dims[4 - is_1d];
        const bool zero_padding_needed = !output_d.is_dense();

        const bool req_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8;
        const bool has_asymmetric_comp = output_d.extra().flags
                & memory_extra_flags::compensation_conv_asymmetric_src;

        assert(req_comp || has_asymmetric_comp);

        float adj_scale
                = (output_d.extra().flags & memory_extra_flags::scale_adjust)
                ? output_d.extra().scale_adjust
                : 1.f;

        auto ker_out = [&](const data_t<type_i> *inp, data_t<type_o> *out,
                               const float *src_scales, const float *dst_scales,
                               const dim_t g_block) {
            PRAGMA_OMP_SIMD()
            for (dim_t g = 0; g < g_block; g++) {
                const auto i_off = g * input_d.blocking_desc().strides[0];
                const float src_scale
                        = src_scales[src_scales_mask == 0 ? 0 : g * OC];
                const float dst_scale
                        = dst_scales[dst_scales_mask == 0 ? 0 : g * OC];
                out[g] = q10n::qz_b0<data_t<type_i>, data_t<type_o>>()(
                        inp[i_off], src_scale * adj_scale * dst_scale);
            }
        };

        /* Note: having separate kernels for s8 and zero-point fixes a
         * compiler-generated bug which results in seg-fault. */
        auto ker_s8 = [&](const data_t<type_o> *out, int32_t *cp,
                              const dim_t g_block) {
            PRAGMA_OMP_SIMD()
            for (dim_t g = 0; g < g_block; g++) {
                cp[g * OC] -= 128 * (int32_t)(out[g]);
            }
        };
        auto ker_zp = [&](const data_t<type_o> *out, int32_t *zp,
                              const dim_t g_block) {
            PRAGMA_OMP_SIMD()
            for (dim_t g = 0; g < g_block; g++) {
                zp[g * OC] -= (int32_t)(out[g]);
            }
        };

        size_t offset = output_d.size() - output_d.additional_buffer_size();
        size_t comp_size = output_d.additional_buffer_size(
                memory_extra_flags::compensation_conv_s8s8);
        size_t zp_offset = offset + (req_comp ? comp_size : 0);
        int32_t *cp = req_comp ? reinterpret_cast<int32_t *>(output + offset)
                               : nullptr;
        int32_t *zp = has_asymmetric_comp
                ? reinterpret_cast<int32_t *>(output + zp_offset)
                : nullptr;

        parallel_nd((Gp / blksize) * OC, [&](dim_t ib) {
            PRAGMA_OMP_SIMD()
            for (dim_t i = 0; i < blksize; i++) {
                if (req_comp) cp[ib * blksize + i] = 0;
                if (has_asymmetric_comp) zp[ib * blksize + i] = 0;
            }
        });

#define wei_blk_off(md, g, o, i, h, w) \
    (is_1d ? (md).blk_off(g, o, i, w) : (md).blk_off(g, o, i, h, w))

        parallel_nd(Gp / blksize, OC, [&](dim_t gb, dim_t O) {
            for (dim_t I = 0; I < IC; I++) {
                for_(dim_t h = 0; h < H; h++)
                for (dim_t w = 0; w < W; w++) {
                    const dim_t g_block = nstl::min(G - gb * blksize, blksize);
                    const auto inp = &input[wei_blk_off(
                            input_d, gb * blksize, O, I, h, w)];
                    const auto out
                            = &output[wei_blk_off(output_d, gb, O, I, h, w)];
                    dim_t offset = gb * blksize + O;
                    const float *src_scales_ptr
                            = &src_scales[src_scales_mask == 0 ? 0 : offset];
                    const float *dst_scales_ptr
                            = &dst_scales[dst_scales_mask == 0 ? 0 : offset];

                    ker_out(inp, out, src_scales_ptr, dst_scales_ptr, g_block);
                    if (req_comp) ker_s8(out, &cp[offset], g_block);
                    if (has_asymmetric_comp) ker_zp(out, &zp[offset], g_block);

                    if (zero_padding_needed) {
                        PRAGMA_OMP_SIMD()
                        for (int off = g_block; off < blksize; off++)
                            out[off] = 0;
                    }
                }
            }
        });

#undef wei_blk_off

        return status::success;
    }
};

/* bf16 reorders */
template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<(
                (tag_i == format_tag::goihw || tag_i == format_tag::oihw)
                && (tag_o == format_tag::gOIhw16i16o
                        || tag_o == format_tag::OIhw16i16o
                        || tag_o == format_tag::gOIhw8i16o2i
                        || tag_o == format_tag::OIhw8i16o2i
                        || tag_o == format_tag::gOIhw8o16i2o
                        || tag_o == format_tag::OIhw8o16i2o
                        || tag_o == format_tag::gIOhw8o16i2o
                        || tag_o == format_tag::IOhw8o16i2o)
                && type_i == data_type::f32
                && type_o == data_type::bf16)>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace data_type;

        if (input_d.has_runtime_dims_or_strides()) return false;

        return order_keep && input_d.matches_tag(tag_i)
                && output_d.matches_tag(tag_o) && input_d.data_type() == f32
                && output_d.data_type() == bf16 && attr->has_default_values();
    }

    static size_t get_scratchpad_size(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        const dim_t blksize = 16;
        return sizeof(float) * blksize * blksize * dnnl_get_max_threads();
    }

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace format_tag;

        static constexpr bool w_groups = tag_i == goihw;
        const dim_t blksize = 16;
        const int sblk = 2;

        const auto &plain_d = input_d;
        const auto &dims = input_d.dims();
        const auto &pdims = output_d.padded_dims();

        const dim_t G = w_groups ? dims[0] : 1;
        const dim_t OC = dims[w_groups + 0];
        const dim_t NB_OC = pdims[w_groups + 0] / blksize;
        const dim_t IC = dims[w_groups + 1];
        const dim_t NB_IC = pdims[w_groups + 1] / blksize;
        const dim_t H = dims[w_groups + 2];
        const dim_t W = dims[w_groups + 3];

        const size_t wsp_size = blksize * blksize;
        float *wspace = scratchpad.template get<float>(
                memory_tracking::names::key_reorder_space);

        auto index = [&](dim_t ic, dim_t oc) -> dim_t {
            if (utils::one_of(tag_o, gOIhw16i16o, OIhw16i16o))
                return (ic * blksize + oc);
            else if (utils::one_of(tag_o, gOIhw8i16o2i, OIhw8i16o2i))
                return ((ic / sblk) * blksize * sblk + sblk * oc + ic % sblk);
            else if (utils::one_of(tag_o, gOIhw8o16i2o, gIOhw8o16i2o,
                             OIhw8o16i2o, IOhw8o16i2o))
                return ((oc / sblk) * blksize * sblk + sblk * ic + oc % sblk);
            else
                assert(!"Invalid weight format");
            return dim_t(0);
        };

        auto ker = [&](const data_t<type_i> *inp, data_t<type_i> *out,
                           const dim_t curr_oc_block, const dim_t oc_block,
                           const dim_t curr_ic_block, const dim_t ic_block) {
            dim_t ic = 0;
            for (ic = 0; ic < curr_ic_block; ++ic) {
                dim_t oc = 0;
                for (oc = 0; oc < curr_oc_block; ++oc) {
                    const auto plain_off
                            = oc * plain_d.blocking_desc().strides[w_groups + 0]
                            + ic
                                    * plain_d.blocking_desc()
                                              .strides[w_groups + 1];
                    out[index(ic, oc)] = inp[plain_off];
                }
                for (/* continue */; oc < oc_block; ++oc) {
                    out[index(ic, oc)] = (data_t<type_i>)0;
                }
            }
            for (/* continue */; ic < ic_block; ++ic) {
                for (dim_t oc = 0; oc < oc_block; ++oc) {
                    out[index(ic, oc)] = (data_t<type_i>)0;
                }
            }
        };

        constexpr int i_mult = blksize;
        constexpr int o_mult = 1;

        parallel_nd_ext(0, G, NB_OC, NB_IC, H, W,
                [&](int ithr, int, dim_t g, dim_t O, dim_t I, dim_t h,
                        dim_t w) {
                    float *_wspace = wspace + wsp_size * ithr;
                    auto i = &input[input_d.blk_off<!w_groups>(
                            g, i_mult * O, i_mult * I, h, w)];
                    auto o = &output[output_d.blk_off<!w_groups>(
                            g, o_mult * O, o_mult * I, h, w)];
                    const dim_t oc_block = nstl::min(blksize, OC - O * blksize);
                    const dim_t ic_block = nstl::min(blksize, IC - I * blksize);
                    ker(i, _wspace, oc_block, blksize, ic_block, blksize);
                    cvt_float_to_bfloat16(o, _wspace, wsp_size);
                });

        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<(tag_i == format_tag::nchw
                                          && tag_o == format_tag::nChw16c)
                && type_i == data_type::f32
                && type_o == data_type::bf16>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        using namespace data_type;

        if (input_d.has_runtime_dims_or_strides()) return false;

        return input_d.matches_tag(tag_i) && output_d.matches_tag(tag_o)
                && input_d.data_type() == f32 && output_d.data_type() == bf16
                && attr->has_default_values();
    }

    static size_t get_scratchpad_size(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d) {
        const size_t blksize = 16;
        const size_t W = input_d.dims()[3];
        return sizeof(float) * blksize * W * dnnl_get_max_threads();
    }

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        const dim_t blksize = 16;

        const auto &flat_d = input_d;
        const auto &dims = input_d.dims();
        const auto &pdims = output_d.padded_dims();

        const dim_t C = dims[1];
        const dim_t H = dims[2];
        const dim_t W = dims[3];

        const dim_t wsp_size = W * blksize;
        float *wspace = scratchpad.template get<float>(
                memory_tracking::names::key_reorder_space);

        auto ker = [&](const data_t<type_i> *i, data_t<type_i> *o,
                           const dim_t curr_c_block, const dim_t c_block) {
            for (dim_t w = 0; w < W; ++w) {
                dim_t c = 0;
                for (c = 0; c < curr_c_block; ++c) {
                    const ptrdiff_t flat_off = 0
                            + c * flat_d.blocking_desc().strides[1]
                            + w * flat_d.blocking_desc().strides[3];
                    o[w * blksize + c] = i[flat_off];
                }
                for (/* continue */; c < c_block; ++c) {
                    o[w * blksize + c] = (data_t<type_i>)0;
                }
            }
        };

        constexpr int i_c_mult = blksize;
        constexpr int o_c_mult = 1;

        parallel_nd_ext(0, dims[0], pdims[1] / blksize, H,
                [&](int ithr, int, dim_t n, dim_t nb_c, dim_t h) {
                    float *_wspace = wspace + wsp_size * ithr;
                    auto i = &input[input_d.blk_off(n, i_c_mult * nb_c, h)];
                    auto o = &output[output_d.blk_off(n, o_c_mult * nb_c, h)];
                    const dim_t c_block
                            = nstl::min(blksize, C - nb_c * blksize);
                    ker(i, _wspace, c_block, blksize);
                    cvt_float_to_bfloat16(o, _wspace, wsp_size);
                });

        return status::success;
    }
};

/* reorders with tail support */

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<false
                || (utils::one_of(
                            tag_i, format_tag::nCdhw4c, format_tag::nCdhw8c)
                        && tag_o == format_tag::nCdhw16c)
                || (utils::one_of(tag_i, format_tag::nChw4c, format_tag::nChw8c)
                        && tag_o == format_tag::nChw16c)
                || (utils::one_of(tag_i, format_tag::nCw4c, format_tag::nCw8c)
                        && tag_o == format_tag::nCw16c)>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        return simple_fmt_check(order_keep, tag_i, tag_o, input_d, output_d)
                && simple_attr_check(attr, false, true);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace format_tag;

        constexpr int is_1d = utils::one_of(tag_i, nCw4c, nCw8c);
        constexpr int is_3d = utils::one_of(tag_i, nCdhw4c, nCdhw8c);

        constexpr dim_t blksize_i
                = tag_traits<tag_i>::inner_blks == ib::_4b ? 4 : 8;
        constexpr dim_t blksize_16 = 16;

        constexpr dim_t ic_mult = order_keep ? blksize_16 / blksize_i : 1;
        constexpr dim_t oc_mult = order_keep ? 1 : blksize_16 / blksize_i;

        const auto &dims = input_d.dims();
        const auto &pdims
                = order_keep ? output_d.padded_dims() : input_d.padded_dims();

        const auto &d_i = order_keep ? input_d : output_d;
        const auto stride_C_in_blk_i = d_i.blocking_desc().strides[1];

        const dim_t C = dims[1];
        const dim_t D = is_3d ? dims[2] : 1;
        const dim_t H = is_1d ? 1 : dims[2 + is_3d];
        const dim_t W = dims[3 + is_3d - is_1d];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
                           const int block) {
            const int nb = utils::div_up(block, blksize_i);
            if (alpha == 1.0 && beta == 0.0) {
                for (int b = 0; b < nb; ++b) {
                    const ptrdiff_t i_off
                            = b * (order_keep ? stride_C_in_blk_i : blksize_i);
                    const ptrdiff_t o_off
                            = b * (order_keep ? blksize_i : stride_C_in_blk_i);
                    const int block_i
                            = nstl::min(blksize_i, block - b * blksize_i);
                    for (int c = 0; c < block_i; ++c) {
                        o[o_off + c] = _qz_a1b0<type_i, type_o>()(i[i_off + c]);
                    }
                    if (b + 1 == nb) {
                        // zero padding
                        const auto pad_size = order_keep
                                ? blksize_16 - ((nb - 1) * blksize_i)
                                : blksize_i;
                        const auto pad_start = block_i + o_off;
                        const auto pad_end = pad_size + o_off;
                        PRAGMA_OMP_SIMD()
                        for (int i = pad_start; i < pad_end; i++) {
                            o[i] = 0;
                        }
                    }
                }
            } else {
                for (int b = 0; b < nb; ++b) {
                    const ptrdiff_t i_off
                            = b * (order_keep ? stride_C_in_blk_i : blksize_i);
                    const ptrdiff_t o_off
                            = b * (order_keep ? blksize_i : stride_C_in_blk_i);
                    const int block_i
                            = nstl::min(blksize_i, block - b * blksize_i);
                    for (int c = 0; c < block_i; ++c) {
                        o[o_off + c] = _qz<type_i, type_o>()(
                                i[i_off + c], o[o_off + c], alpha, beta);
                    }
                    if (b + 1 == nb) {
                        // zero padding
                        const auto pad_size = order_keep
                                ? blksize_16 - ((nb - 1) * blksize_i)
                                : blksize_i;
                        const auto pad_start = block_i + o_off;
                        const auto pad_end = pad_size + o_off;
                        PRAGMA_OMP_SIMD()
                        for (int i = pad_start; i < pad_end; i++) {
                            o[i] = 0;
                        }
                    }
                }
            }
        };

#define data_blk_off(md, n, c, d, h, w) \
    (is_1d                  ? (md).blk_off(n, c, w) \
                    : is_3d ? (md).blk_off(n, c, d, h, w) \
                            : (md).blk_off(n, c, h, w))

        parallel_nd(dims[0], pdims[1] / blksize_16, D, H, W,
                [&](dim_t n, dim_t nb_c, dim_t d, dim_t h, dim_t w) {
                    auto i = &input[data_blk_off(
                            input_d, n, ic_mult * nb_c, d, h, w)];
                    auto o = &output[data_blk_off(
                            output_d, n, oc_mult * nb_c, d, h, w)];
                    const int block
                            = nstl::min(blksize_16, C - nb_c * blksize_16);
                    ker(i, o, block);
                });

#undef data_blk_off

        return status::success;
    }
};

#define PLAIN_TO_BLOCKED_IS_APPLICABLE() \
    static bool is_applicable(const memory_desc_wrapper &input_d, \
            const memory_desc_wrapper &output_d, \
            const primitive_attr_t *attr) { \
        return !input_d.has_runtime_dims_or_strides() \
                && simple_attr_check(attr, false, true) \
                && (order_keep ? output_d.matches_tag(tag_o) \
                                        && input_d.is_plain() \
                               : input_d.matches_tag(tag_o) \
                                        && output_d.is_plain()); \
    }

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                && (tag_traits<tag_o>::block_dims == bd::_A
                        || tag_traits<tag_o>::block_dims == bd::_B)
                && tag_traits<tag_o>::ndims >= 3
                && tag_traits<tag_o>::ndims <= 6>::type> {
    PLAIN_TO_BLOCKED_IS_APPLICABLE();

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        const auto &flat_d = order_keep ? input_d : output_d;
        const auto &block_d = order_keep ? output_d : input_d;
        const dims_t &dims = input_d.dims();
        const dims_t &pdims = block_d.padded_dims();

        const int ndims = tag_traits<tag_o>::ndims;
        const int blk_idx = tag_traits<tag_o>::block_dims == bd::_A ? 0 : 1;

        const dim_t H0 = dims[0];
        const dim_t H1 = dims[1];
        const dim_t M0 = ndims == 6 ? dims[ndims - 4] : 1;
        const dim_t M1 = ndims >= 5 ? dims[ndims - 3] : 1;
        const dim_t M2 = ndims >= 4 ? dims[ndims - 2] : 1;
        const dim_t L = dims[ndims - 1];
        const dim_t l_blk_stride = block_d.blocking_desc().strides[ndims - 1];
        const dim_t l_flat_stride = flat_d.blocking_desc().strides[ndims - 1];
        const dim_t blk_flat_stride = flat_d.blocking_desc().strides[blk_idx];
        using namespace data_type;
        using namespace utils;

        dim_t blksize = -1;
        switch (tag_traits<tag_o>::inner_blks) {
            case ib::_4a:
            case ib::_4b: blksize = 4; break;
            case ib::_8a:
            case ib::_8b: blksize = 8; break;
            default: blksize = 16;
        }

        constexpr bool f32bf16
                = one_of(type_i, f32, bf16) && one_of(type_o, f32, bf16);

        auto wrap_qz_a1b0 = [=](data_t<type_o> &out, data_t<type_i> inp) {
            if (f32bf16)
                out = inp;
            else
                out = _qz_a1b0<type_i, type_o>()(inp);
        };

        auto wrap_qz = [=](data_t<type_o> &out, data_t<type_i> inp, float alpha,
                               float beta) {
            if (f32bf16)
                out = alpha * inp + (beta ? beta * out : 0);
            else
                out = _qz<type_i, type_o>()(inp, out, alpha, beta);
        };

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o, int block) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int l = 0; l < L; ++l) {
                    for (int blk = 0; blk < block; ++blk) {
                        const dim_t flat_off
                                = blk * blk_flat_stride + l * l_flat_stride;
                        const dim_t blk_offset = l * l_blk_stride + blk;
                        if (order_keep) {
                            wrap_qz_a1b0(o[blk_offset], i[flat_off]);
                        } else {
                            wrap_qz_a1b0(o[flat_off], i[blk_offset]);
                        }
                    }
                    if (order_keep) {
                        // zero padding
                        const auto pad_start = block + l * l_blk_stride;
                        const auto pad_end = blksize + l * l_blk_stride;
                        PRAGMA_OMP_SIMD()
                        for (int i = pad_start; i < pad_end; ++i) {
                            o[i] = 0;
                        }
                    }
                }
            } else {
                for (int l = 0; l < L; ++l) {
                    for (int blk = 0; blk < block; ++blk) {
                        const dim_t flat_off
                                = blk * blk_flat_stride + l * l_flat_stride;
                        const dim_t blk_offset = l * l_blk_stride + blk;
                        if (order_keep)
                            wrap_qz(o[blk_offset], i[flat_off], alpha, beta);
                        else
                            wrap_qz(o[flat_off], i[blk_offset], alpha, beta);
                    }
                    if (order_keep) {
                        // zero padding
                        const auto pad_start = block + l * l_blk_stride;
                        const auto pad_end = blksize + l * l_blk_stride;
                        PRAGMA_OMP_SIMD()
                        for (int i = pad_start; i < pad_end; ++i) {
                            o[i] = 0;
                        }
                    }
                }
            }
        };

#define off(md, h0, h1, m0, m1, m2) \
    (ndims >= 6                  ? (md).blk_off(h0, h1, m0, m1, m2) \
                    : ndims >= 5 ? (md).blk_off(h0, h1, m1, m2) \
                    : ndims >= 4 ? (md).blk_off(h0, h1, m2) \
                                 : /* ndims >= 3 ? */ (md).blk_off(h0, h1))

        const int i_mult = order_keep ? blksize : 1;
        const int o_mult = order_keep ? 1 : blksize;

        if (blk_idx == 0) {
            const dim_t BH0 = pdims[0] / blksize;
            parallel_nd(BH0, H1, M0, M1, M2,
                    [&](dim_t bh0, dim_t h1, dim_t m0, dim_t m1, dim_t m2) {
                        auto i = &input[off(
                                input_d, bh0 * i_mult, h1, m0, m1, m2)];
                        auto o = &output[off(
                                output_d, bh0 * o_mult, h1, m0, m1, m2)];
                        const int block
                                = nstl::min<int>(blksize, H0 - bh0 * blksize);
                        ker(i, o, block);
                    });
        } else if (blk_idx == 1) {
            const dim_t BH1 = pdims[1] / blksize;
            parallel_nd(H0, BH1, M0, M1, M2,
                    [&](dim_t h0, dim_t bh1, dim_t m0, dim_t m1, dim_t m2) {
                        auto i = &input[off(
                                input_d, h0, bh1 * i_mult, m0, m1, m2)];
                        auto o = &output[off(
                                output_d, h0, bh1 * o_mult, m0, m1, m2)];
                        const int block
                                = nstl::min<int>(blksize, H1 - bh1 * blksize);
                        ker(i, o, block);
                    });
        } else {
            assert(!"unimplemented");
        }

#undef off

        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                && (tag_traits<tag_o>::block_dims == bd::_AB
                        || tag_traits<tag_o>::block_dims == bd::_BC)
                && IMPLICATION(tag_traits<tag_o>::block_dims == bd::_AB,
                        tag_traits<tag_o>::ndims >= 3
                                && tag_traits<tag_o>::ndims <= 5)
                && IMPLICATION(tag_traits<tag_o>::block_dims == bd::_BC,
                        tag_traits<tag_o>::ndims >= 4
                                && tag_traits<tag_o>::ndims <= 6)>::type> {
    PLAIN_TO_BLOCKED_IS_APPLICABLE();

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        const auto &flat_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims
                = order_keep ? output_d.padded_dims() : input_d.padded_dims();

        constexpr int ndims = tag_traits<tag_o>::ndims;

        static constexpr bool with_g = tag_traits<tag_o>::block_dims == bd::_BC;
        const dim_t G = with_g ? dims[0] : 1;

        const dim_t H0 = dims[0 + with_g];
        const dim_t H1 = dims[1 + with_g];

        const dim_t M0 = ndims >= 5 + with_g ? dims[ndims - 3] : 1;
        const dim_t M1 = ndims >= 4 + with_g ? dims[ndims - 2] : 1;
        const dim_t M2 = ndims >= 3 + with_g ? dims[ndims - 1] : 1;

        const dim_t h0_flat_stride = flat_d.blocking_desc().strides[with_g + 0];
        const dim_t h1_flat_stride = flat_d.blocking_desc().strides[with_g + 1];
        using namespace data_type;
        using namespace utils;

        dim_t blksize_0 = -1;
        dim_t blksize_1 = -1;
        switch (tag_traits<tag_o>::inner_blks) {
            case ib::_4b4a:
            case ib::_4b4c:
            case ib::_4c4b:
                blksize_0 = 4;
                blksize_1 = 4;
                break;
            case ib::_8a8b:
            case ib::_8b8a:
            case ib::_8b8c:
            case ib::_8c8b:
            case ib::_2c8b4c:
                blksize_0 = 8;
                blksize_1 = 8;
                break;
            case ib::_16a16b:
            case ib::_16b16a:
            case ib::_16b16c:
            case ib::_16c16b:
            case ib::_8a16b2a:
            case ib::_4b16a4b:
            case ib::_8b16a2b:
            case ib::_8b16c2b:
            case ib::_4c16b4c:
            case ib::_8c16b2c:
                blksize_0 = 16;
                blksize_1 = 16;
                break;
            default: blksize_0 = -1; blksize_1 = -1;
        }

        const dim_t NB_H0 = pdims[0 + with_g] / blksize_0;
        const dim_t NB_H1 = pdims[1 + with_g] / blksize_1;

        constexpr bool f32bf16
                = one_of(type_i, f32, bf16) && one_of(type_o, f32, bf16);

        auto wrap_qz_a1b0 = [=](data_t<type_o> &out, data_t<type_i> inp) {
            if (f32bf16)
                out = inp;
            else
                out = _qz_a1b0<type_i, type_o>()(inp);
        };

        auto wrap_qz = [=](data_t<type_o> &out, data_t<type_i> inp, float alpha,
                               float beta) {
            if (f32bf16)
                out = alpha * inp + (beta ? beta * out : 0);
            else
                out = _qz<type_i, type_o>()(inp, out, alpha, beta);
        };

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
                           const int block_h0, const int block_h1) {
#define blk_off AB_or_BC_blk_off<tag_traits<tag_o>::inner_blks>
            if (alpha == 1.0 && beta == 0.0) {
                for (int h0 = 0; h0 < block_h0; ++h0) {
                    for (int h1 = 0; h1 < block_h1; ++h1) {
                        const dim_t flat_off
                                = h0 * h0_flat_stride + h1 * h1_flat_stride;
                        if (order_keep)
                            wrap_qz_a1b0(o[blk_off(h0, h1)], i[flat_off]);
                        else
                            wrap_qz_a1b0(o[flat_off], i[blk_off(h0, h1)]);
                    }
                    if (order_keep && block_h1 < blksize_1) {
                        // zero padding
                        PRAGMA_OMP_SIMD()
                        for (int h1 = block_h1; h1 < blksize_1; h1++) {
                            o[blk_off(h0, h1)] = 0;
                        }
                    }
                }
                if (order_keep && block_h0 < blksize_0) {
                    // zero padding
                    for (int h0 = block_h0; h0 < blksize_0; h0++) {
                        PRAGMA_OMP_SIMD()
                        for (int h1 = 0; h1 < blksize_1; ++h1) {
                            o[blk_off(h0, h1)] = 0;
                        }
                    }
                }
            } else {
                for (int h0 = 0; h0 < block_h0; ++h0) {
                    for (int h1 = 0; h1 < block_h1; ++h1) {
                        const dim_t flat_off
                                = h0 * h0_flat_stride + h1 * h1_flat_stride;
                        if (order_keep)
                            wrap_qz(o[blk_off(h0, h1)], i[flat_off], alpha,
                                    beta);
                        else
                            wrap_qz(o[flat_off], i[blk_off(h0, h1)], alpha,
                                    beta);
                    }
                    if (order_keep && block_h1 < blksize_1) {
                        // zero padding
                        PRAGMA_OMP_SIMD()
                        for (int h1 = block_h1; h1 < blksize_1; h1++) {
                            o[blk_off(h0, h1)] = 0;
                        }
                    }
                }
                if (order_keep && block_h0 < blksize_0) {
                    // zero padding
                    for (int h0 = block_h0; h0 < blksize_0; h0++) {
                        PRAGMA_OMP_SIMD()
                        for (int h1 = 0; h1 < blksize_1; ++h1) {
                            o[blk_off(h0, h1)] = 0;
                        }
                    }
                }
            }

#undef blk_off
        };

        const int i_mult_0 = order_keep ? blksize_0 : 1;
        const int o_mult_0 = order_keep ? 1 : blksize_0;

        const int i_mult_1 = order_keep ? blksize_1 : 1;
        const int o_mult_1 = order_keep ? 1 : blksize_1;

#define off(md, g, h0, h1, m0, m1, m2) \
    (ndims >= 5 + with_g ? (md).blk_off<!with_g>(g, h0, h1, m0, m1, m2) \
                    : ndims >= 4 + with_g \
                    ? (md).blk_off<!with_g>(g, h0, h1, m1, m2) \
                    : /* ndims >= 3 + with_g ? */ (md).blk_off<!with_g>( \
                            g, h0, h1, m2))

        parallel_nd(G, NB_H0, NB_H1, M0, M1, M2,
                [&](dim_t g, dim_t nb_h0, dim_t nb_h1, dim_t m0, dim_t m1,
                        dim_t m2) {
                    auto i = &input[off(input_d, g, i_mult_0 * nb_h0,
                            i_mult_1 * nb_h1, m0, m1, m2)];
                    auto o = &output[off(output_d, g, o_mult_0 * nb_h0,
                            o_mult_1 * nb_h1, m0, m1, m2)];
                    const int block_h0
                            = nstl::min<int>(blksize_0, H0 - nb_h0 * blksize_0);
                    const int block_h1
                            = nstl::min<int>(blksize_1, H1 - nb_h1 * blksize_1);
                    ker(i, o, block_h0, block_h1);
                });

#undef off

        return status::success;
    }
};

/* generic and direct-copy reorders */

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                        && tag_o == format_tag::any
                        && order_keep == fmt_order::any,
                spec::direct_copy>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        return !input_d.has_runtime_dims_or_strides()
                && input_d.similar_to(output_d, true, false, 0)
                && input_d.is_dense() && output_d.is_dense()
                && simple_attr_check(attr, false, true);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const size_t nelems = input_d.nelems();

        constexpr int block_size = 16;
        const auto num_blocks = nelems / block_size;
        const auto rem_elems = nelems % block_size;

        parallel(0, [&](const int ithr, const int nthr) {
            size_t start {0}, end {0};
            balance211(num_blocks, nthr, ithr, start, end);
            start = start * block_size;
            end = end * block_size;

            if (alpha == 1.0 && beta == 0.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = q10n::qz_a1b0<data_t<type_i>, data_t<type_o>>()(
                            input[e]);
                }
            } else if (alpha == 1.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = q10n::qz_a1<data_t<type_i>, data_t<type_o>>()(
                            input[e], output[e], beta);
                }
            } else if (beta == 0.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = q10n::qz_b0<data_t<type_i>, data_t<type_o>>()(
                            input[e], alpha);
                }
            } else {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = q10n::qz<data_t<type_i>, data_t<type_o>>()(
                            input[e], output[e], alpha, beta);
                }
            }

            if (rem_elems != 0 && ithr == nthr - 1) {
                if (alpha == 1.0 && beta == 0.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = q10n::qz_a1b0<data_t<type_i>,
                                data_t<type_o>>()(input[e]);
                    }
                } else if (alpha == 1.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e]
                                = q10n::qz_a1<data_t<type_i>, data_t<type_o>>()(
                                        input[e], output[e], beta);
                    }
                } else if (beta == 0.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e]
                                = q10n::qz_b0<data_t<type_i>, data_t<type_o>>()(
                                        input[e], alpha);
                    }
                } else {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = q10n::qz<data_t<type_i>, data_t<type_o>>()(
                                input[e], output[e], alpha, beta);
                    }
                }
            }
        });
        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                        && tag_o == format_tag::any && type_i == dnnl_f32
                        && utils::one_of(type_o, dnnl_s4, dnnl_u4),
                spec::reference>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        return !input_d.has_runtime_dims_or_strides() && input_d.is_dense()
                && output_d.is_dense()
                && output_d.strides()[output_d.ndims() - 1] == 1
                && simple_attr_check(attr, false, true);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace utils;

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        // To avoid clashes between threads each byte (or 2 elements)
        // is handled by a single thread
        const dim_t work_amount = input_d.nelems() / 2;

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start {0}, end {0};
            balance211(work_amount, nthr, ithr, start, end);
            PRAGMA_OMP_SIMD()
            for_(dim_t idx = start; idx < end; idx++)
            for (int i = 0; i < 2; ++i) {
                const auto i_off = input_d.off_l(2 * idx + i);
                const auto o_off = output_d.off_l(2 * idx + i);
                const auto shift = i % 2 ? int4_extract_t::high_half
                                         : int4_extract_t::low_half;
                auto src_val = _qz_a1b0<data_type::f32, type_o>()(input[i_off]);
                const uint8_t dst_val = i == 0
                        ? 0
                        : reinterpret_cast<uint8_t *>(output)[o_off / 2];
                output[o_off / 2] = src_val.insert(dst_val, shift);
            }
        });

        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                        && tag_o == format_tag::any
                        && utils::one_of(type_i, dnnl_s4, dnnl_u4)
                        && type_o == dnnl_f32,
                spec::reference>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        return !input_d.has_runtime_dims_or_strides() && input_d.is_dense()
                && input_d.strides()[output_d.ndims() - 1] == 1
                && output_d.is_dense() && simple_attr_check(attr, false, true);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace utils;

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        // To avoid clashes between threads each byte (or 2 elements)
        // is handled by a single thread
        const dim_t work_amount = input_d.nelems() / 2;

        parallel(0, [&](const int ithr, const int nthr) {
            dim_t start {0}, end {0};
            balance211(work_amount, nthr, ithr, start, end);
            PRAGMA_OMP_SIMD()
            for_(dim_t idx = start; idx < end; idx++)
            for (int i = 0; i < 2; ++i) {
                const auto i_off = input_d.off_l(2 * idx + i);
                const auto o_off = output_d.off_l(2 * idx + i);
                const auto shift = i % 2 ? int4_extract_t::high_half
                                         : int4_extract_t::low_half;
                auto src_val = data_t<type_i>::extract(
                        reinterpret_cast<const uint8_t *>(input)[i_off / 2],
                        shift);
                reinterpret_cast<data_t<type_o> *>(output)[o_off]
                        = static_cast<float>(src_val);
            }
        });

        return status::success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                        && tag_o == format_tag::any
                        && order_keep == fmt_order::any,
                spec::direct_copy_except_dim_0>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        auto is_dense_no_0 = [](const memory_desc_wrapper &data_d) {
            return nelems_no_dim_0(data_d) == _size_no_dim_0(data_d);
        };
        return !input_d.has_runtime_dims_or_strides()
                && input_d.similar_to(output_d, true, false, 1)
                && is_dense_no_0(input_d) && is_dense_no_0(output_d)
                && simple_attr_check(attr, false, true);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();
        using namespace utils;

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const int N = input_d.dims()[0];
        const dim_t is = input_d.blocking_desc().strides[0];
        const dim_t os = output_d.blocking_desc().strides[0];
        const dim_t nelems_no_d0 = nelems_no_dim_0(input_d);
        const dim_t work_amount = N * nelems_no_d0;

        if (alpha == 1.0 && beta == 0.0) {
            parallel(0, [&](const int ithr, const int nthr) {
                dim_t n {0}, dim1_s {0};
                dim_t start {0}, end {0};
                balance211(work_amount, nthr, ithr, start, end);
                nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
                while (start < end) {
                    dim_t work_rem = end - start;
                    dim_t dim1_e = std::min(dim1_s + work_rem, nelems_no_d0);
                    PRAGMA_OMP_SIMD()
                    for (dim_t e = dim1_s; e < dim1_e; ++e) {
                        output[os * n + e]
                                = _qz_a1b0<type_i, type_o>()(input[is * n + e]);
                    }
                    nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
                }
            });
        } else {
            parallel(0, [&](const int ithr, const int nthr) {
                dim_t n {0}, dim1_s {0};
                dim_t start {0}, end {0};
                balance211(work_amount, nthr, ithr, start, end);
                nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
                while (start < end) {
                    dim_t work_rem = end - start;
                    dim_t dim1_e = std::min(dim1_s + work_rem, nelems_no_d0);
                    PRAGMA_OMP_SIMD()
                    for (dim_t e = dim1_s; e < dim1_e; ++e) {
                        output[os * n + e]
                                = _qz<type_i, type_o>()(input[is * n + e],
                                        output[os * n + e], alpha, beta);
                    }
                    nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
                }
            });
        }

        return status::success;
    }

private:
    static dim_t nelems_no_dim_0(const memory_desc_wrapper &data_d) {
        const int ndims = data_d.ndims();
        if (ndims <= 1) return 1;
        return utils::array_product(data_d.dims() + 1, data_d.ndims() - 1);
    }

    static dim_t _size_no_dim_0(const memory_desc_wrapper &data_d) {
        dims_t blocks;
        data_d.compute_blocks(blocks);

        const auto &blk = data_d.blocking_desc();

        dim_t blk_size = 1;
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk)
            blk_size *= blk.inner_blks[iblk];

        dim_t max_size = blk_size;
        for (int d = 1; d < data_d.ndims(); ++d) {
            max_size = nstl::max(max_size,
                    data_d.padded_dims()[d] / blocks[d] * blk.strides[d]);
        }

        return max_size;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
        typename utils::enable_if<tag_i == format_tag::any
                        && tag_o == format_tag::any
                        && order_keep == fmt_order::any
                        // u4/s4 requires a special implementation
                        && !utils::one_of(type_i, dnnl_s4, dnnl_u4)
                        && !utils::one_of(type_o, dnnl_s4, dnnl_u4),
                spec::reference>::type> {
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        /* supported smask: 0x0...011..10...0,
         * i.e. 1 should be contiguous */
        int src_scales_mask = -1;
        int dst_scales_mask = -1;
        CHECK(get_scales_mask(attr, &src_scales_mask, &dst_scales_mask));

        for (auto smask : {src_scales_mask, dst_scales_mask}) {
            for (; smask > 0 && !(smask & 0x1); smask >>= 1)
                ;
            for (; smask > 0 && smask & 0x1; smask >>= 1)
                ;
            if (smask != 0) return false;
        }

        using skip_mask_t = dnnl_primitive_attr::skip_mask_t;
        return input_d.is_blocking_desc() && output_d.is_blocking_desc()
                && !output_d.is_additional_buffer()
                && !input_d.is_additional_buffer()
                && attr->has_default_values(skip_mask_t::scales_runtime
                        | skip_mask_t::zero_points_runtime
                        | skip_mask_t::post_ops)
                && simple_po_check(attr);
    }

    GET_SCRATCHPAD_SIZE_ZERO();

    static status_t execute(const cpu_reorder_pd_t *pd, const exec_ctx_t &ctx) {
        DECLARE_COMMON_PARAMS();

        // This kernel is used also for tensors with multiple inner
        // blocks for which generic zero padding must be used.
        // TODO: apply zero padding inside parallel_nd()
        ctx.zero_pad_output(DNNL_ARG_TO);

        parallel_nd(D_start, D_mask, D_rest,
                [&](ptrdiff_t ds, ptrdiff_t dm, ptrdiff_t dr) {
                    const float src_scale
                            = src_scales[src_scales_mask == 0 ? 0 : dm];
                    const float dst_scale
                            = dst_scales[dst_scales_mask == 0 ? 0 : dm];

                    const size_t e = (ds * D_mask + dm) * D_rest + dr;
                    const auto &i = input[input_d.off_l(e)];
                    auto &o = output[output_d.off_l(e)];

                    float f = src_scale * ((float)i - src_zp);
                    if (beta) f += beta * o;
                    f = f * dst_scale + dst_zp;
                    o = _qz_a1b0<data_type::f32, type_o>()(f);
                });

        return status::success;
    }
};

/* high level class declaration */

template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("simple:any", simple_reorder_t);

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            using skip_mask_t = dnnl_primitive_attr::skip_mask_t;
            bool args_ok = impl::is_dense_format_kind({src_md, dst_md})
                    && src_md->data_type == type_i
                    && dst_md->data_type == type_o
                    && attr->has_default_values(skip_mask_t::scales_runtime
                            | skip_mask_t::zero_points
                            | skip_mask_t::zero_points_runtime
                            | skip_mask_t::post_ops)
                    && simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
                            spec>::is_applicable(src_md, dst_md, attr);
            if (!args_ok) return status::invalid_arguments;

            int mask = -1;
            bool is_set = false;
            CHECK(attr->scales_.get(DNNL_ARG_DST, &mask, &is_set));
            const memory_desc_wrapper input_d(src_md);
            if (input_d.has_runtime_dims_or_strides() && is_set && mask > 0)
                return status::unimplemented;

            auto _pd = make_unique_pd<pd_t>(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            CHECK(_pd->init(engine, src_engine, dst_engine));

            const size_t scratchpad_sz_
                    = simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
                            spec>::get_scratchpad_size(src_md, dst_md);
            auto scratchpad = _pd->scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_reorder_space,
                    scratchpad_sz_, 1, 16);

            if (is_set && mask > 0) {
                dim_t D_mask;
                _pd->get_D_values(input_d, mask, nullptr, &D_mask, nullptr);
                scratchpad.template book<float>(
                        memory_tracking::names::
                                key_reorder_precomputed_dst_scales,
                        D_mask);
            }

            CHECK(_pd->init_scratchpad_md());
            return safe_ptr_assign(*reorder_pd, _pd.release());
        }
        friend dnnl::impl::impl_list_item_t;
    };

    simple_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL, spec>::execute(
                pd(), ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

#undef SIMPLE_REORDER_TEMPL_DECL
#undef SIMPLE_REORDER_TEMPL_CALL

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
