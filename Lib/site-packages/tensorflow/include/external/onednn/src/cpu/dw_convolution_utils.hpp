/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef CPU_DW_CONVOLUTION_UTILS_HPP
#define CPU_DW_CONVOLUTION_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

inline status_t get_depthwise_conv_desc(convolution_desc_t &cd_dw,
        const memory_desc_t &src_dw_md, const primitive_attr_t &attr_1x1,
        primitive_attr_t &attr_dw, int dw_po_index) {

    const memory_desc_wrapper src_dw_d(src_dw_md);
    const int ndims = src_dw_d.ndims();
    if (ndims != 4) return status::unimplemented;

    if (dw_po_index == -1 || dw_po_index >= attr_1x1.post_ops_.len()
            || !attr_1x1.post_ops_.entry_[dw_po_index].is_convolution())
        return status::invalid_arguments;

    // Create new attributes with scales from depthwise post-op and copy
    // post-ops after depthwise post-op.
    auto &dw_po = attr_1x1.post_ops_.entry_[dw_po_index].depthwise_conv;

    // erase 1x1 conv scales
    for (auto arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
        auto &scale = attr_dw.scales_.get(arg);
        if (!scale.has_default_values()) attr_dw.scales_.reset(arg);
    }

    const auto &dw_src_scales = attr_1x1.scales_.get(DNNL_ARG_DST);
    const auto &dw_wei_scales
            = attr_1x1.scales_.get(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
    const auto &dw_dst_scales
            = attr_1x1.scales_.get(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_DST);
    if (!dw_src_scales.has_default_values())
        attr_dw.scales_.set(DNNL_ARG_SRC, dw_src_scales.mask_);
    if (!dw_wei_scales.has_default_values())
        attr_dw.scales_.set(DNNL_ARG_WEIGHTS, dw_wei_scales.mask_);
    if (!dw_dst_scales.has_default_values())
        attr_dw.scales_.set(DNNL_ARG_DST, dw_dst_scales.mask_);

    auto dw_po_len = attr_1x1.post_ops_.len() - (dw_po_index + 1);
    attr_dw.post_ops_.entry_.resize(dw_po_len);
    for (int i = 0; i < dw_po_len; ++i) {
        attr_dw.post_ops_.entry_[i]
                = attr_1x1.post_ops_.entry_[i + dw_po_index + 1];
    }

    attr_dw.scratchpad_mode_ = attr_1x1.scratchpad_mode_;

    const bool with_bias = dw_po.bias_dt != data_type::undef;

    const auto n = src_dw_d.dims()[0];
    const auto oc = src_dw_d.dims()[1];
    const auto g = src_dw_d.dims()[1];
    const auto ih = src_dw_d.dims()[ndims - 2];
    const auto iw = src_dw_d.dims()[ndims - 1];
    const auto kernel = dw_po.kernel;
    const auto stride = dw_po.stride;
    const auto padding = dw_po.padding;

    const dims_t weights_tz = {g, 1, 1, kernel, kernel};

    // Not following standard convolution formula for output shapes since
    // right/top padding might be greated than left/top one.
    const dim_t oh = utils::div_up(ih, stride);
    const dim_t ow = utils::div_up(iw, stride);
    const dims_t dst_tz = {n, oc, oh, ow};

    const dims_t bias_tz = {oc};
    const dims_t pad_tz = {padding, padding};
    const dims_t stride_tz = {stride, stride};

    const dim_t pad_h_r = (oh - 1) * stride - ih + kernel - padding;
    const dim_t pad_w_r = (ow - 1) * stride - iw + kernel - padding;
    const dims_t pad_r_tz = {pad_h_r, pad_w_r};

    memory_desc_t src_md, weights_md, bias_md, dst_md;

    const auto src_dw_tag = src_dw_d.matches_one_of_tag(
            format_tag::nChw16c, format_tag::nChw8c, format_tag::nhwc);
    const auto data_tag
            = (src_dw_tag == format_tag::undef) ? format_tag::any : src_dw_tag;

    memory_desc_init_by_tag(
            src_md, ndims, src_dw_md.dims, src_dw_md.data_type, data_tag);

    memory_desc_init_by_tag(
            weights_md, ndims + 1, weights_tz, dw_po.wei_dt, format_tag::any);

    if (with_bias)
        memory_desc_init_by_tag(
                bias_md, 1, bias_tz, dw_po.bias_dt, format_tag::a);

    memory_desc_init_by_tag(dst_md, ndims, dst_tz, dw_po.dst_dt, data_tag);

    CHECK(conv_desc_init(&cd_dw, prop_kind::forward_inference,
            alg_kind::convolution_auto, &src_md, &weights_md,
            with_bias ? &bias_md : nullptr, &dst_md, stride_tz, nullptr, pad_tz,
            pad_r_tz));

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
