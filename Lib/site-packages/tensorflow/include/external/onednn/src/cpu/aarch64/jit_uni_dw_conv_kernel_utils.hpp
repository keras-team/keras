/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
* Copyright 2021-2022 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_DW_CONV_KERNEL_UTILS_HPP
#define CPU_AARCH64_JIT_UNI_DW_CONV_KERNEL_UTILS_HPP

#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"
#include "cpu/aarch64/jit_uni_dw_conv_kernel_f32.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_dw_conv_fwd_kernel {

    jit_uni_dw_conv_fwd_kernel(jit_conv_conf_t ajcp) : ker_(nullptr) {
        ker_ = new jit_uni_dw_conv_fwd_kernel_f32<isa>(ajcp);
    }
    status_t create_kernel() { return ker_->create_kernel(); }
    ~jit_uni_dw_conv_fwd_kernel() { delete ker_; }

    static bool post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &weights_md, memory_desc_t &bias_md,
            memory_desc_t &dst_md, const primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_generator *ker() const { return ker_; }
    void operator()(const jit_conv_call_s *p) const { (*ker_)(p); }

private:
    jit_uni_dw_conv_fwd_kernel_f32<isa> *ker_;
};

template <cpu_isa_t isa, data_type_t kernel_dt>
bool jit_uni_dw_conv_fwd_kernel<isa, kernel_dt>::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) {
        return p.entry_[idx].is_eltwise()
                && eltwise_injector::is_supported(
                        isa, p.entry_[idx].eltwise.alg);
    };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len()) {
        case 0: return true; // no post_ops
        case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
        case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
        default: return false;
    }

    return false;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
status_t jit_uni_dw_conv_fwd_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md,
        memory_desc_t &bias_md, memory_desc_t &dst_md,
        const primitive_attr_t &attr) {

    using namespace dnnl::impl::format_tag;
    using namespace dnnl::impl::utils;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const int ndims = src_d.ndims();
    // Currently this kernel only supports 2D convolutions.
    if (ndims != 4) return status::unimplemented;

    const auto blocked_tag = isa == sve_512 ? nChw16c : nChw8c;
    const auto wei_tag = isa == sve_512 ? Goihw16g : Goihw8g;
    const auto nxc_tag = nhwc;
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    if ((blocked_tag != nChw16c) || (wei_tag != Goihw16g))
        return status::unimplemented;

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, blocked_tag));
        jcp.src_tag = blocked_tag;
    } else {
        jcp.src_tag = src_d.matches_one_of_tag(blocked_tag, nxc_tag);
    }

    if (weights_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_md, wei_tag));
        jcp.wei_tag = wei_tag;
    } else {
        jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
    }

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, blocked_tag));
        jcp.dst_tag = blocked_tag;
    } else {
        jcp.dst_tag = dst_d.matches_one_of_tag(blocked_tag, nxc_tag);
    }

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));
    }

    if (jcp.dst_tag != jcp.src_tag) return status::unimplemented;
    const auto data_tag = jcp.src_tag;
    const bool is_data_layout_nxc = data_tag == nxc_tag;

    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.isa = isa;

    if (!mayiuse(isa)) return status::unimplemented;

    const int simd_w = isa == sve_512 ? 16 : 8;
    if (simd_w != 16) return status::unimplemented;

    jcp.prop_kind = cd.prop_kind;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1];
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1];

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];

    jcp.kh = weights_d.dims()[3];
    jcp.kw = weights_d.dims()[4];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.b_pad = cd.padding[1][0];
    jcp.r_pad = cd.padding[1][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad
            || ext_kh <= jcp.b_pad;
    if (kernel_outside_src) return status::unimplemented;

    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_in = types::data_type_size(src_d.data_type());

    jcp.loop_order = loop_ngcw;

    jcp.ur_w = isa == sve_512 ? 6 : isa == sve_256 ? 4 : 3;
    jcp.ur_w = nstl::min(jcp.ur_w, jcp.ow);

    jcp.ch_block = simd_w;
    jcp.nb_ch = div_up(jcp.oc, jcp.ch_block);
    jcp.nb_ch_blocking = isa == sve_512 ? 4 : isa == sve_256 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking) jcp.nb_ch_blocking = jcp.nb_ch;

    if (is_data_layout_nxc) {
        jcp.loop_order = loop_nhwcg;
        bool cache_aliasing
                = (jcp.ngroups * jcp.iw * jcp.typesize_in) % 1024 == 0;
        if (cache_aliasing) {
            // currently only tuned for mobilenet-v1 shapes
            const int limit = jcp.ow > 7 ? 7 : 4;
            jcp.ur_w = nstl::min(jcp.ur_w, limit);
        }
    } else {
        const size_t max_ch_off
                = static_cast<size_t>(jcp.nb_ch_blocking - 1) * jcp.ch_block;
        constexpr size_t max_ex_off = 0;

        // check that input offsets fit into s32
        const size_t max_ic_off = max_ch_off * jcp.ih * jcp.iw;
        const size_t max_iw_idx
                = static_cast<size_t>(jcp.ur_w - 1) * jcp.stride_w
                + (ext_kw - 1);
        const size_t max_iw_off = max_iw_idx * jcp.ch_block;
        const size_t max_input_offset
                = (max_ic_off + max_iw_off + max_ex_off) * jcp.typesize_in;
        if (max_input_offset > INT_MAX) return status::unimplemented;

        // check that output offsets fit into s32
        const size_t max_oc_off = max_ch_off * jcp.oh * jcp.ow;
        const size_t max_ow_off
                = static_cast<size_t>(jcp.ur_w - 1) * jcp.ch_block;
        const size_t max_output_offset
                = (max_oc_off + max_ow_off + max_ex_off) * jcp.typesize_out;
        if (max_output_offset > INT_MAX) return status::unimplemented;
    }

    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    int r_pad_no_tail = nstl::max(0,
            calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                    jcp.stride_w, ext_kw));
    if (jcp.l_pad > jcp.ur_w || r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;

    if (jcp.with_eltwise) {
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (jcp.eltwise.alg == alg_kind::eltwise_pow)
            return status::unimplemented;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
    }
    bool ok_to_pad_channels = true && jcp.oc == jcp.ngroups
            && jcp.ic == jcp.ngroups && isa == sve_512;
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.oc, simd_w);
        jcp.ngroups = rnd_up(jcp.ngroups, simd_w);
    }

    bool args_ok = true && jcp.oc == jcp.ngroups && jcp.ic == jcp.ngroups
            && jcp.ngroups % simd_w == 0 && jcp.wei_tag == wei_tag
            && data_tag != format_tag::undef && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1]
            && jcp.ngroups <= weights_d.padded_dims()[0];
    if (!args_ok) return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_fwd_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    using namespace dnnl::impl::memory_tracking::names;
    if (jcp.with_bias && jcp.oc_without_padding != jcp.oc)
        scratchpad.book<float>(key_conv_padded_bias, jcp.oc);
}

template struct jit_uni_dw_conv_fwd_kernel<sve_512, data_type::f32>;

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_dw_conv_bwd_data_kernel {

    jit_uni_dw_conv_bwd_data_kernel(jit_conv_conf_t ajcp) : ker_(nullptr) {
        ker_ = new jit_uni_dw_conv_bwd_data_kernel_f32<isa>(ajcp);
    }

    status_t create_kernel() { return ker_->create_kernel(); }
    ~jit_uni_dw_conv_bwd_data_kernel() { delete ker_; }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void operator()(const jit_conv_call_s *p) const { (*ker_)(p); }

private:
    jit_uni_dw_conv_bwd_data_kernel_f32<isa> *ker_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_uni_dw_conv_bwd_data_kernel);
};

template <cpu_isa_t isa, data_type_t kernel_dt>
status_t jit_uni_dw_conv_bwd_data_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d) {
    using namespace dnnl::impl::format_tag;
    using namespace dnnl::impl::utils;

    jcp.dsrc_dt = cd.diff_src_desc.data_type;
    jcp.isa = isa;

    if (!mayiuse(isa)) return status::unimplemented;

    const int simd_w = isa == sve_512 ? 16 : 8;

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    if (!with_groups) return status::unimplemented;

    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1];
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1];

    jcp.ih = diff_src_d.dims()[2];
    jcp.iw = diff_src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = weights_d.dims()[3];
    jcp.kw = weights_d.dims()[4];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.b_pad = cd.padding[1][0];
    jcp.r_pad = cd.padding[1][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];
    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad
            || ext_kh <= jcp.b_pad;
    if (kernel_outside_src) { return status::unimplemented; }

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    bool ok_to_pad_channels = true && jcp.oc == jcp.ngroups
            && jcp.ic == jcp.ngroups && isa == sve_512;
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.oc, simd_w);
        jcp.ngroups = rnd_up(jcp.ngroups, simd_w);
    }

    auto dat_tag = isa == sve_512 ? nChw16c : nChw8c;
    auto wei_tag = isa == sve_512 ? Goihw16g : Goihw8g;

    jcp.src_tag = diff_src_d.matches_one_of_tag(dat_tag);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
    jcp.dst_tag = diff_dst_d.matches_one_of_tag(dat_tag);

    bool args_ok = true && jcp.oc == jcp.ngroups && jcp.ic == jcp.ngroups
            && jcp.ngroups % simd_w == 0 && jcp.src_tag == dat_tag
            && jcp.wei_tag == wei_tag && jcp.dst_tag == dat_tag
            && jcp.dilate_h == 0 && jcp.dilate_w == 0
            && jcp.oh == (jcp.ihp - jcp.kh) / jcp.stride_h + 1
            && jcp.ow == (jcp.iwp - jcp.kw) / jcp.stride_w + 1
            && jcp.ic <= diff_src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ngroups <= weights_d.padded_dims()[0];
    if (!args_ok) return status::unimplemented;

    jcp.typesize_out = types::data_type_size(diff_src_d.data_type());
    jcp.typesize_in = types::data_type_size(diff_dst_d.data_type());

    jcp.ur_w = isa == sve_512 ? 6 : isa == sve_256 ? 4 : 3;

    jcp.ch_block = simd_w;
    jcp.nb_ch = jcp.ic / jcp.ch_block;
    jcp.nb_ch_blocking = isa == sve_512 ? 4 : isa == sve_256 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking) jcp.nb_ch_blocking = jcp.nb_ch;

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_bwd_data_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

template struct jit_uni_dw_conv_bwd_data_kernel<sve_512, data_type::f32>;

template <cpu_isa_t isa, data_type_t kernel_dt>
struct jit_uni_dw_conv_bwd_weights_kernel {

    jit_uni_dw_conv_bwd_weights_kernel(jit_conv_conf_t ajcp) : ker_(nullptr) {
        ker_ = new jit_uni_dw_conv_bwd_weights_kernel_f32<isa>(ajcp);
    }

    status_t create_kernel() { return ker_->create_kernel(); }
    ~jit_uni_dw_conv_bwd_weights_kernel() { delete ker_; }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &diff_weights_d,
            const memory_desc_wrapper &diff_dst_d, int nthreads);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    static void balance(jit_conv_conf_t &jcp, int nthreads);

    void operator()(const jit_dw_conv_call_s *p) const { (*ker_)(p); }

private:
    jit_uni_dw_conv_bwd_weights_kernel_f32<isa> *ker_;
};

template <cpu_isa_t isa, data_type_t kernel_dt>
status_t jit_uni_dw_conv_bwd_weights_kernel<isa, kernel_dt>::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &diff_weights_d,
        const memory_desc_wrapper &diff_dst_d, int nthreads) {
    using namespace dnnl::impl::format_tag;
    using namespace dnnl::impl::utils;

    jcp.dwei_dt = cd.diff_weights_desc.data_type;
    jcp.isa = isa;

    if (!mayiuse(isa)) return status::unimplemented;

    jcp.ngroups = diff_weights_d.dims()[0];
    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;

    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.oc, jcp.ic);

    if (!jcp.is_depthwise) return status::unimplemented;

    jcp.ch_block = isa == sve_512 ? 16 : 8;
    jcp.simd_w = jcp.ch_block;

    jcp.mb = src_d.dims()[0];

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = diff_weights_d.dims()[3];
    jcp.kw = diff_weights_d.dims()[4];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.t_pad = cd.padding[0][0];
    jcp.b_pad = cd.padding[1][0];

    jcp.l_pad = cd.padding[0][1];
    jcp.r_pad = cd.padding[1][1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;

    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;

    auto dat_tag = isa == sve_512 ? nChw16c : nChw8c;
    auto wei_tag = isa == sve_512 ? Goihw16g : Goihw8g;

    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.wei_tag = diff_weights_d.matches_one_of_tag(wei_tag);
    jcp.dst_tag = diff_dst_d.matches_one_of_tag(dat_tag);

    bool args_ok = true && jcp.src_tag == dat_tag && jcp.wei_tag == wei_tag
            && jcp.dst_tag == dat_tag && jcp.ngroups % jcp.ch_block == 0
            && jcp.dilate_h == 0 && jcp.dilate_w == 0 && jcp.kw <= 3
            && jcp.stride_w <= jcp.kw // no gaps in kernel
            && jcp.oh == (jcp.ihp - jcp.kh) / jcp.stride_h + 1
            && jcp.ow == (jcp.iwp - jcp.kw) / jcp.stride_w + 1;
    if (!args_ok) return status::unimplemented;

    jcp.nb_ch = jcp.ngroups / jcp.ch_block;

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_hpad = (jcp.kh - 1 + 1) / 2;
    const int max_wpad = (jcp.kw - 1 + 1) / 2;
    const int min_ih = jcp.kh + nstl::modulo(-jcp.t_pad, jcp.stride_h);
    const bool boundaries_ok = true && jcp.t_pad <= max_hpad
            && jcp.b_pad <= max_hpad && jcp.l_pad <= max_wpad
            && jcp.r_pad <= max_wpad
            // input must fully accommodate the filter
            && jcp.ih >= min_ih
            // non-unit padding must be a multiple of the stride
            && IMPLICATION(jcp.t_pad > 1, jcp.t_pad % jcp.stride_h == 0)
            && IMPLICATION(jcp.b_pad > 1, jcp.b_pad % jcp.stride_h == 0);
    if (!boundaries_ok) return status::unimplemented;

    /* BF16: accumulation of output happens in f32, down-conversion to bf16
     * happens during the reduction phase. */
    jcp.typesize_out = sizeof(float);
    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.bia_dt = jcp.with_bias ? cd.diff_bias_desc.data_type : data_type::undef;

    balance(jcp, nthreads);

    return status::success;
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_bwd_weights_kernel<isa, kernel_dt>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    using namespace dnnl::impl::memory_tracking::names;
    /* Notes: if splitting thread work on 'mb', then a reduction has to take
     * place. Hence, book a per-thread, local weights-buffer for the
     * reduction */
    if (jcp.nthr_mb > 1) {
        const size_t mb = jcp.dwei_dt == data_type::bf16 ? jcp.nthr_mb
                                                         : jcp.nthr_mb - 1;
        const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
        scratchpad.book<float>(key_conv_wei_reduction, wei_size * mb);

        if (jcp.with_bias)
            scratchpad.book<float>(
                    key_conv_bia_reduction, jcp.ngroups * (jcp.nthr_mb - 1));
    } else if (jcp.nthr_mb == 1 && jcp.dwei_dt == data_type::bf16) {
        const size_t wei_size = jcp.ngroups * jcp.kh * jcp.kw;
        scratchpad.book<float>(key_conv_wei_reduction, wei_size);
    }
    if (jcp.bia_dt == data_type::bf16)
        scratchpad.book<float>(key_conv_bias_bf16_convert_wsp, jcp.ngroups);
}

template <cpu_isa_t isa, data_type_t kernel_dt>
void jit_uni_dw_conv_bwd_weights_kernel<isa, kernel_dt>::balance(
        jit_conv_conf_t &jcp, int nthreads) {
    jcp.nthr = nthreads;
    jcp.nthr_g = jcp.nthr_mb = 1;

    /* Basic-Heuristics for parallel strategy:
     * 1) Tries to parallel on the number of Groups (g) where tasks are
     * independent. Otherwise,
     * 2) Tries to split the work across g and MiniBatch (mb).
     * Parallelizing on mb requires computing a reduction for weights.
     *
     * NOTE: because of 'task partitioning' scheme, there will be unbalanced
     * per-thread load when the number of threads is high (e.g. > 16).
     */
    jcp.nthr_g = nstl::min(jcp.nb_ch, jcp.nthr);
    jcp.nthr_mb = nstl::min(nstl::max(1, jcp.nthr / jcp.nthr_g), jcp.mb);

    jcp.nthr = jcp.nthr_g * jcp.nthr_mb;
}

template struct jit_uni_dw_conv_bwd_weights_kernel<sve_512, data_type::f32>;
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif /* CPU_X64_JIT_UNI_DW_CONV_KERNEL_UTILS_HPP */
