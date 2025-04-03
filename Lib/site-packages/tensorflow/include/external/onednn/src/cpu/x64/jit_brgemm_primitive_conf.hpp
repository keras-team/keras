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

#ifndef CPU_X64_JIT_BRGEMM_PRIMITIVE_CONF_HPP
#define CPU_X64_JIT_BRGEMM_PRIMITIVE_CONF_HPP

#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_brgemm_primitive_conf_t {
    prop_kind_t prop_kind;
    conv_harness_t harness;
    int simd_w;
    int ndims;
    int mb, os;
    int ngroups, ic, oc, oc_without_padding, ic_without_padding;
    int id = 1, ih = 1, iw = 1;
    int od = 1, oh = 1, ow = 1;
    int kd = 1, kh = 1, kw = 1;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;
    format_tag_t src_tag, wei_tag, dst_tag; // temporary workaround
    bool is_wei_layout_any;
    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_binary;
    bool with_scales;
    bool req_s8s8_compensation;
    int nb_ic, ic_block, ic_block_ext;
    int nb_oc, oc_block, oc_block_ext;
    int nb_iw, iw_block;
    int nb_ow, ow_block;
    int nb_os, os_block;
    int nb_oc_blocking;
    int nb_ic_blocking;
    int nb_os_blocking;

    data_type_t src_dt;
    data_type_t dst_dt;
    data_type_t wei_dt;
    data_type_t acc_dt;
    data_type_t bia_dt;

    bool is_amx;
    bool use_buffer;
    bool use_buffer_a;
    bool use_buffer_b;
    bool is_bf32;

    int is_oc_scale;

    int LDA, LDB, LDC, LDD;
    int M, N, K, M_tail, N_tail, K_tail;
    int gemm_batch_size, adjusted_batch_size;
    brgemm_batch_kind_t brg_type;
    int num_gemm_kernels;
    int nthr, nthr_mb, nthr_oc_b, nthr_ic_b;

    cpu_isa_t isa;
    bool use_uker;
    bool use_interleave_stores;
    int amx_buf_size_per_thread;
    brgemm_kernel_prefetching_t hint_prefetching
            = brgemm_kernel_prefetching_t::brgemm_prf_default;
    bool with_dst_scales;

    // Compute kernel-spatial dimension size.
    int ks() const;

    // Compute foward weights oc-block.
    int get_weights_oc_block() const;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
