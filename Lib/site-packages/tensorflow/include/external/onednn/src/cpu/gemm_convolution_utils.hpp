/*******************************************************************************
* Copyright 2016-2022 Intel Corporation
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

#ifndef CPU_GEMM_CONVOLUTION_UTILS_HPP
#define CPU_GEMM_CONVOLUTION_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_engine.hpp"
#include "cpu/zero_point_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

enum conv_gemm_loop_order_t { gemm_loop_rlb, gemm_loop_lrb, gemm_loop_lbr };
struct conv_gemm_conf_t {
    prop_kind_t prop_kind;

    dim_t mb;
    dim_t ngroups, ic, oc;
    dim_t iw, ih, id, ow, oh, od;
    dim_t l_pad, t_pad, f_pad, e_pad, b_pad, r_pad;
    dim_t kh, kw, kd;
    dim_t stride_h, stride_w, stride_d;
    dim_t dilate_h, dilate_w, dilate_d;
    bool with_bias;
    bool with_eltwise;
    bool with_binary;
    bool with_sum;
    post_ops_t post_ops;
    bool is_nspc;

    dim_t is, os, ks;
    dim_t ic_block, oc_block;

    int nthr;
    ptrdiff_t im2col_sz;
    bool need_wei_reduction;
    bool signed_input;
    dim_t oh_block;
    dim_t ow_block;
    dim_t os_block, os_nb_block;
    bool outer_threading;
    conv_gemm_loop_order_t loop_order;
    int nthr_oc;

    zero_point_config_t zp;

    data_type_t bias_data_type;
    data_type_t dst_data_type;
    data_type_t sum_data_type;
    size_t dst_os_stride;
    size_t scale_idx_mult;
    bool with_dst_scale;
};

struct single_gemm_conv_chunk_desc_t {
    single_gemm_conv_chunk_desc_t() = default;
    single_gemm_conv_chunk_desc_t(dim_t d_off, dim_t d_size, dim_t h_off,
            dim_t h_size, dim_t w_off, dim_t w_size);

    dim_t d_off_ = 0;
    dim_t d_size_ = 0;
    dim_t h_off_ = 0;
    dim_t h_size_ = 0;
    dim_t w_off_ = 0;
    dim_t w_size_ = 0;
};

namespace jit_gemm_convolution_utils {
template <typename data_type_t>
void im2col_3d(const conv_gemm_conf_t &jcp, const data_type_t *im,
        data_type_t *col, dim_t od, int spatial_step, int spatial_block);

template <typename T>
void transpose_dt(const conv_gemm_conf_t &jcp, const T *__restrict im,
        T *__restrict imtr);

template <typename im_dt, typename col_dt>
void im2col_dt_3d(const conv_gemm_conf_t &jcp, const void *__restrict im,
        col_dt *__restrict col, dim_t od);

template <typename data_type_t>
void im2col(const conv_gemm_conf_t &jcp, const data_type_t *__restrict im,
        data_type_t *__restrict col, dim_t ss, dim_t sb, dim_t cs, dim_t cb);

template <typename im_dt, typename col_dt>
void im2col_dt(const conv_gemm_conf_t &jcp, const void *__restrict im,
        void *__restrict imtr, col_dt *__restrict col, dim_t hs, dim_t hb,
        dim_t ws, dim_t wb);

template <typename T>
void col2im_dt(
        const conv_gemm_conf_t &jcp, const T *__restrict col, T *__restrict im);
void col2im_3d(const conv_gemm_conf_t &jcp, const float *col, float *im,
        dim_t od, int spatial_step, int spatial_block);
void col2im(const conv_gemm_conf_t &jcp, const float *col, float *im,
        int spatial_step, int spatial_block);

status_t init_conf(conv_gemm_conf_t &jcp,
        memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int max_threads);

void bwd_weights_balance(int ithr, int nthr, int ngroups, int mb, int &ithr_g,
        int &nthr_g, int &ithr_mb, int &nthr_mb);
void bwd_weights_reduction_par_ncsp(int ithr, int nthr,
        const conv_gemm_conf_t &jcp, const float *weights_reduce_ws,
        float *weights);
void bwd_weights_reduction_par_nspc(int ithr, int nthr, size_t g_start,
        size_t g_end, const conv_gemm_conf_t &jcp,
        const float *weights_reduce_base, float *diff_weights);

bool padding_exists(const conv_gemm_conf_t &jcp) noexcept;

} // namespace jit_gemm_convolution_utils

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
