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

#ifndef CPU_X64_JIT_PRIMITIVE_CONF_HPP
#define CPU_X64_JIT_PRIMITIVE_CONF_HPP

#include <queue>
#include <stdint.h>

#include "common/primitive_attr.hpp"
#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

/* convolution */
enum conv_loop_order_t {
    loop_cgn,
    loop_gnc,
    loop_ngc,
    loop_gncw,
    loop_cwgn,
    loop_ngcw,
    loop_nhwcg,
    loop_nwcg
};
enum conv_1x1_loop_order_t {
    loop_rbl,
    loop_rlb,
    loop_lbr,
    loop_lrb,
    loop_blr,
    loop_brl
};

enum conv_kernel_kind_t { embd_bcast, expl_bcast };
enum conv_harness_t {
    harness_2d_reduction,
    harness_3d_reduction,
    harness_mb_reduction,
    harness_compute_full_spatial,
    harness_nxc
};

enum {
    FLAG_MB_FIRST = 1 << 0,
    FLAG_MB_LAST = 1 << 1,
    FLAG_OC_FIRST = 1 << 2,
    FLAG_OC_LAST = 1 << 3,
    FLAG_IC_FIRST = 1 << 4,
    FLAG_IC_LAST = 1 << 5,
    FLAG_SP_FIRST = 1 << 6,
    FLAG_SP_LAST = 1 << 7,
    FLAG_REDUCE_FIRST = 1 << 8,
    FLAG_REDUCE_LAST = 1 << 9,
    FLAG_ZERO_FILTER = 1 << 0, /* Controls whether the inner kernel skips
                                   loading weights-data from memory; this
                                   needs to happen on the first Group/16
                                   iteration. */
    FLAG_ZERO_BIAS = 1 << 1, /* Controls whether the inner kernel skip
                               loading bias data from memory */
    FLAG_COMPUTE_BIAS = 1 << 2, /* Controls bias computation during execution
                                    pass */
};

enum class jit_memory_tag_kind_t { ncsp, nspc, blocked, undef };

struct jit_conv_conf_t {
    prop_kind_t prop_kind;
    bool has_vnni;
    conv_loop_order_t loop_order;
    conv_harness_t harness;

    int simd_w;
    int ndims;
    int mb;
    int ngroups, ic, oc, oc_without_padding, ic_without_padding;
    int id, ih, iw, od, oh, ow;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;
    format_tag_t src_tag, wei_tag, dst_tag; // temporary workaround
    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_binary;

    data_type_t sum_dt;

    bool with_binary_per_oc_bcast;
    bool with_binary_no_bcast;

    bool is_fused_conv;
    int dw_conv_buffer_oc;

    post_ops_t::entry_t::eltwise_t eltwise;
    post_ops_t post_ops;
    bool is_fast_postops; // maybe skip injector for sum and/or relu

    int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b, nthr_oh;

    int idp, ihp, iwp, ohp, owp, icp;
    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_iw, iw_block;
    int nb_ow, ow_block;
    int nb_oc_blocking; /* used in jit kernels for nb_oc work blocking taking
                           into account vector registers distribution */
    int nb_oc_blocking_thr_chunk; /* used for distribution of nb_oc work
                                      within threads */
    int nb_ic_blocking, nb_ic_blocking_max; // blocking of nb_ic work
    int nb_ic_L2;
    int h_blocking;
    int nb_oc_L2;
    int ic_tail, oc_tail, ch_tail;
    int ur_h, ur_w;
    int ur_w_tail, ur_w_blocks;
    int ur_ic, ur_kw;
    bool is_1stconv;
    int nonblk_group_off;
    /* fma avx512_core */
    conv_kernel_kind_t kernel_kind;

    int tr_iw, tr_ih;
    int tr_kw, tr_kh;
    int tr_src_num_guard_elems;

    // Transpose buffer management
    size_t tr_src_buf_size, tr_src_buf_count;
    size_t tr_diff_dst_buf_size, tr_diff_dst_buf_count;
    int nthr_mb_work;

    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;
    /* avx512_u8s8u8 */
    int ic_nb1, ic_nb2;
    int oc_nb1;
    int ur_ow_max, ur_ow, ur_ow_tail;
    int ur_ow_nsteps;
    data_type_t bia_dt;
    /* bf16 data-type for output */
    data_type_t dst_dt;
    data_type_t src_dt;
    /* bf16 weights update */
    data_type_t wei_dt;
    data_type_t ddst_dt;
    data_type_t dsrc_dt;
    data_type_t dwei_dt;
    bool is_bf32 {false};
    bool expl_bcast;
    bool large_spatial, large_w_filter;
    int is_ic_scale, is_oc_scale;
    int max_regs_ur; // maximum accumulation registers
    // dw conv
    int nb_ch, ch_block, nb_ch_blocking;
    bool is_depthwise, is_fast_depthwise, is_resrc_depthwise;
    int aligned_threads;
    // large spatial
    int ih_blk_size, oh_blk_size;
    // s8s8 convolution
    bool signed_input;
    bool need_saturation;
    float wei_adj_scale;
    // zero-point compensation
    bool src_zero_point;
    int zp_pbuff_size;
    bool dst_zero_point;
    bool zp_src_is_common; // common, otherwise (TODO) per-channel
    bool req_zero_point_buffer; // used for calculating padding compensation
    bool zp_pbuff_outer_compute; // indicates if zp_bbuff is computed in

    bool dst_scale;

    // a separate parallel region
    int ow_pad, oh_pad, od_pad; // output elements with padding & filter overlap

    //output elements requiring zero-point padding compensation
    int f_pad_output, back_pad_output;
    int t_pad_output, b_pad_output;
    int l_pad_output, r_pad_output;
    // The number of output blocks corresponding to {l_pad, no_pad, r_pad}
    int l_pad_blk, no_pad_w_blk, r_pad_blk;

    bool od_mid, oh_mid, ow_mid; // indicate if there is overlap between the
            //width and height padded regions

    size_t h_blk_limits[5]; // pre-computed limits for output height block

    bool uses_permw_transposition;
    bool transpose_src;
    bool transpose_dst;
    int ic_block_step;

    cpu_isa_t isa;
    // bf16 bwdw conv
    int tr_ow;
    bool is_hw_transp; // spatial dim height-width transposed
    int spatial_blk_size; // Height/depth block size inside the driver
    bool global_transpose; // diff_dst & src tensors are transposed in one go
    bool use_nt_stores_ddst; // Use non temporal stores in diff_dst transform

    // Needed for Intel(R) Advanced Matrix Extensions (Intel(R) AMX) kernels
    bool is_nspc; // activations in nwc, nhwc, or ndhwc layout
    bool is_relo; // reduced lowering optimization
    int nreduce; // used with is_relo
    bool is_pbuffer_strided; // does pbuffer have strided sectors?
    int n_stride_sets; // number of stride sectors (or sets) in pbuffer
    int kw_step; // usually stride_w, unless !is_pbuffer_strided
    int kw_per_tile; // mostly for 1st convs
    // The suffix _int refers to the block sizes of the src and diff_dst tiles,
    // as opposed to the vector registers. This distinction is needed due to
    // support for blocked layout (ie nChw16c) with bf16 data type.
    int ic_block_int, ic_block_int_np, oc_block_int;
    int nb_ic_int, nb_oc_int;
    int nb_ih_blocking, nb_oh_blocking;

    int full_tile_width;
    int max_tiles;
    int tile_width;
    int tile_tail;
    int oh_per_tile;
    int iw_blocks, ow_blocks;

    int per_one_pstore;

    size_t inp_buffer_size;
    size_t wei_buffer_size;
    size_t wsp_buffer_size;

    int nb_os;
    int nb_os_blocking;
    int nb_os2_blocking;
    int os_tail;
    int os_blocked;
    int max_width;

    bool transform_to_vnni;
};

// calculates filter size taking into account dilation
inline int calculate_extended_filter_size(int filter_size, int dilation) {
    return (filter_size - 1) * (dilation + 1) + 1;
}

inline int calculate_end_padding(int start_padding, int dst_size, int src_size,
        int spatial_stride, int dilated_filter_size) {
    return (dst_size - 1) * spatial_stride + dilated_filter_size
            - (src_size + start_padding);
}

inline status_t init_tag(format_tag_t &tag, const memory_desc_wrapper &mdw,
        const format_tag_t &tag_value) {
    if (mdw.format_kind() == format_kind::any) return status::unimplemented;

    tag = mdw.matches_one_of_tag(tag_value);
    return tag == tag_value ? status::success : status::unimplemented;
}

struct jit_conv_call_s {
    const void *src; /* hack, non-const for backward_data */
    const void *dst; /* hack, non-const for forward */
    const void *filt; /* hack, non-const for backward_weights */
    const void *bias; /* hack, non-const for backward_bias */
    const void *src_prf;
    const void *dst_prf;
    const void *filt_prf;
    const void *bias_prf;
    const void *scales;
    const void *acc_s32;
    const void *compensation;
    const int32_t *zp_compensation;
    const int32_t *src_zero_point;
    const int32_t *zero_point_pbuff;
    const int32_t *dst_zero_point;
    const void *tile_cfg;
    const void *tile_cfg_tail;
    const void *dst_scale;

    // ptr to table of void * elements that are pointers to
    // post_op binary src1 tensors
    const void *post_ops_binary_rhs_arg_vec;
    const void *dst_orig; // pointer to dst memory (no offset)

    size_t kd_offset;
    size_t kd_offset_prf;
    size_t kh_offset;
    size_t kh_offset_prf;
    size_t os_index_begin;
    size_t os_index_begin_prf;
    size_t os_index_end;
    size_t os_index_end_prf;
    size_t kd_padding;
    size_t kd_padding_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t iwb;
    size_t iwb_prf;
    size_t owb;
    size_t owb_prf;
    size_t ohb;
    size_t kw_padding;
    size_t channel;
    size_t channel_prf;
    size_t ic_blocks;
    size_t oc_blocks;
    size_t ur_w;
    size_t ur_str_w;
    size_t ch_blocks;
    size_t ch_blocks_prf;
    size_t reduce_work;
    size_t reduce_work_prf;
    size_t load_work;
    size_t load_work_prf;
    size_t l_overflow;
    size_t r_overflow;
    size_t t_overflow;
    size_t b_overflow;
    size_t f_overflow;
    size_t back_overflow;
    size_t last_h;
    size_t tail;
    size_t current_iw;
    size_t is_osb;
    int flags;
    int flags_prf;
    int oc_flag;
    size_t last_ic_block;
    size_t last_oc_block;
};

struct jit_deconv_call_s {
    const void *src; /* hack, non-const for backward_data */
    const void *dst; /* hack, non-const for forward */
    const void *filt; /* hack, non-const for backward_weights */
    const void *bias; /* hack, non-const for backward_bias */
    const void *scales;
    const void *dst_scale;
    const void *compensation;
    const int32_t *zp_src_pad_str_compensation;
    const int32_t *zp_compensation;
    const int32_t *src_zero_point;
    const int32_t *dst_zero_point;

    /*
     * ptr to table of void * elements that are pointers to post_op binary
     * src1 tensors
     */
    const void *post_ops_binary_rhs_arg_vec;
    const void *dst_orig; /* pointer to dst memory (no offset) */
    size_t t_overflow;
    size_t b_overflow;
    size_t f_overflow;
    size_t back_overflow;
    size_t kh_padding;
    size_t kd_padding;
    size_t oc_blocks;
};

struct jit_dw_conv_call_s {
    const void *input;
    const void *output;
    const void *filter;
    const void *bias;
    size_t kh_count;
    size_t oh_count;
    size_t oh_index;
    size_t filter_pad_off;
    unsigned char
            exec_flags; /* Flags passed by driver execution to inner kernel */
};

struct jit_1x1_conv_conf_t {
    prop_kind_t prop_kind;
    bool has_vnni;

    int ndims;
    int mb;
    int ngroups, ic, oc, oc_without_padding, ic_without_padding;
    int id, ih, iw, od, oh, ow;
    int f_pad, t_pad, l_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    format_tag_t src_tag, wei_tag, dst_tag; // temporary workaround
    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_binary;
    bool with_dw_conv;

    post_ops_t post_ops;

    dim_t is, os;
    int ic_block, oc_block;

    int ur, ur_tail;

    dim_t reduce_dim;
    int reduce_block, nb_reduce, nb_reduce_blocking, nb_reduce_blocking_max;
    int load_dim, load_block, nb_load, nb_load_blocking, nb_load_blocking_max,
            nb_load_chunk;
    dim_t bcast_dim;
    int bcast_block, nb_bcast, nb_bcast_blocking, nb_bcast_blocking_max;

    int reduce_loop_unroll;
    dim_t reduce_loop_bcast_step;
    int reduce_loop_load_step;
    int load_loop_load_step, load_loop_iter_step;
    int bcast_loop_output_step, bcast_loop_output_substep;
    int bcast_loop_bcast_step, bcast_loop_bcast_substep;
    int load_grp_count;
    conv_1x1_loop_order_t loop_order;
    bool use_vmovntps;
    /* avx512 core */
    bool expl_bcast;

    int typesize_in;
    int typesize_out;
    int typesize_bia;
    int typesize_acc;

    bool transpose_src;
    int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
    int is_oc_scale;
    data_type_t bia_dt;
    data_type_t dst_dt;
    data_type_t sum_dt;
    bool signed_input;
    float wei_adj_scale;
    // zero-point compensation
    bool src_zero_point;
    bool dst_zero_point;
    bool zp_src_is_common; // common, otherwise (TODO) per-channel

    bool dst_scale;

    cpu_isa_t isa;
    bool uses_permw_transposition;
};

struct jit_1x1_conv_call_s {
    const void *bcast_data;
    const void *load_data;
    const void *output_data;
    const void *bias_data; // used in forward and backward_weights only
    const void *acc_s32;
    const void *scales;
    const void *compensation;
    const void *store_buffer;
    const int32_t *zp_compensation;
    const int32_t *src_zero_point;
    const int32_t *dst_zero_point;
    const void *dst_scale;

    // ptr to table of void * elements that are pointers to
    // post_op binary src1 tensors
    const void *post_ops_binary_rhs_arg_vec;
    const void *dst_orig; // pointer to dst memory (no offset)

    size_t load_dim;
    size_t bcast_dim;
    size_t reduce_dim;

    size_t output_stride; // used in backward_weights only

    size_t first_last_flag;
};

struct jit_pool_conf_t {
    int ndims;
    int mb, c, c_without_padding;
    int id, ih, iw, od, oh, ow;
    int stride_d, stride_h, stride_w;
    int kd, kh, kw;
    int f_pad, t_pad, l_pad;
    alg_kind_t alg;
    bool is_training;
    bool pad_w_is_null;
    bool is_backward;
    bool simple_alg;
    bool is_c_padded;
    data_type_t ind_dt;

    int c_block, c_tail, nb_c;
    int ur_bc, ur_bc_tail;
    int ur_c, ur_c_tail;
    int ur;
    size_t tail[4];
    bool safe_c_tail;
    data_type_t src_dt;
    data_type_t dst_dt;

    int dt_size;
    bool is_bf16;
    bool is_f16;
    jit_memory_tag_kind_t tag_kind;
    bool is_plain() const {
        return (tag_kind == jit_memory_tag_kind_t::ncsp
                || tag_kind == jit_memory_tag_kind_t::nspc);
    }

    cpu_isa_t isa;
    post_ops_t post_ops;
    bool with_postops;
    bool with_eltwise;
    bool with_binary;
    int nthr;
    memory_desc_t tmp_md;
};

struct jit_pool_call_s {
    const void *src;
    const void *dst;
    const void *indices;
    const void *src_prf;
    const void *dst_prf;
    const void *indices_prf;
    const void *post_ops_binary_rhs_arg_vec;
    const void *dst_orig;
    const void *dst_po_helper;
    size_t zero_ih;
    size_t zero_id;
    const void *zero_ptr;
    size_t kd_padding;
    size_t kh_padding;
    size_t kh_padding_shift;
    size_t kd_padding_shift;
    size_t kw_padding;
    const void *init_value;
    float ker_area_h;
    size_t ur_bc; // contains number of channel blocks to processing
    size_t b_c; // contains number of channel blocks already processed
};

struct jit_resampling_conf_t {
    unsigned ndims = 0;

    unsigned c = 0;
    unsigned id = 0, ih = 0, iw = 0;
    unsigned od = 0, oh = 0, ow = 0;

    unsigned stride_d = 0;
    unsigned stride_h = 0;
    unsigned stride_w = 0;
    unsigned inner_stride = 0;

    // The linear algorithm is an approximation of the point
    // value based on the limit values. For one dimension,
    // the approximation is based on the line, for two
    // dimensions it will be a rectangle, and for three
    // dimensions it will be a cuboid. Therefore,
    // the possible variants for the number of corners are 2, 4, 8.
    unsigned number_of_corners = 0;

    bool is_data_size_bigger_than_L3 = false;
    bool is_saturation_needed = false;
    data_type_t src_data_type = data_type::undef;
    data_type_t dst_data_type = data_type::undef;
    size_t src_dt_size = 0;
    size_t dst_dt_size = 0;
    size_t output_data_size = 0;
    size_t el_size_of_indices = 0;

    bool is_blocked_8_format = false;
    format_tag_t src_tag = format_tag::undef;
    jit_memory_tag_kind_t tag_kind = jit_memory_tag_kind_t::undef;
    alg_kind_t alg = alg_kind::undef;

    cpu_isa_t isa = isa_undef;

    post_ops_t post_ops = post_ops_t();
    bool with_postops = false;
    bool with_eltwise = false;
    bool with_binary = false;
    bool with_sum = false;
    std::queue<float> sum_scales;
};

struct jit_resampling_call_s {
    size_t batch_of_sp_points_to_process = 0;

    const void *src = nullptr;
    const void *dst = nullptr;
    const void *indices = nullptr;
    const void *weights = nullptr;
    const void *post_ops_binary_rhs_arg_vec = nullptr;
    const void *dst_orig = nullptr;

    size_t c_offset = 0;

    size_t src_offset_top = 0;
    size_t src_offset_bottom = 0;
    size_t src_offset_front = 0;
    size_t src_offset_back = 0;

    float weight_top = 0.0f;
    float weight_bottom = 0.0f;
    float weight_front = 0.0f;
    float weight_back = 0.0f;
};

struct jit_brdgmm_conv_conf_t {

    int nthr;
    int mb, ngroups, ic, oc;
    int id, ih, iw, od, oh, ow;
    int f_pad, back_pad, l_pad, r_pad, t_pad, b_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int nb_ch, ch_block, chb_tail;
    int nb_ch_blocking;
    int ow_block, ow_tail, nb_ow;
    // idx of jit kernel when mutiple jit kernels are used in a primitive.
    int chb_tail_idx, ow_tail_idx, nb_ch_blocking_idx;
    int adjusted_batch_size;

    bool with_bias;
    bool with_post_ops;
    bool with_scale;
    bool is_oc_scale;

    data_type_t src_dt;
    data_type_t wei_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;

    brgemm_batch_kind_t batch_kind;

    size_t src_dsz;
    size_t wei_dsz;
    size_t bia_dsz;
    size_t dst_dsz;

    bool s8s8_compensation_required;
    bool src_zero_point;
    bool dst_zero_point;

    cpu_isa_t isa;
};

enum conv_brgemm_loop_order_t {
    loop_ndhwgc,
    loop_ngcdhw,
};

enum conv_brgemm_exec_type_t {
    exec_undefined = 0,
    exec_base,
    exec_trans,
    exec_vpad,
};

enum class conv_brgemm_relo_type_t : unsigned {
    undefined = 0,
    whi,
    wi,
};

struct jit_brgemm_conv_conf_t {
    cpu_isa_t isa;
    prop_kind_t prop_kind;
    conv_brgemm_loop_order_t loop_order;
    conv_harness_t harness;
    int simd_w, acc_simd_w, amx_w, amx_h;
    int ndims;
    int mb;
    int ngroups, ic, oc, oc_without_padding, ic_without_padding;

    int od_block, oh_block, nb_od,
            nb_oh; // blocking  - included in parallelization
    int id_block, ih_block, nb_id, nb_ih;
    dim_t inp_buffer_size, inp_buffer_mask_size, out_buffer_size;
    conv_brgemm_exec_type_t exec_type;

    conv_brgemm_relo_type_t relo_type {conv_brgemm_relo_type_t::undefined};
    bool relo_conv_weights {true};
    inline bool is_relo_whi() const {
        return (relo_type == conv_brgemm_relo_type_t::whi);
    }
    inline bool is_relo_wi() const {
        return (relo_type == conv_brgemm_relo_type_t::wi);
    }
    inline bool is_relo() const { return is_relo_whi() || is_relo_wi(); }

    int id, ih, iw, od, oh, ow, os, is, idp, ihp, iwp, icp, odp, ohp, owp, ocp;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int l_ovf, r_ovf, t_ovf, b_ovf, f_ovf, back_ovf;
    int kd, kh, kw;
    int ext_kd, ext_kh, ext_kw;
    int kd_block, kh_block, kw_block, kd_block_pad, kh_block_pad, kw_block_pad;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;
    format_tag_t src_tag, wei_tag, dst_tag; // temporary workaround
    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_binary;

    bool is_fused_conv;
    bool is_is_blocking;
    bool is_os_blocking;
    bool is_rtus;
    bool is_reduced_rtus;
    size_t rtus_ic_size, rtus_padded_ic_size;
    bool ununroll_bd_loop {false};
    int nb_ic, ic_block, inp_ic_block;
    int nb_tr_ic, tr_ic_block, tr_ic_tail;
    int nb_oc, oc_block;
    int nb_iw, iw_block, iw_tail;
    int nb_ow, ow_block, ow_tail;
    int nb_is, is_block;
    int nb_os, os_block;
    int nb_oc_blocking;
    int nb_ic_blocking;
    int nb_is_blocking;
    int nb_os_blocking;

    data_type_t src_dt;
    data_type_t dst_dt;
    data_type_t wei_dt;
    data_type_t acc_dt;
    data_type_t bia_dt;
    size_t src_dsz;
    size_t wei_dsz;
    size_t dst_dsz;
    size_t acc_dsz;
    size_t bia_dsz;

    bool use_buffer;
    dim_t buffer_size;
    dim_t ker_ranges_size;
    dim_t comp_a_buffer_size;
    dim_t s8s8_comp_buffer_size;

    bool with_scales;
    int is_ic_scale, is_oc_scale;

    int LDA, LDB, LDC, LDD;

    int M, N, K, M_tail, N_tail, K_tail;
    // Note: M for brgemm kernel. For use_store_mask it is usually greater than
    // M (M_tail). Otherwise it is equal to M (M_tail).
    int brgM, brgM_tail;
    int gemm_batch_size, adjusted_batch_size;
    brgemm_batch_kind_t brg_type;
    // strides for brg_type == brgemm_strd
    dim_t brg_stride_a, brg_stride_b;
    int nthr;

    int max_batch;
    int max_vpad;
    int amx_buf_size_per_thread;

    bool wei_plain;
    bool is_rd_padded_to_block {false}, is_rd_padded_to_vnni {false},
            is_oc_padded;
    bool copy_input {true};
    bool copy_block_only;
    bool amx_tile_load_xx;
    int use_M_mask;
    int oskip, iskip;
    bool brgemm_bd_loop_innermost;

    bool use_uker;
    bool var_bs {false};
    bool use_interleave_stores;
    brgemm_kernel_prefetching_t hint_prefetching;
    bool is_1x1;
    bool s8s8_compensation_required;
    bool src_zero_point;
    bool dst_zero_point;
    bool req_brg_comp_pad;
    bool req_cal_comp_pad;
    bool is_bf32;
    bool comp_with_vpads;

    int nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b, nthr_oh;
    bool transform_to_vnni;
    bool has_int8_vnni;
    float scale_adjust_factor;
    int ic_tail, oc_tail;
    size_t tr_src_block_size, tr_src_buf_size, tr_src_buf_count;
    size_t tr_diff_dst_block_size, tr_diff_dst_buf_size, tr_diff_dst_buf_count;
    int tr_src_num_guard_elems;
    bool global_transpose; // diff_dst & src tensors are transposed in one go
    int nthr_mb_work;
    int tr_iw, tr_ow;
    int spatial_blk_size; // Height/depth block size inside the driver
    int typesize_in;
    int typesize_out;
    bool tr_ocb_chunk = false;
    bool tr_icb_chunk = false;
    int vnni_block {1};
    bool has_uneven_iw;
    int trans_dim_koef {1};
};

struct jit_shuffle_conf_t {
    unsigned ndims = 0;

    dim_t mb = 0, c = 0, d = 0, h = 0, w = 0, sp = 0;

    unsigned stride_mb = 0;
    unsigned blk_size = 0;
    unsigned group_size = 0;
    unsigned axis = 0;
    unsigned axis_size = 0;
    unsigned simd_tail = 0;
    unsigned simd_w = 0;

    jit_memory_tag_kind_t tag_kind = jit_memory_tag_kind_t::undef;
    data_type_t data_type = data_type::undef;
    size_t dt_size = 0;
    unsigned el_size_of_indices = 0;
    dim_t c_split_size = 0;
    dim_t sp_split_size = 0;

    cpu_isa_t isa = isa_undef;
};

struct jit_shuffle_call_s {
    const void *src = nullptr;
    void *dst = nullptr;
    const void *input_off_ptr = nullptr;

    dim_t cb_loop_size
            = 0; // number of loop iterations over corresponding C batches
    bool is_padded_block = false;
};

enum class binary_op_t : unsigned { none, c_blocked, n_spatial_c, n_c_spatial };

enum class binary_bcast_t : unsigned {
    none, // tensor operation
    scalar,
    per_batch,
    per_c,
    per_w
};

struct jit_binary_conf_t {
    binary_op_t op_type = binary_op_t::none;
    binary_bcast_t bcast_type = binary_bcast_t::none;
    bool do_scale_src0 = false;
    bool do_scale_src1 = false;
    bool do_sum = false;
    bool with_eltwise = false;
    bool with_binary = false;
    bool with_postops = false;
    float sum_scale = 0.f;
    bool use_stride_src1 = false;
    bool broadcast_src1_value = false;
    bool use_stride_rhs_postops = false;
    bool postops_per_oc_broadcast_exists = false;
    bool is_i8 = false;
    bool is_bf16 = false;
    bool is_f16 = false;
    bool is_src_different_layouts = false;
    dim_t outer_dims = 1;
    int src1_stride = 1;
    int not_bcasted_sp_dims = 0;
    cpu_isa_t isa = isa_undef;

    data_type_t src0_type = data_type::undef;
    data_type_t src1_type = data_type::undef;
    data_type_t dst_type = data_type::undef;
};

struct jit_binary_call_s {
    // keep all sizes at 8 bytes -- jit code expects this
    const void *src0, *src1, *dst, *indices;
    const float *scales_src0, *scales_src1;
    size_t spat_offt_count;
    const void *post_ops_binary_rhs_arg_vec;
    size_t src1_stride_range;
    const void *dst_orig;
};

struct jit_reduction_conf_t {
    data_type_t src_type = data_type::undef;
    data_type_t dst_type = data_type::undef;
    data_type_t acc_type = data_type::undef;

    std::size_t src_dt_size = 0;
    std::size_t dst_dt_size = 0;
    std::size_t acc_dt_size = 0;

    alg_kind_t alg = alg_kind::undef;
    cpu_isa_t isa = isa_undef;

    dim_t idle_size = 0;
    dim_t reduce_size = 0;

    bool is_saturation_needed = false;

    post_ops_t post_ops = post_ops_t();
    bool with_postops = false;
    bool with_eltwise = false;
    bool with_binary = false;
    bool with_sum = false;
    std::queue<float> sum_scales;
};

struct jit_reduction_call_s {
    const void *src = nullptr;
    void *dst = nullptr;
    const void *post_ops_binary_rhs_arg_vec = nullptr;
    const void *dst_orig = nullptr;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
